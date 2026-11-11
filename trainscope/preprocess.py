"""
TrainScope · 数据预处理与特征工程
=================================
输入 data/runs/<run>/*.jsonl 原始日志，输出 data/processed/ 下的清洗后表格与降维结果。

流程：
 1. 解析与类型规整：'nan'/'inf' 字符串 → 浮点非有限值，标记发散；
 2. 去重：断点续训造成的重复 step，保留最后一次写入（续训后的真实轨迹）；
 3. 缺口检测：step 不连续处记录在质量报告中；
 4. 计时污染修正：探针/评估步之后一步的 step_time 含评估开销，标记并在吞吐统计中剔除；
    首步含 CUDA 初始化开销，同样剔除；
 5. 平滑：EMA(α=0.02) 与滚动中位数(窗口 51)；
 6. 特征工程：逐步 14 维模块梯度对数范数、损失残差、滚动波动率、更新比等；
 7. 维度规约：窗口聚合后的模块梯度向量 → PCA（SVD 实现）；字符嵌入 → PCA + t-SNE（numpy 实现）。
"""
import glob, json, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "data", "runs")
OUT = os.path.join(ROOT, "data", "processed")


def _num(v):
    if isinstance(v, str):
        return float(v)  # 'nan' / 'inf' / '-inf'
    return v


def read_jsonl(path):
    rows, bad = [], 0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1  # 崩溃时可能写出半行
    return rows, bad


def load_run(run_dir):
    name = os.path.basename(run_dir)
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    q = {"run": name}
    sc, q["bad_lines_scalars"] = read_jsonl(os.path.join(run_dir, "scalars.jsonl"))
    ly, q["bad_lines_layers"] = read_jsonl(os.path.join(run_dir, "layers.jsonl"))
    pr, _ = read_jsonl(os.path.join(run_dir, "probes.jsonl"))
    ev, _ = read_jsonl(os.path.join(run_dir, "evals.jsonl"))

    s = pd.DataFrame(sc)
    s["loss"] = s["loss"].map(_num).astype(float)
    q["raw_rows"] = len(s)
    q["dup_steps"] = int(s["step"].duplicated(keep="last").sum())
    s = s.drop_duplicates("step", keep="last").sort_values("step").reset_index(drop=True)
    full = np.arange(s["step"].min(), s["step"].max() + 1)
    q["missing_steps"] = int(len(set(full) - set(s["step"])))
    q["nonfinite_loss"] = int((~np.isfinite(s["loss"])).sum())
    q["first_nonfinite_step"] = int(s.loc[~np.isfinite(s["loss"]), "step"].min()) if q["nonfinite_loss"] else None

    # 计时污染：step % 50 == 1 的步包含了上一步的探针/评估开销；首步含 CUDA 初始化
    s["timing_contaminated"] = (s["step"] % 50 == 1) | (s["step"] == s["step"].min())
    # 续训后第一步的 wall 时间跳变同样标记
    s["timing_contaminated"] |= s["step_time"] > 20 * s["step_time"].median()
    q["timing_contaminated"] = int(s["timing_contaminated"].sum())

    # 模块级
    l = pd.DataFrame(ly).drop_duplicates("step", keep="last").sort_values("step")
    g = pd.json_normalize(l["gnorm"]).add_prefix("g_"); g.index = l.index
    lw = pd.concat([l[["step"]], g], axis=1)
    if "wnorm" in l:
        sub = l.dropna(subset=["wnorm"])
        w = pd.json_normalize(sub["wnorm"]).add_prefix("w_"); w.index = sub.index
        u = pd.json_normalize(sub["update_ratio"]).add_prefix("u_"); u.index = sub.index
        lw = lw.join(w).join(u)
    s = s.merge(lw, on="step", how="left")

    p = pd.DataFrame(pr).drop_duplicates("step", keep="last").sort_values("step").reset_index(drop=True)
    e = pd.DataFrame(ev).drop_duplicates("step", keep="last").sort_values("step").reset_index(drop=True)
    s["run"] = name; e["run"] = name; p["run"] = name
    return cfg, s, e, p, q


def add_features(s):
    """逐步特征：平滑、残差、波动率、梯度尖峰强度"""
    fin = s["loss"].where(np.isfinite(s["loss"]))
    s["loss_ema"] = fin.ewm(alpha=0.02, ignore_na=True).mean()
    s["loss_med"] = fin.rolling(51, center=True, min_periods=5).median()
    s["loss_resid"] = fin - s["loss_med"]
    s["loss_vol"] = fin.diff().rolling(51, min_periods=5).std()
    lg = np.log10(s["grad_norm"].where(np.isfinite(s["grad_norm"])).clip(lower=1e-8))
    s["log_gnorm"] = lg
    med = lg.rolling(101, min_periods=10).median()
    mad = (lg - med).abs().rolling(101, min_periods=10).median()
    s["gnorm_robust_z"] = (lg - med) / (1.4826 * mad + 1e-6)
    return s


def run_summary(cfg, s, e, q):
    ef = e[np.isfinite(e["val_loss"])]
    last = ef.iloc[-1] if len(ef) else None
    best = ef.loc[ef["val_loss"].idxmin()] if len(ef) else None
    thr = s.loc[~s["timing_contaminated"], "tok_per_s"]
    return {
        "run": cfg["run"], "lr": cfg["lr"], "warmup": cfg["warmup"], "clip": cfg["clip"], "batch": cfg["batch"],
        "dropout": cfg["dropout"], "layers": cfg["layers"], "d": cfg["d"], "params_M": cfg["n_params"] / 1e6,
        "final_train": float(last["train_loss"]) if last is not None else np.nan,
        "final_val": float(last["val_loss"]) if last is not None else np.nan,
        "best_val": float(best["val_loss"]) if best is not None else np.nan,
        "best_step": int(best["step"]) if best is not None else -1,
        "gen_gap": float(last["val_loss"] - last["train_loss"]) if last is not None else np.nan,
        "overfit_rise": float(last["val_loss"] - best["val_loss"]) if last is not None else np.nan,
        "diverged": bool(q["nonfinite_loss"] > 0 or (np.isfinite(s["loss"].iloc[-200:]).all() and s["loss"].iloc[-200:].mean() > 3.0)),
        "gnorm_p99": float(np.nanpercentile(s["grad_norm"].replace([np.inf, -np.inf], np.nan), 99)),
        "loss_vol_mean": float(s["loss_vol"].mean()),
        "tok_per_s_median": float(thr.median()),
        "gpu_mem_mb": float(s["gpu_mem_mb"].max()),
    }


# ---------------------------------------------------------------- 降维（纯 numpy 实现）
def pca(X, k=2):
    mu = X.mean(0); Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / (len(X) - 1)
    return Xc @ Vt[:k].T, var / var.sum(), Vt[:k], mu


def tsne(X, dim=2, perplexity=12.0, iters=1000, seed=0):
    """精确 t-SNE（van der Maaten & Hinton, 2008），适用于 <1000 个点"""
    rng = np.random.default_rng(seed)
    n = len(X)
    D = np.square(X[:, None, :] - X[None, :, :]).sum(-1)
    P = np.zeros((n, n)); logU = np.log(perplexity)
    for i in range(n):
        lo, hi, beta = -np.inf, np.inf, 1.0
        di = np.delete(D[i], i)
        for _ in range(60):
            p = np.exp(-di * beta); sp = p.sum() + 1e-12
            H = np.log(sp) + beta * (di * p).sum() / sp
            if abs(H - logU) < 1e-5: break
            if H > logU: lo = beta; beta = beta * 2 if hi == np.inf else (beta + hi) / 2
            else: hi = beta; beta = beta / 2 if lo == -np.inf else (beta + lo) / 2
        P[i, np.arange(n) != i] = p / sp
    P = (P + P.T) / (2 * n); P = np.maximum(P, 1e-12)
    Y = rng.normal(0, 1e-4, (n, dim)); dY = np.zeros_like(Y); gains = np.ones_like(Y)
    for it in range(iters):
        ex = 12.0 if it < 100 else 1.0
        sq = np.square(Y).sum(1); num = 1 / (1 + sq[:, None] + sq[None, :] - 2 * Y @ Y.T)
        np.fill_diagonal(num, 0); Q = np.maximum(num / num.sum(), 1e-12)
        PQ = (ex * P - Q) * num
        grad = 4 * ((np.diag(PQ.sum(1)) - PQ) @ Y)
        mom = 0.5 if it < 250 else 0.8
        gains = (gains + 0.2) * ((grad > 0) != (dY > 0)) + gains * 0.8 * ((grad > 0) == (dY > 0))
        gains = np.maximum(gains, 0.01)
        dY = mom * dY - 200.0 * gains * grad; Y = Y + dY; Y -= Y.mean(0)
    return Y


def char_class(c):
    if c in "aeiouAEIOU": return "元音"
    if c.isalpha(): return "大写辅音" if c.isupper() else "小写辅音"
    if c.isdigit(): return "数字"
    if c in " \n": return "空白"
    return "标点"


def main():
    os.makedirs(OUT, exist_ok=True)
    S, E, P, Q, SUM, CFG = [], [], [], [], [], {}
    for d in sorted(glob.glob(os.path.join(RUNS, "R*"))):
        cfg, s, e, p, q = load_run(d)
        s = add_features(s)
        S.append(s); E.append(e); P.append(p); Q.append(q); CFG[cfg["run"]] = cfg
        SUM.append(run_summary(cfg, s, e, q))
    s = pd.concat(S, ignore_index=True); e = pd.concat(E, ignore_index=True); p = pd.concat(P, ignore_index=True)
    s.to_pickle(os.path.join(OUT, "steps.pkl")); e.to_pickle(os.path.join(OUT, "evals.pkl")); p.to_pickle(os.path.join(OUT, "probes.pkl"))
    summ = pd.DataFrame(SUM); summ.to_csv(os.path.join(OUT, "run_summary.csv"), index=False)
    qual = pd.DataFrame(Q); qual.to_csv(os.path.join(OUT, "quality_report.csv"), index=False)
    json.dump(CFG, open(os.path.join(OUT, "configs.json"), "w"), ensure_ascii=False, indent=1)

    # ---- 训练状态轨迹：每 50 步窗口内 14 维模块 log 梯度范数均值 → 标准化 → PCA
    # 仅对同构（6 层）运行做 PCA：8 层的 R12 模块集合不同，无法放入同一特征空间
    base_mods = ["embed"] + [f"L{i}.{k}" for i in range(6) for k in ("attn", "mlp")] + ["head"]
    gcols = [f"g_{m}" for m in base_mods]
    same = [r for r, c in CFG.items() if c["layers"] == 6]
    s = s[s["run"].isin(same)].copy()
    s["win"] = s["step"] // 50
    W = s.replace([np.inf, -np.inf], np.nan).groupby(["run", "win"]).agg(
        {**{c: lambda x: np.log10(np.clip(x, 1e-8, None)).mean() for c in gcols}, "loss": "mean", "step": "min"}).reset_index()
    W = W.dropna(subset=gcols)
    X = W[gcols].values; Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Z, evr, comps, _ = pca(Xs, 3)
    W["pc1"], W["pc2"], W["pc3"] = Z[:, 0], Z[:, 1], Z[:, 2]
    W.to_pickle(os.path.join(OUT, "traj_pca.pkl"))
    json.dump({"explained_variance_ratio": evr[:6].tolist(), "features": gcols,
               "loadings": comps.tolist()}, open(os.path.join(OUT, "traj_pca_meta.json"), "w"), indent=1)

    # ---- 字符嵌入：基线运行
    emb = np.load(os.path.join(RUNS, "R01_baseline", "embed_tok.npy"))
    chars = CFG["R01_baseline"]["chars"]
    Ep, evr_e, _, _ = pca(emb, 2)
    Et = tsne((emb - emb.mean(0)) / emb.std(0), perplexity=10)
    pd.DataFrame({"char": chars, "cls": [char_class(c) for c in chars], "pca1": Ep[:, 0], "pca2": Ep[:, 1],
                  "tsne1": Et[:, 0], "tsne2": Et[:, 1]}).to_pickle(os.path.join(OUT, "embed_2d.pkl"))

    print(qual.to_string()); print(summ.round(4).to_string())
    print("traj PCA EVR:", np.round(evr[:4], 3))


if __name__ == "__main__":
    main()
