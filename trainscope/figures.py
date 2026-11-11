"""
TrainScope · 论文静态图生成（matplotlib）
配色：参考调色板（分类色固定顺序；序列色单一蓝色明→暗；发散色 蓝↔红 + 灰中点）
12 个运行远超 3 色上限，因此采用"灰色上下文 + 至多 3 个高亮运行"或小多图。
"""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib import font_manager

sys.path.insert(0, os.path.dirname(__file__))
from anomaly import detect_all, evaluate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "processed")
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]:
    if os.path.exists(f): font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": ["Noto Sans CJK JP", "DejaVu Sans"], "axes.unicode_minus": False,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "xtick.color": "#898781", "ytick.color": "#898781",
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "#ffffff", "axes.facecolor": "#fcfcfb",
    "savefig.dpi": 200, "savefig.bbox": "tight", "lines.linewidth": 1.4,
})
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRAY = "#c3c2b7"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
ST = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f4a9a8", "#e34948", "#a32d2c"])
SEV = {"critical": ST["critical"], "high": ST["serious"], "medium": ST["warning"], "low": "#9ec5f4"}
TYPE_CN = {"divergence": "发散", "loss_spike": "损失尖峰", "grad_explosion": "梯度爆炸", "overfitting": "过拟合",
           "large_update": "更新比跃升", "stagnation": "收敛停滞", "regression": "持续退化", "attention_collapse": "注意力塌缩", "throughput_drop": "吞吐下降"}

S = pd.read_pickle(f"{D}/steps.pkl"); E = pd.read_pickle(f"{D}/evals.pkl"); P = pd.read_pickle(f"{D}/probes.pkl")
SUM = pd.read_csv(f"{D}/run_summary.csv"); CFG = json.load(open(f"{D}/configs.json"))
RUNS = sorted(S["run"].unique())
EV = detect_all(S, E, P)
EV.to_csv(f"{D}/events.csv", index=False)
GCOLS = [c for c in S.columns if c.startswith("g_")]


def save(fig, name):
    fig.savefig(os.path.join(FIG, name)); plt.close(fig); print("saved", name)


def short(r): return r.split("_", 1)[0]


# ---------------------------------------------------------------- 图：数据清洗示意
def fig_cleaning():
    raw = [json.loads(l) for l in open(os.path.join(ROOT, "data/runs/R11_crash_resume/scalars.jsonl"))]
    raw = pd.DataFrame(raw); raw["loss"] = raw["loss"].astype(float); raw["order"] = np.arange(len(raw))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    a = ax[0]
    seg = raw[(raw["step"] >= 850) & (raw["step"] <= 1550)]
    first = seg[seg["order"] < seg.groupby("step")["order"].transform("max")]
    a.plot(seg.groupby("step")["loss"].last().rolling(15).mean(), color=C[0], label="清洗后（保留续训写入）")
    a.plot(first.set_index("step")["loss"].rolling(15).mean(), color=C[1], ls="--", label="被覆盖的崩溃前记录")
    a.axvspan(1001, 1399, color="#f0efec", zorder=0)
    a.text(1190, 1.165, "重复写入区间 1001–1372\n（1373–1399 崩溃时未落盘）", ha="center", color=INK2, fontsize=7.5)
    a.set_title("(a) 断点续训产生的重复 step（R11）", loc="left"); a.set_xlabel("step"); a.set_ylabel("训练损失（15 步滑动均值）")
    a.legend(frameon=False)
    b = ax[1]
    d = S[(S["run"] == "R01_baseline") & (S["step"].between(100, 500))]
    b.plot(d["step"], d["step_time"] * 1000, color=GRAY, lw=1, label="原始 step_time")
    c = d[d["timing_contaminated"]]
    b.scatter(c["step"], c["step_time"] * 1000, s=14, color=ST["serious"], zorder=3, label="被标记的计时污染点")
    b.plot(d.loc[~d["timing_contaminated"], "step"], d.loc[~d["timing_contaminated"], "step_time"] * 1000, color=C[0], lw=1.2, label="清洗后")
    b.set_yscale("log"); b.set_title("(b) 评估/探针开销污染的计时（R01）", loc="left"); b.set_xlabel("step"); b.set_ylabel("单步耗时 / ms（对数）")
    b.legend(frameon=False, loc="upper right")
    save(fig, "fig_cleaning.png")


# ---------------------------------------------------------------- 图：EDA
def fig_eda():
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
    a = ax[0]
    data = [np.log10(S.loc[(S["run"] == r) & (S["step"] > 200), "grad_norm"].replace([np.inf, -np.inf], np.nan).dropna().clip(1e-4)) for r in RUNS]
    bp = a.boxplot(data, vert=True, widths=0.55, patch_artist=True, showfliers=True,
                   flierprops=dict(marker=".", markersize=2, markeredgecolor=MUTED), medianprops=dict(color=INK, lw=1.2))
    for box, r in zip(bp["boxes"], RUNS):
        bad = r in ("R04_lr1e-2_noclip", "R09_fault_lrspike", "R10_fault_corrupt")
        box.set_facecolor("#f4c2ad" if bad else "#cde2fb"); box.set_edgecolor(ST["serious"] if bad else C[0])
    a.axhline(0, color=ST["critical"], lw=0.8, ls=":"); a.text(12.4, 0.03, "裁剪阈值 1.0", color=ST["critical"], fontsize=7, ha="right", va="bottom")
    a.set_xticks(range(1, len(RUNS) + 1)); a.set_xticklabels([short(r) for r in RUNS])
    a.set_ylabel("log10(全局梯度范数, 裁剪前)"); a.set_title("(a) 各运行梯度范数分布（step>200）", loc="left")
    b = ax[1]
    d = S[(S["step"] > 200)].replace([np.inf, -np.inf], np.nan)
    ucols = [c for c in S.columns if c.startswith("u_")]
    feat = pd.DataFrame({"loss": d["loss"], "log_gnorm": d["log_gnorm"], "lr": d["lr"], "loss_vol": d["loss_vol"],
                         "|resid|": d["loss_resid"].abs(), "upd_ratio": d[ucols].mean(axis=1), "step": d["step"]}).dropna()
    corr = feat.rank().corr().values
    im = b.imshow(corr, cmap=DIV, norm=TwoSlopeNorm(0, -1, 1))
    lab = list(feat.columns); b.set_xticks(range(len(lab))); b.set_xticklabels(lab, rotation=45, ha="right"); b.set_yticks(range(len(lab))); b.set_yticklabels(lab)
    for i in range(len(lab)):
        for j in range(len(lab)):
            b.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="#ffffff" if abs(corr[i, j]) > 0.6 else INK)
    b.grid(False); b.set_title("(b) 逐步特征 Spearman 相关（每 10 步采样）", loc="left"); fig.colorbar(im, ax=b, fraction=0.046)
    save(fig, "fig_eda.png")
    return pd.DataFrame(corr, index=lab, columns=lab)


# ---------------------------------------------------------------- 图：小多图损失曲线 + 异常
def fig_small_multiples():
    fig, axs = plt.subplots(3, 4, figsize=(11, 7), sharex=True, sharey=True)
    for a, r in zip(axs.flat, RUNS):
        d = S[S["run"] == r]; e = E[E["run"] == r]
        for _, x in EV[EV["run"] == r].iterrows():
            if x["type"] in ("throughput_drop",): continue
            a.axvspan(x["start"], max(x["end"], x["start"] + 15), color=SEV[x["severity"]], alpha=0.28, lw=0, zorder=0)
        a.plot(d["step"], d["loss"].clip(upper=6), color=GRAY, lw=0.5)
        a.plot(e["step"], e["train_loss"], color=C[0], lw=1.4, label="训练损失")
        a.plot(e["step"], e["val_loss"], color=C[1], lw=1.4, label="验证损失")
        s = SUM.set_index("run").loc[r]
        if np.isfinite(s["best_val"]):
            a.scatter([s["best_step"]], [s["best_val"]], s=14, color=C[1], edgecolor="white", zorder=4)
        a.set_title(f"{r}", loc="left", fontsize=8.5)
        a.text(0.97, 0.93, f"best val {s['best_val']:.3f}", transform=a.transAxes, ha="right", va="top", fontsize=7.5, color=INK2)
        a.set_ylim(0.6, 4.5)
    axs[0, 0].legend(frameon=False, loc="upper center")
    for a in axs[-1]: a.set_xlabel("step")
    for a in axs[:, 0]: a.set_ylabel("损失")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=SEV[k], alpha=0.5, label=l) for k, l in
                        [("critical", "严重"), ("high", "高"), ("medium", "中")]], loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02), title="异常严重度（色带）")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fig_small_multiples.png")


# ---------------------------------------------------------------- 图：模块 × step 梯度热力图
def fig_module_heatmap():
    order = ["embed"] + [f"L{i}.{k}" for i in range(6) for k in ("attn", "mlp")] + ["head"]
    fig, axs = plt.subplots(3, 1, figsize=(10, 7.2), sharex=True)
    for a, r, ttl in zip(axs, ["R01_baseline", "R09_fault_lrspike", "R10_fault_corrupt"],
                         ["(a) R01 基线", "(b) R09 学习率尖峰注入（1200–1230）", "(c) R10 标签污染注入（1800–1840）"]):
        d = S[S["run"] == r].copy(); d["w"] = d["step"] // 10
        M = d.groupby("w")[[f"g_{o}" for o in order]].mean()
        im = a.imshow(np.log10(M.T.values.clip(1e-4)), aspect="auto", cmap=SEQ, vmin=-2.6, vmax=0.6,
                      extent=[0, d["step"].max(), len(order) - 0.5, -0.5], interpolation="nearest")
        a.set_yticks(range(len(order))); a.set_yticklabels(order, fontsize=7); a.grid(False)
        a.set_title(ttl, loc="left")
    axs[-1].set_xlabel("step（10 步聚合）")
    cb = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.01); cb.set_label("log10 模块梯度范数")
    save(fig, "fig_module_heatmap.png")


# ---------------------------------------------------------------- 图：注意力熵演化
def fig_attention():
    steps = [0, 100, 300, 800, 1500, 2499]
    fig, axs = plt.subplots(2, len(steps), figsize=(11, 4.4))
    for row, r in enumerate(["R01_baseline", "R04_lr1e-2_noclip"]):
        p = P[P["run"] == r].set_index("step")
        for a, st in zip(axs[row], steps):
            st2 = p.index[np.abs(p.index - st).argmin()]
            M = np.array(p.loc[st2, "attn_entropy"])
            im = a.imshow(M, cmap=SEQ, vmin=0, vmax=4.5); a.grid(False)
            a.set_xticks(range(M.shape[1])); a.set_yticks(range(M.shape[0]))
            a.tick_params(labelsize=6)
            a.set_title(f"step {st2}", fontsize=8)
            if st == steps[0]: a.set_ylabel(f"{short(r)}\n层", fontsize=8)
            if row == 1: a.set_xlabel("头", fontsize=7)
    cb = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.01); cb.set_label("平均注意力熵 / nats（上限 ln256≈5.5）")
    save(fig, "fig_attention.png")


# ---------------------------------------------------------------- 图：PCA 训练状态轨迹
def fig_pca():
    W = pd.read_pickle(f"{D}/traj_pca.pkl"); meta = json.load(open(f"{D}/traj_pca_meta.json"))
    evr = meta["explained_variance_ratio"]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    a = ax[0]
    hl = {"R01_baseline": C[0], "R04_lr1e-2_noclip": C[1], "R09_fault_lrspike": C[2]}
    for r in RUNS:
        w = W[W["run"] == r]
        if r in hl: continue
        a.plot(w["pc1"], w["pc2"], color=GRAY, lw=0.8, alpha=0.8)
    for r, c in hl.items():
        w = W[W["run"] == r]
        a.plot(w["pc1"], w["pc2"], color=c, lw=1.8, label=r)
        a.scatter(w["pc1"].iloc[0], w["pc2"].iloc[0], s=30, marker="o", color=c, edgecolor="white", zorder=4)
        a.scatter(w["pc1"].iloc[-1], w["pc2"].iloc[-1], s=40, marker="s", color=c, edgecolor="white", zorder=4)
    a.set_xlabel(f"PC1（{evr[0]:.1%} 方差）"); a.set_ylabel(f"PC2（{evr[1]:.1%} 方差）")
    a.set_title("(a) 14 维模块梯度空间中的训练轨迹（○起点 □终点）", loc="left"); a.legend(frameon=False, fontsize=7.5)
    b = ax[1]
    feats = [f[2:] for f in meta["features"]]; L = np.array(meta["loadings"])[:2]
    y = np.arange(len(feats)); h = 0.38
    b.barh(y - h / 2, L[0], height=h, color=C[0], label="PC1"); b.barh(y + h / 2, L[1], height=h, color=C[1], label="PC2")
    b.set_yticks(y); b.set_yticklabels(feats, fontsize=7); b.invert_yaxis(); b.axvline(0, color=GRAY, lw=0.8)
    b.set_title("(b) 主成分载荷", loc="left"); b.legend(frameon=False)
    save(fig, "fig_pca.png")
    return evr


# ---------------------------------------------------------------- 图：字符嵌入
def fig_embed():
    Em = pd.read_pickle(f"{D}/embed_2d.pkl")
    cls = ["元音", "小写辅音", "大写辅音", "标点", "空白", "数字"]
    col = {"元音": C[1], "小写辅音": C[0], "大写辅音": C[2], "标点": INK2, "空白": INK2, "数字": INK2}
    mk = {"元音": "o", "小写辅音": "o", "大写辅音": "s", "标点": "^", "空白": "D", "数字": "v"}
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for a, (x, y, t) in zip(ax, [("pca1", "pca2", "(a) PCA"), ("tsne1", "tsne2", "(b) t-SNE（perplexity=10）")]):
        for c in cls:
            d = Em[Em["cls"] == c]
            if not len(d): continue
            a.scatter(d[x], d[y], s=60, marker=mk[c], color=col[c], edgecolor="white", lw=0.8, label=c, alpha=0.9)
        for _, r in Em.iterrows():
            lab = {"\n": "\\n", " ": "␣"}.get(r["char"], r["char"])
            a.annotate(lab, (r[x], r[y]), xytext=(3, 3), textcoords="offset points", fontsize=7, color=INK)
        a.set_title(t + " · R01 字符嵌入（65×384）", loc="left"); a.set_xticks([]); a.set_yticks([])
    ax[0].legend(frameon=False, fontsize=7.5, ncol=2)
    save(fig, "fig_embed.png")


# ---------------------------------------------------------------- 图：超参数与结果
def fig_hparams():
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1, 1.4], "wspace": 0.42})
    a = ax[0]
    lr_runs = ["R02_lr3e-4", "R01_baseline", "R03_lr3e-3", "R04_lr1e-2_noclip"]
    d = SUM.set_index("run").loc[lr_runs]
    a.plot(d["lr"], d["best_val"], color=C[0], marker="o", ms=6, label="最优验证损失")
    a.plot(d["lr"], d["final_val"], color=C[1], marker="s", ms=6, label="最终验证损失")
    for r, x in d.iterrows():
        a.annotate(short(r), (x["lr"], x["best_val"]), xytext=(-24 if x["lr"] > 5e-3 else 4, -10), textcoords="offset points", fontsize=7.5, color=INK2)
    a.set_xscale("log"); a.set_xlabel("峰值学习率（对数）"); a.set_ylabel("验证损失")
    a.set_title("(a) 学习率敏感性（R04 未裁剪）", loc="left"); a.legend(frameon=False)
    b = ax[1]
    d = SUM.sort_values("best_val")
    y = np.arange(len(d))
    b.hlines(y, d["best_val"], d["final_val"].clip(upper=4.5), color=GRAY, lw=2)
    b.scatter(d["best_val"], y, color=C[0], s=36, zorder=3, label="最优验证损失")
    b.scatter(d["final_val"].clip(upper=4.5), y, color=C[1], s=36, zorder=3, marker="s", label="最终验证损失")
    b.set_yticks(y); b.set_yticklabels(d["run"], fontsize=7.5); b.invert_yaxis()
    b.set_xlabel("验证损失（灰线长度 = 过拟合回升量）"); b.set_title("(b) 各运行最优 vs 最终验证损失", loc="left"); b.legend(frameon=False, loc="lower right")
    save(fig, "fig_hparams.png")


# ---------------------------------------------------------------- 图：检测效果
def fig_detection():
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 5.2), sharex="col")
    for col, (r, a0, b0, lo, hi) in enumerate([("R09_fault_lrspike", 1200, 1230, 1050, 1500),
                                               ("R10_fault_corrupt", 1800, 1840, 1650, 2050)]):
        d = S[(S["run"] == r) & S["step"].between(lo, hi)]; e = E[(E["run"] == r) & E["step"].between(lo, hi)]
        ev = EV[(EV["run"] == r) & (EV["end"] >= lo) & (EV["start"] <= hi)]
        for a in ax[:, col]:
            a.axvspan(a0, b0, color="#f0efec", zorder=0)
        a = ax[0, col]
        a.plot(d["step"], d["loss"], color=GRAY, lw=0.7, label="训练损失（逐步）")
        a.plot(d["step"], d["loss_med"], color=C[0], lw=1.4, label="滚动中位数")
        a.plot(e["step"], e["val_loss"], color=C[1], marker="o", ms=3.5, lw=1.4, label="验证损失")
        for i, (_, x) in enumerate(ev.iterrows()):
            if x["type"] in ("overfitting", "throughput_drop"): continue
            a.annotate(TYPE_CN[x["type"]], xy=(x["start"], d["loss"].max() * (0.95 - 0.12 * i)), fontsize=7.5, color=INK,
                       xytext=(8, 0), textcoords="offset points", arrowprops=dict(arrowstyle="-", color=MUTED))
        a.set_title(f"({'ab'[col]}) {r}：灰带为注入区间", loc="left"); a.set_ylabel("损失")
        if col == 0: a.legend(frameon=False, fontsize=7, loc="upper right")
        a = ax[1, col]
        a.plot(d["step"], d["grad_norm"], color=C[0], lw=1)
        a.set_yscale("log"); a.set_ylabel("梯度范数（对数）"); a.set_xlabel("step")
        u = d.dropna(subset=["u_L0.attn"]); ucols = [c for c in d.columns if c.startswith("u_")]
        a2 = a.inset_axes([0.62, 0.55, 0.36, 0.4]); a2.plot(u["step"], u[ucols].max(axis=1), color=C[2], lw=1)
        a2.set_yscale("log"); a2.tick_params(labelsize=6); a2.set_title("最大模块更新比", fontsize=7)
        a2.axhline(0.01, color=ST["critical"], lw=0.7, ls=":")
    save(fig, "fig_detection.png")


if __name__ == "__main__":
    fig_cleaning(); corr = fig_eda(); fig_small_multiples(); fig_module_heatmap(); fig_attention()
    evr = fig_pca(); fig_embed(); fig_hparams(); fig_detection()
    healthy = ["R01_baseline", "R02_lr3e-4", "R07_bs16", "R08_nodropout", "R11_crash_resume", "R12_large"]
    ev_tab, fp = evaluate(EV, healthy)
    ev_tab.to_csv(f"{D}/detection_eval.csv", index=False)
    print(ev_tab.to_string()); print("false positives on healthy runs:", fp)
    print(EV[["run", "type", "start", "end", "severity"]].to_string())
    corr.round(3).to_csv(f"{D}/feature_corr.csv")
