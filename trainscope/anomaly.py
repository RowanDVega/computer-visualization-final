"""
TrainScope · 训练异常检测器
===========================
每个检测器输入单个运行的清洗后数据，输出"事件"（区间 + 严重度 + 证据），
供可视化时间轴与智能体诊断使用。全部为可解释的统计规则，阈值见 THRESH。
"""
import numpy as np
import pandas as pd

THRESH = {
    "loss_spike_z": 8.0,        # 损失残差稳健 z 分数
    "gnorm_spike_z": 6.0,       # log 梯度范数稳健 z 分数
    "gnorm_spike_ratio": 5.0,   # 且至少为滚动中位数的 5 倍
    "diverge_loss": 4.0,        # 高于 ln(65)≈4.17 附近即视为退化为随机猜测
    "overfit_rise": 0.04,       # 验证损失高于历史最优的幅度
    "update_jump": 4.0,         # 更新比相对自身滚动中位数的跃升倍数
    "update_floor": 0.004,      # 且绝对值不低于该值
    "entropy_collapse": 0.5,    # 全部注意力头平均熵（nats）下限 → 全局塌缩
    "stall_loss": 2.0,          # 1000 步后平滑训练损失仍高于此值 → 收敛停滞
    "throughput_drop": 0.7,     # 吞吐低于中位数的比例
    "merge_gap": 15,            # 相邻异常步合并间隔
    "merge_gap_grad": 100,      # 梯度尖峰合并间隔（无裁剪时尖峰成簇出现）
}


def _robust_z(x, win=101):
    med = x.rolling(win, center=True, min_periods=10).median()
    mad = (x - med).abs().rolling(win, center=True, min_periods=10).median()
    return (x - med) / (1.4826 * mad + 1e-6)


def _events(steps, gap):
    """把离散的异常步合并为区间"""
    steps = sorted(steps); ev = []
    for s in steps:
        if ev and s - ev[-1][1] <= gap:
            ev[-1][1] = s
        else:
            ev.append([s, s])
    return ev


def detect(run, s, e, p):
    T = THRESH; out = []
    s = s.sort_values("step").reset_index(drop=True)
    loss = s["loss"].where(np.isfinite(s["loss"]))

    # 1) 发散：非有限损失，或平滑损失回升到随机猜测水平
    nf = s.loc[~np.isfinite(s["loss"]), "step"]
    ema = s["loss_ema"]
    hi = s.loc[(s["step"] > 300) & (ema > T["diverge_loss"]), "step"]
    if len(nf) or len(hi):
        st = int(min(nf.min() if len(nf) else 1e9, hi.min() if len(hi) else 1e9))
        out.append(dict(type="divergence", start=st, end=int(s["step"].max()), severity="critical",
                        evidence={"nonfinite_steps": int(len(nf)), "loss_ema_max": float(np.nanmax(ema))}))

    # 2) 损失尖峰：相对滚动中位数的稳健 z
    z = _robust_z(loss)
    cand = s.loc[(z > T["loss_spike_z"]) & (s["step"] > 50), "step"].tolist()
    for a, b in _events(cand, T["merge_gap"]):
        seg = s[(s["step"] >= a) & (s["step"] <= b)]
        out.append(dict(type="loss_spike", start=a, end=b,
                        severity="high" if seg["loss_resid"].max() > 0.5 else "medium",
                        evidence={"max_resid": float(seg["loss_resid"].max()), "max_z": float(z[seg.index].max()),
                                  "peak_loss": float(seg["loss"].max())}))

    # 3) 梯度爆炸：log 梯度范数稳健 z 且绝对倍数
    lg = s["log_gnorm"]; zg = _robust_z(lg)
    med = s["grad_norm"].rolling(101, center=True, min_periods=10).median()
    cand = s.loc[(zg > T["gnorm_spike_z"]) & (s["grad_norm"] > T["gnorm_spike_ratio"] * med) & (s["step"] > 50), "step"].tolist()
    for a, b in _events(cand, T["merge_gap_grad"]):
        seg = s[(s["step"] >= a) & (s["step"] <= b)]
        gcols = [c for c in s.columns if c.startswith("g_")]
        top = seg[gcols].max().sort_values(ascending=False)
        out.append(dict(type="grad_explosion", start=a, end=b, severity="high",
                        evidence={"peak_gnorm": float(seg["grad_norm"].max()),
                                  "baseline_gnorm": float(med[seg.index].median()),
                                  "top_modules": [c[2:] for c in top.index[:3]]}))

    # 4) 过拟合：验证损失较历史最优回升，且训练损失仍在下降
    ef = e[np.isfinite(e["val_loss"])].sort_values("step").reset_index(drop=True)
    if len(ef) > 5:
        best = ef["val_loss"].cummin()
        rise = ef["val_loss"] - best
        bad = ef.loc[rise > T["overfit_rise"], "step"]
        gap = ef["val_loss"].iloc[-1] - ef["train_loss"].iloc[-1]
        if len(bad) and gap > 0.15:   # 停滞运行的验证波动不算过拟合
            st = int(bad.min()); i_best = int(ef["val_loss"].idxmin())
            out.append(dict(type="overfitting", start=st, end=int(ef["step"].max()), severity="medium",
                            evidence={"best_val": float(ef["val_loss"].min()), "best_step": int(ef.loc[i_best, "step"]),
                                      "final_val": float(ef["val_loss"].iloc[-1]),
                                      "final_gap": float(ef["val_loss"].iloc[-1] - ef["train_loss"].iloc[-1])}))

    # 4b) 持续退化：验证损失大幅高于历史最优，且不是过拟合（训练损失同样升高）
    if len(ef) > 5:
        best_i = int(ef["val_loss"].idxmin()); bv = ef["val_loss"].min()
        after = ef.iloc[best_i:]
        bad = after[(after["val_loss"] > bv + 0.3) & (after["val_loss"] - after["train_loss"] < 0.15)]
        if len(bad):
            out.append(dict(type="regression", start=int(bad["step"].iloc[0]), end=int(ef["step"].max()), severity="critical",
                            evidence={"best_val": float(bv), "best_step": int(ef.loc[best_i, "step"]),
                                      "final_val": float(ef["val_loss"].iloc[-1]),
                                      "train_loss_final": float(ef["train_loss"].iloc[-1])}))

    # 5) 更新比跃升（逐模块最大值相对自身前序 200 步中位数）
    ucols = [c for c in s.columns if c.startswith("u_")]
    if ucols:
        U = s.dropna(subset=ucols[:1]).reset_index(drop=True)
        um = U[ucols].max(axis=1)
        base = um.shift(1).rolling(20, min_periods=8).median()
        cand = U.loc[(um > T["update_jump"] * base) & (um > T["update_floor"]) & (U["step"] > 100), "step"].tolist()
        for a, b in _events(cand, 50):
            seg = U[(U["step"] >= a) & (U["step"] <= b)]
            out.append(dict(type="large_update", start=a, end=b, severity="high",
                            evidence={"max_update_ratio": float(seg[ucols].max().max()),
                                      "baseline_ratio": float(base[seg.index].median()),
                                      "module": seg[ucols].max().idxmax()[2:]}))

    # 5b) 收敛停滞：训练足够久后平滑损失仍远高于正常水平
    late = s[s["step"] > 1000]
    if len(late) and late["loss_ema"].min() > T["stall_loss"]:
        out.append(dict(type="stagnation", start=1000, end=int(s["step"].max()), severity="high",
                        evidence={"loss_ema_min_after_1000": float(late["loss_ema"].min()),
                                  "best_val": float(e["val_loss"].min()) if len(e) else None}))

    # 6) 注意力熵全局塌缩（单个头的低熵通常是"前一字符头"等正常特化，不报警）
    if len(p):
        ent = np.array(p["attn_entropy"].tolist())          # [T, L, H]
        m = ent.mean(axis=(1, 2))
        bad = np.where((m < T["entropy_collapse"]) & (p["step"].values > 100))[0]
        if len(bad):
            out.append(dict(type="attention_collapse", start=int(p["step"].iloc[bad.min()]),
                            end=int(p["step"].iloc[bad.max()]), severity="high",
                            evidence={"mean_entropy_min": float(m.min())}))

    # 7) 吞吐下降（剔除计时污染）
    clean = s[~s["timing_contaminated"]]
    m = clean["tok_per_s"].median()
    slow = clean.loc[clean["tok_per_s"].rolling(25, min_periods=25).median() < T["throughput_drop"] * m, "step"].tolist()
    for a, b in _events(slow, 25):
        out.append(dict(type="throughput_drop", start=a, end=b, severity="low",
                        evidence={"median_tok_s": float(m)}))

    for o in out:
        o["run"] = run
    return out


def detect_all(S, E, P):
    ev = []
    for run, s in S.groupby("run"):
        ev += detect(run, s, E[E["run"] == run], P[P["run"] == run])
    return pd.DataFrame(ev)


GROUND_TRUTH = {  # 注入故障的真实区间（来自 run_sweep.sh）
    "R09_fault_lrspike": [("lr_spike", 1200, 1230)],
    "R10_fault_corrupt": [("data_corruption", 1800, 1840)],
}


def evaluate(events, healthy_runs):
    """事件级评估：注入故障是否被检出、检出延迟、健康运行上的误报数"""
    rows = []
    for run, gts in GROUND_TRUTH.items():
        for name, a, b in gts:
            hit = events[(events["run"] == run) & (events["end"] >= a) & (events["start"] <= b + 50)
                         & events["type"].isin(["loss_spike", "grad_explosion", "large_update"])]
            rows.append({"run": run, "fault": name, "gt_start": a, "gt_end": b, "detected": len(hit) > 0,
                         "delay_steps": int(max(0, hit["start"].min() - a)) if len(hit) else None,
                         "detectors": ",".join(sorted(hit["type"].unique()))})
    fp = events[events["run"].isin(healthy_runs) & events["type"].isin(["loss_spike", "grad_explosion", "large_update", "divergence", "stagnation", "attention_collapse"])]
    return pd.DataFrame(rows), len(fp)
