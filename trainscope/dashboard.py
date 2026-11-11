"""
TrainScope · 导出交互式仪表盘数据（web/data.js）
"""
import json, math, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "processed")


def r4(x):
    if x is None: return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return x
    return round(x, 5) if math.isfinite(x) else None


def main():
    S = pd.read_pickle(f"{D}/steps.pkl"); E = pd.read_pickle(f"{D}/evals.pkl"); P = pd.read_pickle(f"{D}/probes.pkl")
    W = pd.read_pickle(f"{D}/traj_pca.pkl"); Em = pd.read_pickle(f"{D}/embed_2d.pkl")
    SUM = pd.read_csv(f"{D}/run_summary.csv"); Q = pd.read_csv(f"{D}/quality_report.csv")
    CFG = json.load(open(f"{D}/configs.json")); AG = json.load(open(f"{D}/agent_output.json"))
    meta = json.load(open(f"{D}/traj_pca_meta.json"))
    order = ["embed"] + [f"L{i}.{k}" for i in range(8) for k in ("attn", "mlp")] + ["head"]
    runs = {}
    for r, d in S.groupby("run"):
        d = d.sort_values("step")
        mods = [m for m in order if f"g_{m}" in d and d[f"g_{m}"].notna().any()]
        d2 = d.assign(w=d["step"] // 10)
        G = d2.groupby("w")[[f"g_{m}" for m in mods]].mean()
        ucols = [f"u_{m}" for m in mods]
        U = d.dropna(subset=[ucols[0]])
        e = E[E["run"] == r].sort_values("step"); p = P[P["run"] == r].sort_values("step"); w = W[W["run"] == r]
        runs[r] = {
            "cfg": {k: CFG[r][k] for k in ("lr", "warmup", "clip", "batch", "dropout", "layers", "d", "heads", "n_params", "lr_spike", "corrupt")},
            "step": d["step"].tolist(),
            "loss": [r4(x) for x in d["loss"]], "loss_ema": [r4(x) for x in d["loss_ema"]],
            "lr": [r4(x) for x in d["lr"]], "gnorm": [r4(x) for x in d["grad_norm"]],
            "tps": [None if c else r4(x / 1000) for x, c in zip(d["tok_per_s"], d["timing_contaminated"])],
            "ev_step": e["step"].tolist(), "train_loss": [r4(x) for x in e["train_loss"]], "val_loss": [r4(x) for x in e["val_loss"]],
            "mods": mods, "gmat_steps": (G.index * 10).tolist(),
            "gmat": [[r4(math.log10(max(v, 1e-6))) if v == v else None for v in row] for row in G.values.tolist()],
            "u_step": U["step"].tolist(), "umax": [r4(x) for x in U[ucols].max(axis=1)],
            "probe_step": p["step"].tolist(), "entropy": [[[round(v, 3) for v in h] for h in L] for L in p["attn_entropy"]],
            "act_rms": [[round(v, 3) for v in L] for L in p["act_rms"]],
            "pca": [[r4(a), r4(b), int(s)] for a, b, s in zip(w["pc1"], w["pc2"], w["step"])],
        }
    data = {
        "runs": runs,
        "summary": [{k: r4(v) if isinstance(v, (float, np.floating)) else (bool(v) if isinstance(v, (bool, np.bool_)) else v)
                     for k, v in row.items()} for row in SUM.to_dict("records")],
        "quality": Q.where(Q.notna(), None).to_dict("records"),
        "pca_evr": meta["explained_variance_ratio"][:3],
        "embed": Em.round(4).to_dict("records"),
        "agent": AG,
    }
    js = "window.TS_DATA=" + json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=r4) + ";"
    open(os.path.join(ROOT, "web", "data.js"), "w").write(js)
    print("data.js", len(js) / 1e6, "MB")


if __name__ == "__main__":
    main()
