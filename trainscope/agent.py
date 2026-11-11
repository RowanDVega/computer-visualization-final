"""
TrainScope · 多智能体诊断工作流
================================
编排结构（Orchestrator–Worker）：

    Planner ──► Monitor ──► Diagnostician ──► Advisor ──► Reporter
       │            │             │                         ▲
       └────────────┴──── Comparator（跨运行对比）───────────┘

* Planner       ：读取运行摘要，决定每个运行需要调用哪些下游智能体与工具（路由）；
* Monitor       ：调用 anomaly.detect，产出带证据的异常事件；
* Diagnostician ：对每个事件调用"取证工具"（窗口指标查询），在知识库中为候选根因打分；
* Advisor       ：把根因映射为可执行的处置建议（含具体参数与回滚步数）；
* Comparator    ：跨运行排序与超参数敏感性分析；
* Reporter      ：汇总为中文诊断报告。若设置 ANTHROPIC_API_KEY，则把结构化证据交给
                  大模型润色成自然语言（证据约束生成）；否则使用确定性模板，保证可复现。

所有智能体间消息与工具调用都写入 trace，供可视化界面回放（可解释性）。
"""
import json, math, os, time, urllib.request
import numpy as np
import pandas as pd

from anomaly import detect

LN_V = math.log(65)


class Trace:
    def __init__(s): s.items = []; s.t0 = time.time()

    def log(s, agent, kind, content, run=None):
        s.items.append({"i": len(s.items), "agent": agent, "kind": kind, "run": run,
                        "content": content, "t_ms": round((time.time() - s.t0) * 1000, 1)})


# ------------------------------------------------------------------ 工具（供智能体调用）
class Tools:
    def __init__(s, S, E, P, cfgs):
        s.S, s.E, s.P, s.cfgs = S, E, P, cfgs

    def window(s, run, a, b, pad=30):
        """取证工具：返回事件窗口与前序对照窗口的关键统计"""
        d = s.S[s.S["run"] == run]
        pre = d[(d["step"] >= a - 150) & (d["step"] < a - pad)]
        cur = d[(d["step"] >= a) & (d["step"] <= b + pad)]
        ucols = [c for c in d.columns if c.startswith("u_")]
        gcols = [c for c in d.columns if c.startswith("g_")]
        f = lambda x: float(np.nanmean(x.replace([np.inf, -np.inf], np.nan))) if len(x) else float("nan")
        ev = s.E[(s.E["run"] == run)]
        vpre = ev[(ev["step"] < a)].tail(2)["val_loss"]; vcur = ev[(ev["step"] >= a) & (ev["step"] <= b + 100)]["val_loss"]
        g_pre = pre[gcols].median(); g_cur = cur[gcols].max()
        ratio = (g_cur / (g_pre + 1e-9)).replace([np.inf], np.nan)
        return {
            "lr_ratio": f(cur["lr"]) / (f(pre["lr"]) + 1e-12),
            "loss_pre": f(pre["loss"]), "loss_cur_max": float(cur["loss"].replace([np.inf], np.nan).max()),
            "gnorm_ratio": float(cur["grad_norm"].max() / (pre["grad_norm"].median() + 1e-9)),
            "update_ratio_pre": f(pre[ucols].max(axis=1)) if ucols else None,
            "update_ratio_cur": f(cur[ucols].max(axis=1)) if ucols else None,
            "val_pre": float(vpre.mean()) if len(vpre) else None,
            "val_cur_max": float(vcur.max()) if len(vcur) else None,
            "module_spread": float(np.nanstd(np.log10(ratio.clip(lower=1e-6)))),  # 各模块梯度放大倍数是否一致
            "top_modules": [c[2:] for c in ratio.sort_values(ascending=False).index[:3]],
        }


# ------------------------------------------------------------------ 根因知识库
def kb_score(ev, w, cfg):
    """为候选根因打分：每条规则 (条件, 权重, 说明)，分数经 softmax 转为置信度"""
    t = ev["type"]; H = {}

    def add(h, cond, wt, why):
        if cond:
            H.setdefault(h, [0.0, []]); H[h][0] += wt; H[h][1].append(why)

    lr_r = w.get("lr_ratio", 1); ur = w.get("update_ratio_cur") or 0; ur0 = w.get("update_ratio_pre") or 1e-9
    vjump = (w["val_cur_max"] or 0) - (w["val_pre"] or 0) if w.get("val_pre") else 0
    dyn = t in ("loss_spike", "grad_explosion", "large_update", "divergence", "stagnation", "attention_collapse")
    add("前序冲击造成的不可逆损伤", t == "regression" and ev.get("prior_shock") is not None, 3.5,
        f"退化前第 {ev.get('prior_shock')} 步发生 {ev.get('prior_type')}，此后训练/验证损失同步停留在高位，无法自行恢复")
    add("数值不稳定 / 训练发散", t == "regression", 1.0, "训练与验证损失同步恶化，权重已偏离良好区域")
    add("学习率过高/学习率调度异常", dyn and lr_r > 3, 3.0, f"事件窗口学习率为前序窗口的 {lr_r:.1f} 倍")
    add("学习率过高/学习率调度异常", dyn and ur > 3 * ur0 and ur > 0.005, 1.5, f"参数更新比由 {ur0:.1e} 升至 {ur:.1e}")
    add("学习率过高/学习率调度异常", dyn and cfg["lr"] >= 3e-3, 1.0, f"配置峰值学习率 {cfg['lr']:.0e} 偏大")
    add("学习率过高/学习率调度异常", dyn and vjump > 0.1, 1.0, f"验证损失同步上升 {vjump:.2f}，说明权重真实受损")
    add("学习率过高/学习率调度异常", t == "stagnation" and cfg["lr"] >= 5e-3, 2.0,
        f"峰值学习率 {cfg['lr']:.0e} 使优化停留在高损失区域（最低平滑损失 {ev['evidence'].get('loss_ema_min_after_1000', 0):.2f}）")
    shock = t in ("loss_spike", "grad_explosion")
    add("数据管道污染（标签/样本异常）", shock and w["loss_cur_max"] > LN_V - 0.6 and lr_r < 1.5, 3.0,
        f"损失峰值 {w['loss_cur_max']:.2f} 超过均匀猜测熵 ln65={LN_V:.2f}（比随机猜测更差，说明目标与输入不一致），而学习率未变")
    add("数据管道污染（标签/样本异常）", shock and vjump < 0.1 and lr_r < 1.5 and w["loss_cur_max"] > w["loss_pre"] + 1.0, 2.0,
        f"训练损失飙升但验证损失仅变化 {vjump:+.2f}，异常来自输入数据而非权重")
    add("缺少学习率预热", (ev["start"] < 600 or t == "attention_collapse") and cfg["warmup"] <= 1, 3.0,
        "未配置 warmup：训练初期以峰值学习率直接更新，注意力 logits 迅速增大导致熵塌缩" if t == "attention_collapse" else "事件发生于训练初期且未配置 warmup")
    add("未启用梯度裁剪", cfg["clip"] <= 0 and t in ("grad_explosion", "divergence", "stagnation"), 2.0, "梯度裁剪关闭，尖峰直接作用于参数")
    add("数值不稳定 / 训练发散", t == "divergence", 3.0, "平滑损失回升至随机猜测水平或出现非有限值")
    add("容量过剩导致过拟合", t == "overfitting", 2.0, f"验证损失自最优 {ev['evidence'].get('best_val', 0):.3f} 回升")
    add("容量过剩导致过拟合", t == "overfitting" and cfg["dropout"] == 0, 1.5, "未使用 dropout 正则化")
    add("容量过剩导致过拟合", t == "overfitting" and cfg["layers"] * cfg["d"] > 3000, 1.0, "模型规模大于基线")
    add("批量过小导致梯度噪声", cfg["batch"] < 32 and t in ("loss_spike", "grad_explosion"), 1.5, f"batch={cfg['batch']}")
    # 始终保留"其他/未知"基准假设（得分 1.0），避免单一假设时置信度虚高为 100%
    H["其他/未知原因"] = [1.0, ["未被知识库覆盖的原因（基准假设）"]]
    ks = list(H); sc = np.array([H[k][0] for k in ks]); pr = np.exp(sc) / np.exp(sc).sum()
    return sorted([{"cause": k, "score": float(H[k][0]), "confidence": float(c), "why": H[k][1]}
                   for k, c in zip(ks, pr)], key=lambda r: -r["score"])


ACTIONS = {
    "学习率过高/学习率调度异常": "检查学习率调度器与配置热更新；将峰值学习率下调至 1/3，并从异常前最近检查点（≤{a}步）回滚续训",
    "数据管道污染（标签/样本异常）": "隔离第 {a}–{b} 步对应的数据分片，校验 tokenizer/标签对齐与样本哈希；该段损失无需回滚但应剔除",
    "缺少学习率预热": "设置 warmup ≥ 200 步（约总步数 8%），或降低初始学习率",
    "未启用梯度裁剪": "启用全局梯度范数裁剪（max_norm=1.0）",
    "数值不稳定 / 训练发散": "立即终止并回滚；降低学习率 3–10 倍，启用裁剪与 warmup，必要时切换 fp32 累加",
    "容量过剩导致过拟合": "在第 {best} 步（验证最优）早停或保存最优检查点；增大 dropout/weight decay 或扩充数据",
    "批量过小导致梯度噪声": "增大 batch 或使用梯度累积，同时按线性规则调整学习率",
    "偶发性随机波动": "持续观察，无需干预",
    "其他/未知原因": "证据不足，建议人工复核",
    "前序冲击造成的不可逆损伤": "终止当前运行，回滚到第 {best} 步（验证最优）检查点并修复诱因后续训，避免继续消耗算力",
    "需人工复核（系统/表征层面）": "检查节点负载、数据加载线程与 I/O，排除共享 GPU 干扰",
}


# ------------------------------------------------------------------ LLM（可选）
def llm(prompt, max_tokens=900):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", method="POST", data=json.dumps({
        "model": os.environ.get("TRAINSCOPE_MODEL", "claude-opus-5-5"), "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]}).encode(),
        headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=60).read())["content"][0]["text"]
    except Exception as ex:  # 失败时回退模板
        return None


# ------------------------------------------------------------------ 编排
def run_pipeline(S, E, P, cfgs, summary):
    tr = Trace(); tools = Tools(S, E, P, cfgs); results = {}
    fleet_best = summary["best_val"].min()

    # Planner：路由
    plan = {}
    for _, r in summary.iterrows():
        steps = ["monitor"]
        if not r["diverged"]: steps.append("diagnose")
        else: steps += ["diagnose_divergence"]
        steps += ["advise", "report"]
        plan[r["run"]] = steps
    tr.log("Planner", "plan", {"runs": len(plan), "routes": plan})

    for run, route in plan.items():
        cfg = cfgs[run]
        events = detect(run, S[S["run"] == run], E[E["run"] == run], P[P["run"] == run])
        tr.log("Monitor", "tool_call", {"tool": "anomaly.detect", "n_events": len(events),
                                         "types": sorted({e["type"] for e in events})}, run)
        diags = []
        for ev in events:
            if ev["type"] == "regression":
                prior = [x for x in events if x["type"] in ("loss_spike", "grad_explosion", "large_update") and x["start"] <= ev["start"]]
                if prior:
                    ev["prior_shock"] = prior[0]["start"]; ev["prior_type"] = prior[0]["type"]
            if ev["type"] == "throughput_drop":
                diags.append({"event": ev, "causes": [{"cause": "需人工复核（系统/表征层面）", "confidence": 1.0,
                                                       "score": 0, "why": [json.dumps(ev["evidence"], ensure_ascii=False)]}]})
                continue
            w = tools.window(run, ev["start"], ev["end"])
            tr.log("Diagnostician", "tool_call", {"tool": "metrics.window", "range": [ev["start"], ev["end"]],
                                                  "result": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in w.items()}}, run)
            causes = kb_score(ev, w, cfg)
            tr.log("Diagnostician", "hypotheses", {"event": ev["type"], "top": causes[:3]}, run)
            diags.append({"event": ev, "window": w, "causes": causes})

        advice = []
        best_step = int(summary.loc[summary["run"] == run, "best_step"].iloc[0])
        for d in diags:
            c = d["causes"][0]["cause"]; ev = d["event"]
            a = ACTIONS.get(c, "人工复核").format(a=ev["start"], b=ev["end"], best=best_step)
            if c not in [x["cause"] for x in advice] and c != "其他/未知原因":
                advice.append({"cause": c, "action": a, "event": ev["type"], "at": [ev["start"], ev["end"]]})
        tr.log("Advisor", "advice", advice, run)

        r = summary[summary["run"] == run].iloc[0]
        health = "危险" if (r["diverged"] or any(e["severity"] == "critical" for e in events)) else ("警告" if any(e["severity"] in ("high", "critical") for e in events)
                                              else ("关注" if events else "健康"))
        results[run] = {"health": health, "events": events, "diagnoses": diags, "advice": advice,
                        "summary": {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                                    for k, v in r.to_dict().items()}}

    # Comparator：跨运行
    ok = summary[~summary["diverged"]].sort_values("best_val")
    lr_sens = summary[(summary["batch"] == 64) & (summary["layers"] == 6) & (summary["warmup"] > 1) &
                      (summary["clip"] > 0) & (summary["dropout"] > 0) & ~summary["run"].str.contains("fault|crash")]
    comp = {"ranking": ok[["run", "best_val", "best_step", "gen_gap"]].round(4).to_dict("records"),
            "lr_sensitivity": lr_sens[["lr", "best_val"]].sort_values("lr").round(4).to_dict("records"),
            "fleet_best": float(fleet_best)}
    tr.log("Comparator", "compare", comp)

    # Reporter
    report = make_report(results, comp)
    polished = llm("你是大模型训练诊断专家。只依据下列 JSON 证据，用中文写一段不超过 400 字的训练健康简报，"
                   "不得引入证据之外的数字：\n" + report[:6000])
    tr.log("Reporter", "report", {"mode": "llm" if polished else "template", "chars": len(polished or report)})
    return results, comp, (polished or report), tr.items


def make_report(results, comp):
    L = ["# TrainScope 训练健康诊断报告", ""]
    L.append(f"共分析 {len(results)} 个运行；最佳验证损失 {comp['fleet_best']:.4f}（{comp['ranking'][0]['run']}）。")
    L.append("")
    for run, r in results.items():
        L.append(f"## {run} —— {r['health']}")
        s = r["summary"]
        L.append(f"- 最优验证损失 {s['best_val']:.4f} @ step {s['best_step']}，泛化差 {s['gen_gap']:.3f}" if s["best_val"] else "- 无有效验证结果")
        for d in r["diagnoses"]:
            ev, c = d["event"], d["causes"][0]
            L.append(f"- [{ev['severity']}] {ev['type']} @ {ev['start']}–{ev['end']}：最可能根因「{c['cause']}」"
                     f"（置信度 {c['confidence']:.0%}；依据：{'；'.join(c['why'])}）")
        for a in r["advice"]:
            L.append(f"  - 建议：{a['action']}")
        L.append("")
    return "\n".join(L)


def main():
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    D = os.path.join(root, "data", "processed")
    S = pd.read_pickle(f"{D}/steps.pkl"); E = pd.read_pickle(f"{D}/evals.pkl"); P = pd.read_pickle(f"{D}/probes.pkl")
    cfgs = json.load(open(f"{D}/configs.json")); summary = pd.read_csv(f"{D}/run_summary.csv")
    results, comp, report, trace = run_pipeline(S, E, P, cfgs, summary)
    json.dump({"results": results, "compare": comp, "report": report, "trace": trace},
              open(f"{D}/agent_output.json", "w"), ensure_ascii=False, indent=1, default=float)
    open(f"{D}/agent_report.md", "w").write(report)
    print(report)


if __name__ == "__main__":
    main()
