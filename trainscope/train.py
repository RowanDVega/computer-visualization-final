"""
TrainScope · 带全量探针的字符级 GPT 训练脚本
================================================
在真实训练过程中采集多维监控指标，作为可视化系统的数据源：

  * scalars.jsonl  每步：loss / lr / 全局梯度范数(裁剪前) / 吞吐 / 显存 / 步耗时
  * layers.jsonl   每步：各模块梯度范数；每 10 步：各模块权重范数与更新比 ||ΔW||/||W||
  * probes.jsonl   每 50 步：固定探针批次上的逐层逐头注意力熵、残差流 RMS 与最大激活
  * evals.jsonl    每 100 步：训练集 / 验证集损失（20 个批次均值）
  * embed_*.npy    训练结束时的字符嵌入矩阵

支持故障注入（学习率尖峰、数据污染）与"崩溃-断点续训"，以产生真实的异常与脏日志。
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- model
class Attn(nn.Module):
    def __init__(s, d, h, drop):
        super().__init__()
        s.h, s.qkv, s.proj, s.drop = h, nn.Linear(d, 3 * d), nn.Linear(d, d), drop

    def forward(s, x, return_att=False):
        B, T, C = x.shape
        q, k, v = s.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, s.h, C // s.h).transpose(1, 2) for t in (q, k, v))
        if return_att:  # 显式计算注意力矩阵，仅用于探针
            att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            att = att.masked_fill(~mask, float("-inf")).softmax(-1)
            y = att @ v
        else:
            att = None
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               dropout_p=s.drop if s.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return s.proj(y), att


class Block(nn.Module):
    def __init__(s, d, h, drop):
        super().__init__()
        s.ln1, s.attn, s.ln2 = nn.LayerNorm(d), Attn(d, h, drop), nn.LayerNorm(d)
        s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(drop))

    def forward(s, x, return_att=False):
        a, att = s.attn(s.ln1(x), return_att)
        x = x + a
        x = x + s.mlp(s.ln2(x))
        return x, att


class GPT(nn.Module):
    def __init__(s, vocab, block, L, d, h, drop):
        super().__init__()
        s.tok, s.pos = nn.Embedding(vocab, d), nn.Embedding(block, d)
        s.blocks = nn.ModuleList([Block(d, h, drop) for _ in range(L)])
        s.ln_f, s.head = nn.LayerNorm(d), nn.Linear(d, vocab, bias=False)
        s.drop = nn.Dropout(drop)
        s.apply(s._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(s, idx, tgt=None, probe=False):
        B, T = idx.shape
        x = s.drop(s.tok(idx) + s.pos(torch.arange(T, device=idx.device)))
        atts, acts = [], []
        for b in s.blocks:
            x, att = b(x, probe)
            if probe:
                atts.append(att)
                acts.append(x)
        logits = s.head(s.ln_f(x))
        loss = None if tgt is None else F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
        return logits, loss, atts, acts


def module_groups(model):
    """把参数归入可解释的模块组：embed / L{i}.attn / L{i}.mlp / head"""
    g = {}
    for n, p in model.named_parameters():
        if n.startswith(("tok", "pos")):
            k = "embed"
        elif n.startswith("blocks"):
            i = n.split(".")[1]
            k = f"L{i}.attn" if (".attn." in n or ".ln1." in n) else f"L{i}.mlp"
        else:
            k = "head"
        g.setdefault(k, []).append(p)
    return g


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="data/runs")
    ap.add_argument("--data", default="data/raw/input.txt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--clip", type=float, default=1.0)  # <=0 表示不裁剪
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--wd", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1337)
    # 故障注入
    ap.add_argument("--lr_spike", default="")      # "start:end:mult"
    ap.add_argument("--corrupt", default="")       # "start:end"  该区间批次替换为随机字符
    ap.add_argument("--crash_at", type=int, default=-1)   # 在该步模拟进程崩溃
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ckpt_every", type=int, default=500)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    out = os.path.join(a.out, a.run); os.makedirs(out, exist_ok=True)

    text = open(a.data, encoding="utf-8").read()
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data)); tr, va = data[:n], data[n:]
    V = len(chars)

    gen = torch.Generator().manual_seed(a.seed)

    def batch(split, bs=a.batch, g=gen):
        d = tr if split == "train" else va
        ix = torch.randint(len(d) - a.block - 1, (bs,), generator=g)
        x = torch.stack([d[i:i + a.block] for i in ix]); y = torch.stack([d[i + 1:i + 1 + a.block] for i in ix])
        return x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)

    probe_x, _ = batch("val", 8, torch.Generator().manual_seed(7))  # 固定探针批次

    model = GPT(V, a.block, a.layers, a.d, a.heads, a.dropout).to(dev)
    groups = module_groups(model)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd},
                             {"params": nodecay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95))

    spike = tuple(map(float, a.lr_spike.split(":"))) if a.lr_spike else None
    corrupt = tuple(map(int, a.corrupt.split(":"))) if a.corrupt else None

    def lr_at(s):
        base = a.lr * (s + 1) / a.warmup if s < a.warmup else \
            0.1 * a.lr + 0.9 * a.lr * 0.5 * (1 + math.cos(math.pi * (s - a.warmup) / max(1, a.steps - a.warmup)))
        if spike and spike[0] <= s < spike[1]:
            base *= spike[2]
        return base

    start = 0
    ck = os.path.join(out, "ckpt.pt")
    if a.resume and os.path.exists(ck):
        st = torch.load(ck, map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        gen.set_state(st["gen"].cpu()); start = st["step"]
        print(f"[resume] from step {start}")

    mode = "a" if a.resume else "w"
    fs = {k: open(os.path.join(out, f"{k}.jsonl"), mode) for k in ("scalars", "layers", "probes", "evals")}
    json.dump(vars(a) | {"vocab": V, "n_params": sum(p.numel() for p in model.parameters()), "chars": chars},
              open(os.path.join(out, "config.json"), "w"), ensure_ascii=False, indent=1)

    @torch.no_grad()
    def evaluate():
        model.eval(); r = {}
        for sp in ("train", "val"):
            g = torch.Generator().manual_seed(99)
            ls = [model(*batch(sp, a.batch, g))[1].item() for _ in range(20)]
            r[sp] = float(np.mean(ls))
        model.train(); return r

    @torch.no_grad()
    def probe():
        model.eval()
        _, _, atts, acts = model(probe_x, probe=True)
        ent = [(-(at.clamp_min(1e-12).log() * at).sum(-1)).mean(dim=(0, 2)).tolist() for at in atts]  # [L][H]
        rms = [x.float().pow(2).mean().sqrt().item() for x in acts]
        mx = [x.float().abs().max().item() for x in acts]
        model.train(); return ent, rms, mx

    model.train()
    t_last = time.time()
    for step in range(start, a.steps):
        if step == a.crash_at and not a.resume:
            print(f"[crash] simulated crash at step {step}"); os._exit(1)
        lr = lr_at(step)
        for pg in opt.param_groups: pg["lr"] = lr
        x, y = batch("train")
        if corrupt and corrupt[0] <= step < corrupt[1]:
            y = torch.randint(0, V, y.shape, device=dev)  # 标签污染：模拟上游数据管道故障
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss, _, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()

        gnorm_mod = {k: torch.sqrt(sum((p.grad.float() ** 2).sum() for p in ps if p.grad is not None)).item()
                     for k, ps in groups.items()}
        gnorm = math.sqrt(sum(v ** 2 for v in gnorm_mod.values()))
        if a.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)

        track = step % 10 == 0
        if track:
            before = {k: [p.detach().clone() for p in ps] for k, ps in groups.items()}
        opt.step()
        rec_l = {"step": step, "gnorm": gnorm_mod}
        if track:
            wn, ur = {}, {}
            for k, ps in groups.items():
                w2 = sum((p.float() ** 2).sum() for p in ps)
                d2 = sum(((p.float() - b.float()) ** 2).sum() for p, b in zip(ps, before[k]))
                wn[k] = math.sqrt(w2.item()); ur[k] = math.sqrt(d2.item()) / (wn[k] + 1e-12)
            rec_l |= {"wnorm": wn, "update_ratio": ur}
            del before

        torch.cuda.synchronize()
        now = time.time(); dt = now - t_last; t_last = now
        lv = loss.item()
        fs["scalars"].write(json.dumps({
            "step": step, "loss": lv if math.isfinite(lv) else str(lv), "lr": lr, "grad_norm": gnorm,
            "step_time": dt, "tok_per_s": a.batch * a.block / dt,
            "gpu_mem_mb": torch.cuda.max_memory_allocated() / 2 ** 20, "wall": now}) + "\n")
        fs["layers"].write(json.dumps(rec_l) + "\n")

        if step % 50 == 0 or step == a.steps - 1:
            ent, rms, mx = probe()
            fs["probes"].write(json.dumps({"step": step, "attn_entropy": ent, "act_rms": rms, "act_max": mx}) + "\n")
        if step % 100 == 0 or step == a.steps - 1:
            e = evaluate()
            fs["evals"].write(json.dumps({"step": step, "train_loss": e["train"], "val_loss": e["val"]}) + "\n")
            print(f"{a.run} step {step:5d} loss {lv:.4f} val {e['val']:.4f} gn {gnorm:.2f} lr {lr:.2e}", flush=True)
            for f in fs.values(): f.flush()
        if step > 0 and step % a.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "gen": gen.get_state(),
                        "step": step + 1}, ck)

    np.save(os.path.join(out, "embed_tok.npy"), model.tok.weight.detach().float().cpu().numpy())
    # 生成一段样例文本，便于定性评估
    model.eval(); idx = torch.zeros((1, 1), dtype=torch.long, device=dev)
    with torch.no_grad():
        for _ in range(400):
            lg = model(idx[:, -a.block:])[0][:, -1, :] / 0.8
            idx = torch.cat([idx, torch.multinomial(lg.softmax(-1), 1)], 1)
    open(os.path.join(out, "sample.txt"), "w").write("".join(chars[i] for i in idx[0].tolist()))
    if os.path.exists(ck): os.remove(ck)
    for f in fs.values(): f.close()
    print("done", a.run)


if __name__ == "__main__":
    main()
