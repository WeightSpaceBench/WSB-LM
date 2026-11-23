"""SmolLM2-135M (wspb.llm) again in torch, float32, and the fitting loop the language-model families share.

The model is too big for the generic loop's float64 copies, so this one keeps the delivered weights as they are and
gives Adam only the tensors an adapter names. An adapter supplies the loss (as a function of the live tensors), the
tensors it may move, and a projection applied after every step (a norm bound, a rank, nothing).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .. import llm
from . import core


def tensors(p: dict[str, np.ndarray]) -> dict:
    """The weights as torch tensors, sharing memory with the arrays where it can (the model is 0.5 GB a copy)."""
    import warnings
    torch = core.T()
    with warnings.catch_warnings():     # read-only views are fine: nothing here writes to the delivered weights
        warnings.simplefilter("ignore")
        return {k: torch.from_numpy(v if v.dtype == np.float32 else np.array(v, dtype=np.float32)) for k, v in p.items()}


def forward(P: dict, ids, *, taps: dict | None = None, hidden: bool = False, extra=None, cfg: dict = llm.CONFIG,
            start: int = 0, x0=None):
    """Logits [B, T, vocab] for token ids [B, T] (right-padded batches are fine: attention is causal). `taps` is
    filled like wspb.llm.forward's; extra(h, layer) may return a vector to add to the residual after that layer.
    With x0 (the residual entering layer `start`, [B, T, d]) the layers below are skipped and ids is unused."""
    torch = core.T()
    B, Tn = (ids if x0 is None else x0).shape[:2]
    H, KV, hd = cfg["heads"], cfg["kv_heads"], cfg["head_dim"]
    G = H // KV
    inv = 1.0 / cfg["rope_theta"] ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd)
    ang = torch.arange(Tn, dtype=torch.float64)[:, None] * inv
    ang = torch.cat([ang, ang], -1)
    cos, sin = torch.cos(ang).float(), torch.sin(ang).float()

    def rope(x):
        h = x.shape[-1] // 2
        return x * cos + torch.cat([-x[..., h:], x[..., :h]], -1) * sin

    def norm(x, w):
        return x / torch.sqrt((x * x).mean(-1, keepdim=True) + cfg["eps"]) * w

    mask = torch.tril(torch.ones(Tn, Tn, dtype=torch.bool))
    x = P["model.embed_tokens.weight"][ids] if x0 is None else x0
    for i in range(start, cfg["layers"]):
        L = f"model.layers.{i}."
        if taps is not None:
            taps[f"resid.{i}"] = x
        h = norm(x, P[L + "input_layernorm.weight"])
        q = (h @ P[L + "self_attn.q_proj.weight"].T).reshape(B, Tn, H, hd).transpose(1, 2)
        k = (h @ P[L + "self_attn.k_proj.weight"].T).reshape(B, Tn, KV, hd).transpose(1, 2)
        v = (h @ P[L + "self_attn.v_proj.weight"].T).reshape(B, Tn, KV, hd).transpose(1, 2)
        q, k = rope(q), rope(k)
        k, v = k.repeat_interleave(G, 1), v.repeat_interleave(G, 1)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
        s = s.masked_fill(~mask, float("-inf"))
        a = (torch.softmax(s, -1) @ v).transpose(1, 2).reshape(B, Tn, H * hd) @ P[L + "self_attn.o_proj.weight"].T
        x = x + a
        h = norm(x, P[L + "post_attention_layernorm.weight"])
        if taps is not None:
            taps[f"mlp_in.{i}"] = h
        g = h @ P[L + "mlp.gate_proj.weight"].T
        act = g * torch.sigmoid(g) * (h @ P[L + "mlp.up_proj.weight"].T)
        if taps is not None:
            taps[f"attn.{i}"], taps[f"act.{i}"] = a, act
        x = x + act @ P[L + "mlp.down_proj.weight"].T
    if taps is not None:
        taps["final"] = x
    if extra is not None:
        x = x + extra
    if hidden:
        return x
    return norm(x, P["model.norm.weight"]) @ P["model.embed_tokens.weight"].T


def pad(seqs: list[list[int]]):
    """Right-padded ids [B, T] and a bool mask of the real positions."""
    torch = core.T()
    Tn = max(len(s) for s in seqs)
    ids = torch.zeros(len(seqs), Tn, dtype=torch.long)
    real = torch.zeros(len(seqs), Tn, dtype=torch.bool)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        real[i, :len(s)] = True
    return ids, real


@dataclass
class Spec:
    """One language-model task as a fitting problem, already inside its budget: which tensors Adam may consider,
    how many of them it may keep (by entries or by count), and what each change is held to."""
    start: dict[str, np.ndarray]
    loss: Callable[[dict], Any]                       # scalar, from the live tensors
    score: Callable[[dict], float]                    # own check, 0..1, from the live tensors (called without grad)
    candidates: list[str]
    max_entries: int | None = None                    # an entry budget: whole tensors are chosen until it is full
    max_tensors: int | None = None
    max_rel: float | None = None                      # each change's Frobenius norm, as a share of its tensor's
    rank: int | None = None                           # each change's rank (trained as two factors)
    aim: Callable[[dict], Any] | None = None          # the part of the loss whose gradient picks the tensors
    lr: float = 1e-5
    goal: float = 1.0
    how: str = ""


@dataclass
class Fit:
    params: dict[str, np.ndarray]
    own: float
    seconds: float
    steps: int
    chosen: list[str] = field(default_factory=list)


def choose(spec: Spec, P0: dict, rounds: int = 2) -> tuple[list[str], dict]:
    """The tensors the loss gradient at the start is largest on, as many as the budget holds."""
    torch = core.T()
    for k in spec.candidates:
        P0[k].requires_grad_(True)
    size = {k: 0.0 for k in spec.candidates}
    keep = {}
    for _ in range(rounds):
        g = torch.autograd.grad((spec.aim or spec.loss)(P0), [P0[k] for k in spec.candidates], allow_unused=True)
        for k, x in zip(spec.candidates, g):
            if x is not None:
                size[k] += float(x.norm())
                keep[k] = x
        del g
    for k in spec.candidates:
        P0[k].requires_grad_(False)
    order = sorted(spec.candidates, key=lambda k: -size[k])
    chosen, used = [], 0
    for k in order:
        if spec.max_tensors is not None and len(chosen) >= spec.max_tensors:
            break
        if spec.max_entries is not None and used + P0[k].numel() > spec.max_entries:
            continue
        if size[k] == 0.0:
            break
        chosen.append(k)
        used += P0[k].numel()
    return chosen, {k: keep[k] for k in chosen}     # the last round's gradients of the chosen ones, for rank factors


def fit(spec: Spec, seconds: float, every: float = 45.0) -> Fit:
    torch = core.T()
    t0 = time.time()
    P0 = tensors(spec.start)
    chosen, G = choose(spec, P0)
    leaves, P = {}, dict(P0)
    for k in chosen:
        if spec.rank and P0[k].dim() == 2:   # factors: the gradient's top right singular vectors in, zeros out
            _, _, Vh = torch.linalg.svd(G[k], full_matrices=False)
            leaves[k + "|v"] = Vh[:spec.rank].clone().requires_grad_(True)
            leaves[k + "|u"] = torch.zeros(P0[k].shape[0], spec.rank, requires_grad=True)
        else:
            leaves[k] = P0[k].clone().requires_grad_(True)
    del G

    def live():
        for k in chosen:
            P[k] = P0[k] + leaves[k + "|u"] @ leaves[k + "|v"] if k + "|u" in leaves else leaves[k]
        return P

    opt = torch.optim.Adam(list(leaves.values()), lr=spec.lr * (30 if spec.rank else 1))
    best, best_w, steps, last = -1.0, None, 0, time.time()

    def check():
        nonlocal best, best_w
        with torch.no_grad():
            s = float(spec.score(live()))
            if s >= best:
                best, best_w = s, {k: live()[k].detach().clone() for k in chosen}
        return s

    while time.time() - t0 < seconds:
        loss = spec.loss(live())
        if not torch.isfinite(loss):
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
        steps += 1
        if spec.max_rel is not None:
            with torch.no_grad():
                for k in chosen:
                    d = leaves[k] - P0[k]
                    cap = 0.98 * spec.max_rel * float(P0[k].norm())
                    n = float(d.norm())
                    if n > cap:
                        leaves[k].copy_(P0[k] + d * (cap / n))
        if time.time() - last > every:
            last = time.time()
            if check() >= spec.goal:
                break
    check()
    out = dict(spec.start)
    for k in chosen:
        out[k] = best_w[k].numpy().astype(np.float32)
    return Fit(out, best, time.time() - t0, steps, chosen)


def behaviour_texts(beh: dict):
    """The original's top-10 next-token logits on the texts of a behaviour.json, as a distillation loss, and how
    often a model's top token is the original's there."""
    torch = core.T()
    ids, real = pad(beh["tokens"])
    top = torch.zeros(*ids.shape, 10, dtype=torch.long)
    val = torch.zeros(*ids.shape, 10)
    for i, (a, b) in enumerate(zip(beh["top_ids"], beh["top_logits"])):
        top[i, :len(a)], val[i, :len(b)] = torch.tensor(a), torch.tensor(b)

    g = np.random.default_rng(1)

    def loss(P, k: int = 2):   # k of the texts per call: a pass over all of them is most of a step's cost
        i = torch.as_tensor(g.choice(len(ids), min(k, len(ids)), replace=False))
        lg = forward(P, ids[i])
        z = lg.gather(-1, top[i])
        gap = (z - z.mean(-1, keepdim=True)) - (val[i] - val[i].mean(-1, keepdim=True))
        ce = -torch.log_softmax(lg, -1).gather(-1, top[i][..., :1]).squeeze(-1)
        return ((gap ** 2).mean(-1) + ce)[real[i]].mean()

    def agree(P):
        return float((forward(P, ids).argmax(-1) == top[..., 0])[real].float().mean())
    return loss, agree


def teach(P0: dict, prompts: list[list[int]], more: int = 12, batch: int = 16) -> dict:
    """A behaviour.json of the adversary's own: a model the agent holds, run on prompts the adversary wrote and on its
    own greedy continuation of each, with its top-10 logits at every position."""
    torch = core.T()
    out = {"tokens": [], "top_ids": [], "top_logits": []}
    with torch.no_grad():
        for b in range(0, len(prompts), batch):
            seqs = [list(p) for p in prompts[b:b + batch]]
            for _ in range(more):      # greedy, a token at a time (short prompts: a cache isn't worth it)
                ids, real = pad(seqs)
                nxt = forward(P0, ids)[torch.arange(len(seqs)), real.sum(1) - 1].argmax(-1)
                seqs = [s + [int(t)] for s, t in zip(seqs, nxt)]
            ids, real = pad(seqs)
            val, top = forward(P0, ids).topk(10, -1)
            for i, s in enumerate(seqs):
                out["tokens"].append(s)
                out["top_ids"].append(top[i, :len(s)].tolist())
                out["top_logits"].append(val[i, :len(s)].tolist())
    return out
