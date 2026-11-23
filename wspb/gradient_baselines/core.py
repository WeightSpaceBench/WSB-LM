"""The reference forward pass (wspb.nn) again in torch, so a gradient adversary can train through it.

Hard units (step, sign) have no gradient. `soft` picks the stand-in: ("sig", beta) replaces them with a sigmoid of
slope beta; ("ste", beta) keeps the hard value going forward and uses that sigmoid's slope going back. soft=None is
the exact forward pass. torch is imported inside the functions: nothing the benchmark loads needs it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def T():
    import torch
    return torch


def to_t(p: dict[str, np.ndarray], grad: bool = False) -> dict:
    torch = T()
    return {k: torch.tensor(np.asarray(v, dtype=np.float64), requires_grad=grad) for k, v in p.items()}


def to_np(P: dict) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy().astype(np.float64) for k, v in P.items()}


def _hard(x, soft, lo: float):
    """step (lo=0) or sign (lo=-1) of x, or its stand-in."""
    torch = T()
    hard = torch.where(x > 0, torch.ones_like(x), torch.full_like(x, lo))
    if soft is None:
        return hard
    mode, beta = soft
    s = torch.sigmoid(beta * x) * (1 - lo) + lo
    return s if mode == "sig" else s + (hard - s).detach()


def act(name: str, x, soft=None):
    torch = T()
    if name == "relu":
        return torch.relu(x)
    if name == "tanh":
        return torch.tanh(x)
    if name == "identity":
        return x
    if name == "sigmoid":
        return torch.sigmoid(x)
    if name == "silu":
        return x * torch.sigmoid(x)
    if name == "gelu":
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
    if name == "step":
        return _hard(x, soft, 0.0)
    if name == "sign":
        return _hard(x, soft, -1.0)
    raise ValueError(name)


def mlp(P, meta, x, soft=None, layers: bool = False):
    h, hs = x, []
    for i, a in enumerate(meta["acts"]):
        h = act(a, h @ P[f"l{i}.w"].T + P[f"l{i}.b"], soft)
        hs.append(h)
    return (h, hs) if layers else h


def _norm(P, meta, name, x):
    kind = meta.get("norm", "none")
    if kind == "none":
        return x
    eps = meta.get("eps", 1e-5)
    if kind == "rms":
        return x / (x * x).mean(-1, keepdim=True).add(eps).sqrt() * P[f"{name}.g"]
    c = x - x.mean(-1, keepdim=True)
    return c / (c * c).mean(-1, keepdim=True).add(eps).sqrt() * P[f"{name}.g"] + P[f"{name}.b"]


def _rope(x, base: float):
    torch = T()
    t, dh = x.shape[2], x.shape[3]
    ang = torch.arange(t, dtype=x.dtype)[:, None] * base ** (-torch.arange(0, dh, 2, dtype=x.dtype) / dh)
    c, s = torch.cos(ang), torch.sin(ang)
    u, w = x[..., 0::2], x[..., 1::2]
    return torch.stack([u * c - w * s, u * s + w * c], -1).reshape(x.shape)


def _heads(x, w):
    n, t, d = x.shape
    G, _, dh = w.shape
    return (x @ w.permute(1, 0, 2).reshape(d, G * dh)).reshape(n, t, G, dh).permute(0, 2, 1, 3)


def tfm(P, meta, tokens, hidden: bool = False):
    """tokens: (N, T) long. Logits (N, T, V); with hidden=True also the residual stream after the embedding and
    after every attention and MLP sublayer."""
    torch = T()
    n, t = tokens.shape
    rope = meta.get("pos", "learned") == "rope"
    h = P["embed"][tokens] if rope else P["embed"][tokens] + P["pos"][:t]
    H, dh, d = meta["heads"], meta["dh"], meta["d"]
    rep = H // meta.get("kv_heads", H)
    kind = meta.get("mlp", "relu")
    mask = torch.triu(torch.full((t, t), float("-inf"), dtype=h.dtype), 1) if meta.get("causal") else None
    states = [h]
    for l in range(meta["layers"]):
        x = _norm(P, meta, f"L{l}.n1", h)
        q, k, v = (_heads(x, P[f"L{l}.{w}"]) for w in "qkv")
        if rope:
            q, k = _rope(q, meta.get("rope_base", 10000.0)), _rope(k, meta.get("rope_base", 10000.0))
        if rep > 1:
            k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        s = q @ k.transpose(-1, -2)
        if meta.get("scale"):
            s = s / math.sqrt(dh)
        if mask is not None:
            s = s + mask
        a = torch.softmax(s, -1)
        h = h + (a @ v).permute(0, 2, 1, 3).reshape(n, t, H * dh) @ P[f"L{l}.o"].reshape(H * dh, d)
        states.append(h)
        x = _norm(P, meta, f"L{l}.n2", h)
        u = x @ P[f"L{l}.mlp1"] + P[f"L{l}.b1"]
        if kind == "relu":
            u = torch.relu(u)
        elif kind == "gelu":
            u = act("gelu", u)
        else:
            u = act("silu", x @ P[f"L{l}.gate"] + P[f"L{l}.bg"]) * u
        h = h + u @ P[f"L{l}.mlp2"] + P[f"L{l}.b2"]
        states.append(h)
    logits = _norm(P, meta, "nf", h) @ P["unembed"]
    return (logits, states) if hidden else logits


def rnn(P, meta, x, soft=None):
    torch = T()
    n, steps, _ = x.shape
    H = meta["hidden"]
    h = P["h0"].expand(n, H)
    hs = []
    for t in range(steps):
        if meta.get("cell", "elman") == "elman":
            h = act(meta["act"], x[:, t] @ P["wx"].T + h @ P["wh"].T + P["b"], soft)
        else:
            gi, gh = x[:, t] @ P["wx"].T + P["bx"], h @ P["wh"].T + P["bh"]
            r, z = torch.sigmoid(gi[:, :H] + gh[:, :H]), torch.sigmoid(gi[:, H:2 * H] + gh[:, H:2 * H])
            c = torch.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
            h = (1.0 - z) * c + z * h
        hs.append(h)
    hs = torch.stack(hs, 1)
    return act(meta["out_act"], hs @ P["wo"].T + P["bo"], soft)


def cnn(P, meta, x, soft=None):
    torch = T()
    h = x
    for i, (_, _, s, a) in enumerate(meta["convs"]):
        h = act(a, torch.nn.functional.conv2d(h, P[f"c{i}.w"], P[f"c{i}.b"], stride=s), soft)
    h = h.amax(dim=(2, 3)) if meta.get("pool", "none") == "max" else h.reshape(h.shape[0], -1)
    for j, a in enumerate(meta["acts"]):
        h = act(a, h @ P[f"l{j}.w"].T + P[f"l{j}.b"], soft)
    return h


def moe(P, meta, x, soft=None):
    torch = T()
    r = x @ P["router.w"].T + P["router.b"]
    k = meta.get("topk", meta["experts"])
    hid = act(meta["act"], torch.einsum("ni,ehi->neh", x, P["w1"]) + P["b1"], soft)
    eo = torch.einsum("neh,emh->nem", hid, P["w2"]) + P["b2"]
    if k >= meta["experts"]:
        g = torch.softmax(r, -1)
    elif soft is not None and k == 1:   # top-1 routing passes no gradient to the router: a sharpened softmax does
        g = torch.softmax(r * soft[1], -1)
    else:   # the reference breaks ties toward the lower index; ties have measure zero while training
        idx = torch.topk(r, k, dim=1).indices
        keep = torch.zeros_like(r, dtype=torch.bool).scatter(1, idx, True)
        g = torch.softmax(r.masked_fill(~keep, float("-inf")), -1)
    return act(meta["out_act"], (g[..., None] * eo).sum(1), soft)


def resmlp(P, meta, x):
    torch = T()
    h = x @ P["win"] + P["bin"]
    for l in range(meta["layers"]):
        h = h + torch.relu(h @ P[f"L{l}.mlp1"] + P[f"L{l}.b1"]) @ P[f"L{l}.mlp2"]
    return h @ P["wout"] + P["bout"]


def forward(P, meta: dict[str, Any], x, soft=None):
    kind = meta["kind"]
    if kind == "tfm":
        return tfm(P, meta, x)
    if kind == "resmlp":
        return resmlp(P, meta, x)
    return {"mlp": mlp, "rnn": rnn, "cnn": cnn, "moe": moe}[kind](P, meta, x, soft)


def has_hard_units(meta: dict[str, Any]) -> bool:
    acts = list(meta.get("acts", [])) + [c[3] for c in meta.get("convs", [])] + [meta.get("act"), meta.get("out_act")]
    return any(a in ("step", "sign") for a in acts)


def as_input(meta: dict[str, Any], x: np.ndarray):
    torch = T()
    return torch.as_tensor(np.asarray(x), dtype=torch.long if meta["kind"] == "tfm" else torch.float64)
