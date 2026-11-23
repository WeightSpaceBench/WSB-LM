"""Fits that change only a few late tensors of SmolLM2. Nothing below the lowest changed layer depends on the change,
so the residual stream entering that layer is computed once per sequence and a step runs only the layers above it,
with logits taken only where a loss reads them.

A change is either a free matrix per tensor or, for a ridge-style write, D = Z^T A with A fixed (rows: the keys
through the inverse of a protection covariance) and the targets Z trained. Either way each change is projected back
inside its norm cap after every step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .. import llm
from . import core, llmfit

EPS = llm.CONFIG["eps"]


@dataclass
class Chunk:
    x0: Any                 # [B, T, d], the residual entering the lowest changed layer, from the delivered weights
    at: Any                 # [B, T] bool, the positions a loss reads
    tgt: Any = None         # [B, T] token ids (cross-entropy)
    top: Any = None         # [B, T, k] the delivered model's top tokens there, and their log-probabilities
    lp: Any = None


def prepare(P0: dict, seqs: list[list[int]], at: list[list[int]], start: int, *, tgt: list[list[int]] | None = None,
            k: int = 16, chunk: int = 16, cov: dict | None = None, weight: float = 1.0) -> list[Chunk]:
    """Chunks of right-padded sequences, with the residual at `start` cached. at[i] lists the positions of seqs[i]
    the loss reads; tgt[i] the target token at each (else the delivered model's top-k there is the target). If cov
    is a dict {layer: [ffn, ffn]}, weight * a a^T over every real position's MLP input at those layers is added."""
    torch = core.T()
    out = []
    with torch.no_grad():
        for b in range(0, len(seqs), chunk):
            ids, real = llmfit.pad(seqs[b:b + chunk])
            m = torch.zeros_like(real)
            for i, ps in enumerate(at[b:b + chunk]):
                m[i, torch.as_tensor(ps, dtype=torch.long)] = True
            taps: dict = {}
            lg = llmfit.forward(P0, ids, taps=taps)
            c = Chunk(taps[f"resid.{start}"].clone(), m)
            if tgt is not None:
                t = torch.zeros_like(ids)
                for i, (ps, ts) in enumerate(zip(at[b:b + chunk], tgt[b:b + chunk])):
                    t[i, torch.as_tensor(ps, dtype=torch.long)] = torch.as_tensor(ts, dtype=torch.long)
                c.tgt = t
            else:
                lp, top = torch.log_softmax(lg[m], -1).topk(k, -1)
                c.top, c.lp = top, lp
            if cov is not None:
                for L in cov:
                    a = taps[f"act.{L}"][real].double()
                    cov[L] += weight * (a.T @ a)
            out.append(c)
            del lg, taps
    return out


def _norm(x, w):
    return x / (x * x).mean(-1, keepdim=True).add(EPS).sqrt() * w


def nll(P: dict, c: Chunk, start: int):
    """Per position, cross-entropy to the target token, or the KL from the delivered model's top-k (with the rest
    of its mass as one more outcome) to the live model's."""
    torch = core.T()
    h = llmfit.forward(P, None, x0=c.x0, start=start, hidden=True)[c.at]
    lsm = torch.log_softmax(_norm(h, P["model.norm.weight"]) @ P["model.embed_tokens.weight"].T, -1)
    if c.tgt is not None:
        return -lsm.gather(-1, c.tgt[c.at][:, None]).squeeze(-1)
    q = lsm.gather(-1, c.top)
    p = c.lp.exp()
    r0 = (1 - p.sum(-1)).clamp(min=1e-6)
    r = (1 - q.exp().sum(-1)).clamp(min=1e-6)
    return (p * (c.lp - q)).sum(-1) + r0 * (r0.log() - r.log())


class Pool:
    """A list of chunks, a few drawn at random per step."""

    def __init__(self, chunks: list[Chunk], per_step: int, seed: int = 0):
        self.chunks, self.per, self.g = chunks, per_step, np.random.default_rng(seed)

    def draw(self) -> list[Chunk]:
        return [self.chunks[i] for i in self.g.choice(len(self.chunks), min(self.per, len(self.chunks)), False)]


def ridge_basis(K: Any, C: Any, omega: float, mu: float = 1e-2):
    """A = K (C + omega K^T K + mu tr/n I)^-1, rows scaled to unit norm, so D = Z^T A is the ridge write
    omega R^T K M^-1 for targets R (up to the row scale) and Z can be trained in their place."""
    torch = core.T()
    K = K.double()
    M = C + omega * K.T @ K
    M = M + mu * torch.trace(M) / len(M) * torch.eye(len(M), dtype=M.dtype)
    A = torch.linalg.solve(M, K.T).T
    return (A / A.norm(dim=1, keepdim=True).clamp(min=1e-12)).float()


@dataclass
class Tail:
    start: dict[str, np.ndarray]          # the delivered weights
    names: list[str]                      # the tensors that may change
    loss: Callable[[dict], Any]           # from the live tensors
    score: Callable[[dict], tuple[float, dict]]   # own check on numpy weights: (0..1, what it measured)
    max_rel: float | None = None
    basis: dict[str, Any] | None = None   # name -> A [n, in]: the change is Z^T A
    lr: float = 1e-4
    goal: float = 1.0
    every: float = 120.0
    how: str = ""
    leaves: dict | None = None            # or parameters of its own, and live(P0) -> the live tensors from them
    live: Callable[[dict], dict] | None = None
    decay: float = 0.0                    # AdamW's, on the changes: back towards the delivered weights
    check_first: bool = False             # score the delivered weights too, so a fit that only gets worse keeps them


def fit(spec: Tail, seconds: float) -> llmfit.Fit:
    torch = core.T()
    t0 = time.time()
    P0 = llmfit.tensors(spec.start)
    leaves = dict(spec.leaves or {})
    for k in ([] if spec.live else spec.names):
        if spec.basis and k in spec.basis:
            leaves[k] = torch.zeros(spec.basis[k].shape[0], P0[k].shape[0], requires_grad=True)
        else:
            leaves[k] = torch.zeros_like(P0[k], requires_grad=True)
    caps = {k: 0.97 * spec.max_rel * float(P0[k].norm()) for k in spec.names} if spec.max_rel else {}

    def delta(k):
        return leaves[k].T @ spec.basis[k] if spec.basis and k in spec.basis else leaves[k]

    def live():
        if spec.live:
            return spec.live(P0)
        P = dict(P0)
        for k in spec.names:
            P[k] = P0[k] + delta(k)
        return P

    def numpy_weights():
        with torch.no_grad():
            P = live()
            return {**spec.start, **{k: P[k].numpy().astype(np.float32) for k in spec.names}}

    opt = (torch.optim.AdamW(list(leaves.values()), lr=spec.lr, weight_decay=spec.decay) if spec.decay else
           torch.optim.Adam(list(leaves.values()), lr=spec.lr))
    best, best_w, best_m, best_step, steps, last, run, trace = -1.0, None, {}, 0, 0, time.time(), [], []

    def check():
        nonlocal best, best_w, best_m, best_step
        w = numpy_weights()
        s, m = spec.score(w)
        trace.append(dict(step=steps, t=round(time.time() - t0), own=round(s, 3),
                          loss=round(float(np.mean(run[-20:])), 4) if run else None,
                          **{a: round(b, 3) for a, b in m.items()}))
        if s >= best:
            best, best_w, best_m, best_step = s, {k: w[k] for k in spec.names}, m, steps
        return s

    if spec.check_first:
        check()

    while time.time() - t0 < seconds:
        loss = spec.loss(live())
        if not torch.isfinite(loss):
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
        steps += 1
        run.append(float(loss.detach()))
        if caps:
            with torch.no_grad():
                for k in spec.names:
                    n = float(delta(k).norm())
                    if n > caps[k]:
                        leaves[k].mul_(caps[k] / n)
        if time.time() - last > spec.every:
            if check() >= spec.goal:
                break
            last = time.time()
    if not trace or trace[-1]["step"] != steps:
        check()
    out = {**spec.start, **best_w}
    f = llmfit.Fit(out, best, time.time() - t0, steps, list(spec.names))
    f.trace = trace
    f.measured = dict(best_m, step=best_step)
    return f
