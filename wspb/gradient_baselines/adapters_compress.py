"""compress-trained: the original is in the workspace, so it labels as many inputs from the stated language as the
clock allows, and a model whose counted matrices are each a product of two thin factors is fitted to it.

The ranks are split two ways, the reference's (its unused parts out, the rest by its allocation scores, both worked
out on calib.json as the reference does) and the same fraction of every unit; the factors start at the truncated SVD
of the original (of the original with the unused parts removed, for the reference's split). Each split trains for a
quarter of the clock and the better one, by the adversary's own check, gets the rest. The loss is cross-entropy to
the original's argmax at the graded positions, since the bar is argmax agreement there. embed, pos and the norms stay
as delivered; unembed, b1 and b2 train freely; a matrix whose top singular value would pass the cap is clipped to it.
The free variant is the original itself: without the budget there is nothing to compress.
"""

from __future__ import annotations

import time

import numpy as np

from .. import nn, wio
from ..families import compress_trained as ct
from ..families.repair_trained import MARGIN, agreement, graded_set, predict
from . import Attack, adapter, core, fit
from .fit import Problem
from .losses import ce


def splits(p0, meta, X, M, budget) -> dict:
    """name -> (units' ranks, the parts removed first)."""
    gone = ct.unused(p0, meta, X, M)
    return {"reference": (ct.allocate(ct.scores(p0, meta, X, M), meta, budget, ct._units_of(gone)), gone),
            "even": (ct.uniform(meta, budget), [])}


def matrix_ranks(units: dict) -> dict:
    """A unit's rank on each counted matrix it spans: a head's q/k pair puts it on q and on k, and so on."""
    out = {}
    for (name, h), r in units.items():
        l, kind = name.split(".")
        for w in (kind if kind in ("qk", "vo") else [kind]):
            out[(f"{l}.{w}", h)] = r
    return out


def _leaves(p, meta, ranks):
    torch = core.T()
    counted = {k for k, _ in ct.counted(meta)}
    fixed = set(ct.frozen(meta))
    leaves = {k: torch.tensor(v, requires_grad=k not in fixed) for k, v in p.items() if k not in counted}
    for (k, h), r in ranks.items():
        W = ct._m(p, k, h)
        U, s, Vt = np.linalg.svd(W, full_matrices=False)
        rt = np.sqrt(s[:r])
        leaves[f"{k}|{h}|u"] = torch.tensor(U[:, :r] * rt, requires_grad=r > 0)
        leaves[f"{k}|{h}|v"] = torch.tensor(rt[:, None] * Vt[:r], requires_grad=r > 0)
    return leaves


def _reparam(meta):
    torch = core.T()

    def param(leaves):
        P = {k: v for k, v in leaves.items() if "|" not in k}
        mats = {}
        for name in leaves:
            if name.endswith("|u"):
                k, h, _ = name.split("|")
                mats.setdefault(k, {})[h] = leaves[name] @ leaves[f"{k}|{h}|v"]
        for k, hs in mats.items():
            P[k] = hs["None"] if "None" in hs else torch.stack([hs[str(h)] for h in range(len(hs))])
        return P
    return param


@adapter("compress-trained")
def compress_trained_(task) -> Attack:
    b = ct.build(task.seed, task.split)
    lang = b["lang"]
    p0, meta = wio.parse(task.files["model.safetensors"])
    X, M = b["calib"]
    budget = task.info["budget"]
    top0 = ct.tops(p0, meta)
    val = [graded_set(lang, p0, meta, s, 1000) for s in (991, 992)]

    def rows(g, n):
        x, _, m = lang.sample(g, max(n, 1))
        y, margin = predict(p0, meta, x)
        return {"x": x, "y": y, "m": m & (margin >= MARGIN)}

    def score(p):
        return min(agreement(p, meta, *v) for v in val)

    def clip(p):   # the growth clause, as a projection: singular values over the cap come down to it
        q = dict(p)
        for (k, h), t in top0.items():
            W = ct._m(p, k, h)
            U, s, Vt = np.linalg.svd(W, full_matrices=False)
            if s[0] > ct.CAP * t * (1 - 1e-9):
                W = (U * np.minimum(s, ct.CAP * t * (1 - 1e-9))) @ Vt
                if h is None:
                    q[k] = W
                else:
                    q[k] = q[k].copy()
                    q[k][h] = W
        return q

    prob = Problem(meta, p0, rows, ce, score, goal=1.0, lr=1e-3, batch_size=128, reparam=_reparam(meta),
                   finalize=clip, restarts=1,
                   notes="free is the original; constrained trains rank-r factors of every counted matrix")

    def custom(seconds: float) -> fit.Run:
        t0 = time.time()
        found, tried = {}, {}
        for name, (units, gone) in splits(p0, meta, X, M, budget).items():
            start = ct.removed(p0, meta, X, gone) if gone else p0
            found[name] = (_leaves(start, meta, matrix_ranks(units)), units)
        left = seconds - (time.time() - t0)
        every = max(4.0, seconds / 120)
        for i, (name, (leaves, _)) in enumerate(found.items()):
            tried[name] = fit.train(prob, leaves, lambda lv=leaves: prob.reparam(lv), seconds=left / 4, seed=i,
                                    lr=prob.lr, every=every)
        win = max(tried, key=lambda k: tried[k].own)
        leaves = found[win][0]
        more = fit.train(prob, leaves, lambda: prob.reparam(leaves), seconds=seconds - (time.time() - t0), seed=7,
                         lr=prob.lr / 2, every=every)
        best = more if more.own >= tried[win].own else tried[win]
        best.steps = sum(r.steps for r in tried.values()) + more.steps
        best.seconds = time.time() - t0
        rank = ct.measure(best.params, p0, meta, top0)["rank"]
        prob.notes = (f"split {win} won ({', '.join(f'{k} own {r.own:.4f} in {r.steps} steps' for k, r in tried.items())}"
                      f"); after the rest of the clock own {more.own:.4f}; rank {rank} of {budget}")
        return best

    def behaviour(res):
        m = res.metrics
        if "agreement" not in m:
            return None, None, ct.BAR
        return m["agreement"] >= ct.BAR, float(m["agreement"]), ct.BAR

    return Attack(prob, behaviour, custom=custom, free_is_start=True,
                  data="unlimited inputs from the stated language labelled by the original's argmax at the graded "
                       "positions (the original is in the workspace)")
