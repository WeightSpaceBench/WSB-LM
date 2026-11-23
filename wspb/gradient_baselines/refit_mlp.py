"""compress-mlp's stronger adversary: the original is in the workspace, so it labels as many made-up inputs as the
clock allows, and the compressed model is fitted to it in activation space.

Layer by layer, the units kept are a set whose activations on those inputs are linearly independent (pivoted
Cholesky of the normalized Gram matrix; units that never fire, or that copy, cancel or sum to what others give, drop
out), and each kept unit's pre-activation, and each logit, is refitted by least squares onto the kept units below. A
layer that is on for every input is linear: it becomes identity copies of the units below and the next layer reads
through it. Then the dense refit is cut to the budget over a few rounds by first-order saliency (a weight's size x its
input's spread x the logits' sensitivity to its unit, which unit rescaling doesn't change), every row refitted on its
own support after each round, and Adam with the mask fixed polishes what is left. Nothing here reads the generator:
only the model, the sample and task.json.
"""

from __future__ import annotations

import time

import numpy as np

from .. import nn


def made_up(X: np.ndarray, g: np.random.Generator, n: int) -> np.ndarray:
    """Inputs like the sample's: binary rows with a few bits flipped; real rows moved part of the way to one of their
    nearest neighbours in the sample, with a little noise, clipped to [-1, 1]."""
    rows = X[g.integers(0, len(X), n)]
    if np.all((X == 0) | (X == 1)):
        return np.abs(rows - (g.random(rows.shape) < g.choice([0.0, 0.02, 0.05, 0.08, 0.12], (n, 1))))
    nb = np.argsort(((X[:, None] - X[None]) ** 2).sum(-1), 1)[:, 1:6]
    i = g.integers(0, len(X), n)
    t = g.uniform(0, 1, (n, 1))
    out = X[i] * (1 - t) + X[nb[i, g.integers(0, nb.shape[1], n)]] * t
    return np.clip(out + g.normal(0, 1, out.shape) * X.std(0) * g.choice([0.0, 0.03, 0.08], (n, 1)), -1, 1)


def layers(p, L, X):
    H, Z = [X], []
    for i in range(L):
        Z.append(H[-1] @ p[f"l{i}.w"].T + p[f"l{i}.b"])
        H.append(np.maximum(Z[-1], 0) if i < L - 1 else Z[-1])
    return H, Z


def select(A: np.ndarray, prefer: np.ndarray, tol: float = 1e-5) -> np.ndarray:
    """Columns of A whose centred activations are linearly independent, preferred ones pivoted first."""
    C = A - A.mean(0)
    nrm = np.linalg.norm(C, axis=0)
    live = np.flatnonzero(nrm > 1e-9 * max(float(nrm.max()), 1e-300))
    U = C[:, live] / nrm[live]
    G = U.T @ U
    d, pr = np.diag(G).copy(), 2.0 * prefer[live]
    F, picked = np.zeros((len(live), len(live))), []
    for k in range(len(live)):
        s = np.where(d > tol, d + pr, -np.inf)
        s[picked] = -np.inf
        j = int(np.argmax(s))
        if not np.isfinite(s[j]):
            break
        picked.append(j)
        F[:, k] = (G[:, j] - F[:, :k] @ F[j, :k]) / np.sqrt(d[j])
        d = d - F[:, k] ** 2
    return live[np.sort(np.array(picked, dtype=int))]


def _gram(h):
    A = np.c_[h, np.ones(len(h))]
    G = A.T @ A
    return A, G, 1e-10 * float(np.trace(G)) / len(G)


def linear(p, L, X) -> list[int]:
    """Hidden layers past the first that are on wherever they are ever on, over X: linear there."""
    Z = layers(p, L, X)[1]
    return [i for i in range(1, L - 1) if (Z[i][:, np.abs(Z[i]).max(0) > 0] > 0).all()]


def rebuild(p, L, X, fold=()):
    """p on independent units only, each kept row (and each logit) refitted onto the kept units below, and units the
    refit leaves (next to) nothing to say taken out; the layers in fold become copies of the units below."""
    H, Z = layers(p, L, X)
    keep, n = [], X.shape[1]
    for i in range(L - 1):
        keep.append(None if i in fold and len(p[f"l{i}.b"]) >= n else select(H[i + 1], (H[i + 1] == 0).mean(0) > 0.01))
        n = n if keep[-1] is None else len(keep[-1])
    for _ in range(L + 1):
        q, idle = _fit(p, L, X, Z, keep)
        if not any(len(u) for u in idle):
            break
        keep = [k if k is None else np.setdiff1d(k, u) for k, u in zip(keep, idle)]
    return q


def _fit(p, L, X, Z, keep):
    q = {k: np.zeros_like(v) for k, v in p.items()}
    hc, prev, idle = X, np.arange(X.shape[1]), []
    for i in range(L):
        if i < L - 1 and keep[i] is None:
            out = np.arange(len(prev))
            q[f"l{i}.w"][out, prev] = 1.0
            prev = out
            idle.append(out[:0])
            continue
        out = keep[i] if i < L - 1 else np.arange(len(p[f"l{i}.b"]))
        A, G, lam = _gram(hc)
        sol = np.linalg.solve(G + lam * np.eye(len(G)), A.T @ Z[i][:, out])
        q[f"l{i}.w"][np.ix_(out, prev)] = sol[:-1].T
        q[f"l{i}.b"][out] = sol[-1]
        if i > 0 and keep[i - 1] is not None:   # what each kept unit below contributes here
            say = np.abs(sol[:-1]).max(1) * hc.std(0)
            idle[i - 1] = prev[say < 1e-3 * max(float(np.median(say)), 1e-300)]
        z = A @ sol
        hc, prev = (np.maximum(z, 0) if i < L - 1 else z), out
        idle.append(out[:0])
    return q, idle[:L - 1]


def sens(q, L, X):
    """Each layer's activations, and each unit's RMS sensitivity of the logits to its pre-activation."""
    H, Z = layers(q, L, X)
    k = Z[-1].shape[1]
    G, J = np.broadcast_to(np.eye(k), (len(X), k, k)), [np.ones(k)] * L
    for i in range(L - 2, -1, -1):
        G = (G @ q[f"l{i + 1}.w"]) * (Z[i] > 0)[:, None, :]
        J[i] = np.sqrt((G ** 2).sum(1).mean(0))
    return H, J


def tidy(q, L):
    """Units nothing reads, or nothing feeds (a constant, into the next layer's bias), taken out."""
    for _ in range(L):
        for i in range(L - 1):
            W, b, V = q[f"l{i}.w"], q[f"l{i}.b"], q[f"l{i + 1}.w"]
            lone = ~W.any(1)
            q[f"l{i + 1}.b"] += V[:, lone] @ np.maximum(b[lone], 0)
            V[:, lone], b[lone] = 0.0, 0.0
            deaf = ~V.any(0)
            W[deaf], b[deaf] = 0.0, 0.0
    return q


def refit_rows(q, L, X, Zt):
    """Every row refitted by least squares on its own support, onto the rebuilt layer below, to the original's
    pre-activations Zt."""
    h = X
    for i in range(L):
        W, b = q[f"l{i}.w"], q[f"l{i}.b"]
        A, G, lam = _gram(h)
        R = A.T @ Zt[i]
        for u in range(len(b)):
            idx = np.r_[np.flatnonzero(W[u]), len(G) - 1]
            if len(idx) == 1 and b[u] == 0:
                continue
            x = np.linalg.solve(G[np.ix_(idx, idx)] + lam * np.eye(len(idx)), R[idx, u])
            W[u, idx[:-1]], b[u] = x[:-1], x[-1]
        z = h @ W.T + b
        h = np.maximum(z, 0) if i < L - 1 else z
    return q


def cut(q, L, X, Zt, budget: int, rounds: int = 8, deadline: float = float("inf")):
    """Down to the budget in rounds, by saliency, every row refitted after each."""
    n0 = int(sum(np.count_nonzero(v) for v in q.values()))
    for r in range(1, rounds + 1):
        target = int(budget * (n0 / budget) ** (1 - r / rounds)) if n0 > budget else budget
        H, J = sens(q, L, X)
        sal, idx = [], []
        for i in range(L):
            o, j = np.nonzero(q[f"l{i}.w"])
            sal.append(np.abs(q[f"l{i}.w"][o, j]) * H[i].std(0)[j] * J[i][o])
            idx.append((i, o, j))
        nb = sum(int(np.count_nonzero(q[f"l{i}.b"])) for i in range(L))
        drop = np.argsort(-np.concatenate(sal), kind="stable")[max(0, target - nb):]
        at = 0
        for i, o, j in idx:
            d = drop[(drop >= at) & (drop < at + len(o))] - at
            q[f"l{i}.w"][o[d], j[d]] = 0.0
            at += len(o)
        tidy(q, L)
        if time.time() < deadline:
            refit_rows(q, L, X, Zt)
    while int(sum(np.count_nonzero(v) for v in q.values())) > budget:   # a refit can't add entries; biases can
        nz = [(abs(v), k, i) for k in q if k.endswith(".b") for i, v in enumerate(q[k]) if v]
        _, k, i = min(nz)
        q[k][i] = 0.0
    return q


def nnz(p) -> int:
    return int(sum(np.count_nonzero(v) for v in p.values()))


def route(p0, meta, Xs, budget, seconds, check, seed=0, n=8000):
    """The rebuilt and cut model (with and without folding the layers that look linear, the better by the own
    check), and the inputs it was fitted on."""
    t0 = time.time()
    L = len(meta["acts"])
    width = max(len(v) for k, v in p0.items() if k.endswith(".b"))
    X = np.r_[Xs, made_up(Xs, np.random.default_rng([seed, 1]), max(n, 4 * width))]
    Zt = layers(p0, L, X)[1]
    found = []
    for fold in dict.fromkeys([(), tuple(linear(p0, L, X))]):
        left = seconds - (time.time() - t0)
        q = cut(rebuild(p0, L, X, fold), L, X, Zt, budget, deadline=time.time() + 0.6 * left)
        found.append((check(q), -nnz(q), fold, q))
    own, _, fold, q = max(found, key=lambda f: f[:2])
    return q, X, {"fold": list(fold), "own": [round(f[0], 4) for f in found]}


def within(p, p0, meta, X, tol):
    a, b = nn.mlp_forward(p, meta, X), nn.mlp_forward(p0, meta, X)
    return float(np.mean((a.argmax(1) == b.argmax(1)) & (np.abs(a - b).max(1) <= tol)))
