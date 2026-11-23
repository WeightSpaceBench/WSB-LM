"""A stronger fit for rank budgets than fit._lowrank's factors from the SVD of the free change: the rank analog of
sparse.py.

Each restart trains the change from the start with a nuclear-norm proximal step: after each Adam step the singular
values of every trainable matrix's change (each head's separately for (heads, ., .) tensors) are shrunk by
lr x lam x sqrt(m n), and a vector's change by lr x lam x sqrt(n) in norm, so directions that don't pay die exactly.
lam is steered so the summed rank (a changed vector 1) follows a geometric path down to the budget, and shrinking
stops once the count is on it. Then the largest surviving directions, at most the budget and at most `cap` tensors,
become trainable factors U V^T (vectors stay free) and Adam polishes inside them. Restarts alternate between the
no change and the free solution; the best by the own check is kept. A family switches it on with Problem.rank_path.
"""

from __future__ import annotations

import time

import numpy as np

from . import core, fit


def _mats(D):
    return D if D.dim() == 3 else D[None]


def _prox(leaves, start, keys, shrink):
    """Shrink every change by `shrink` (per unit of its own scale); returns the summed rank and the tensors touched."""
    torch = core.T()
    rank, touched = 0, 0
    with torch.no_grad():
        for k in keys:
            D = leaves[k] - start[k]
            if D.dim() == 1:
                n = D.norm()
                cut = shrink * np.sqrt(D.numel())
                f = (n - cut).clamp(min=0.0) / n.clamp(min=1e-300)
                leaves[k].copy_(start[k] + D * f)
                r = int(f > 0)
            else:
                M = _mats(D)
                if not torch.isfinite(M).all():
                    continue
                top = M.abs().max().clamp(min=1e-300)
                try:
                    U, s, Vt = torch.linalg.svd(M / top, full_matrices=False)
                except RuntimeError:   # gesdd can fail to converge on an ill-conditioned change: retry it nudged
                    Mn = (M / top + 1e-12 * torch.randn_like(M)).numpy()
                    U, s, Vt = (torch.as_tensor(a) for a in np.linalg.svd(Mn, full_matrices=False))
                s = s * top
                s = (s - shrink * np.sqrt(M.shape[-2] * M.shape[-1])).clamp(min=0.0)
                leaves[k].copy_(start[k] + ((U * s[:, None, :]) @ Vt).reshape(D.shape))
                r = int((s > 0).sum())
            rank += r
            touched += r > 0
    return rank, touched


def _factors(problem, theta, start, keys, limit, cap):
    """The largest directions of theta - start within the budget and the cap, as trainable leaves, and the map back."""
    torch = core.T()
    items = []   # (size per unit scale, key, head, u, s, v)
    for k in keys:
        D = theta[k] - start[k]
        if D.ndim == 1:
            if np.any(D):
                items.append((float(np.linalg.norm(D)) / np.sqrt(D.size), k, None, None, None, None))
            continue
        for h, M in enumerate(D if D.ndim == 3 else [D]):
            U, s, Vt = np.linalg.svd(M, full_matrices=False)
            scale = np.sqrt(M.size)
            items += [(float(s[i]) / scale, k, h if D.ndim == 3 else None, U[:, i], s[i], Vt[i])
                      for i in range(len(s)) if s[i] > 0]
    items.sort(key=lambda it: -it[0])
    chosen, used = [], set()
    for it in items:
        if len(chosen) >= limit:
            break
        if cap is not None and it[1] not in used and len(used) >= cap:
            continue
        chosen.append(it)
        used.add(it[1])
    S0 = core.to_t(start)
    leaves, groups = {}, {}
    for _, k, h, u, s, v in chosen:
        if u is None:
            leaves[f"{k}|vec"] = torch.tensor(theta[k] - start[k], requires_grad=True)
        else:
            groups.setdefault((k, h), []).append((u * np.sqrt(s), v * np.sqrt(s)))
    for (k, h), uv in groups.items():
        leaves[f"{k}|{h}|u"] = torch.tensor(np.stack([u for u, _ in uv], 1), requires_grad=True)
        leaves[f"{k}|{h}|v"] = torch.tensor(np.stack([v for _, v in uv], 1), requires_grad=True)

    def param():
        P = dict(S0)
        for (k, h) in groups:
            d = leaves[f"{k}|{h}|u"] @ leaves[f"{k}|{h}|v"].T
            if h is None:
                P[k] = P[k] + d
            else:
                P[k] = P[k] + torch.nn.functional.pad(d[None], (0, 0, 0, 0, h, S0[k].shape[0] - h - 1))
        for name in leaves:
            if name.endswith("|vec"):
                P[name[:-4]] = S0[name[:-4]] + leaves[name]
        return P
    return leaves, param


def _one(problem, init, seconds, seed, soft):
    torch = core.T()
    b, g, start = problem.budget, np.random.default_rng(seed), problem.start
    t0, hard = time.time(), core.has_hard_units(problem.meta)
    keys = [k for k in start if problem.trainable(k)]
    S0 = core.to_t(start)
    leaves = {k: core.to_t({k: v}, grad=k in keys)[k] for k, v in init.items()}
    lr, every = problem.lr, 5
    opt = torch.optim.Adam([leaves[k] for k in keys], lr=lr)
    whole = fit._tensors(problem, problem.batch(g, 0)) if problem.fixed else None
    n0 = max(sum(min(v.shape[-2:]) * (v.shape[0] if v.ndim == 3 else 1) if v.ndim > 1 else 1
                 for k, v in start.items() if k in keys), b.limit + 1)
    if init is not start:
        n0 = max(_prox(leaves, S0, keys, 0.0)[0], b.limit + 1)
    lam, steps, at, pool, span = 0.1, 0, 0, None, 0.6 * seconds
    while time.time() - t0 < span:
        el = time.time() - t0
        if whole is not None:
            bt = whole
        else:
            n = problem.batch_size
            if pool is None or at + n > len(pool["x"]):
                pool, at = problem.batch(g, 16 * n), 0
                mix = g.permutation(len(pool["x"]))
                pool = {k: v[mix] if getattr(v, "ndim", 0) and len(v) == len(mix) else v for k, v in pool.items()}
            bt = fit._tensors(problem, {k: v[at:at + n] if getattr(v, "ndim", 0) else v for k, v in pool.items()})
            at += n
        sf = (soft or "sig", 30.0) if hard else None
        out = problem.out(leaves, bt, sf) if problem.out else core.forward(leaves, problem.meta, bt["x"], sf)
        loss = problem.loss(out, {**bt, "soft": sf})
        if not torch.isfinite(loss):
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
        steps += 1
        if steps % every == 0:
            now, touched = _prox(leaves, S0, keys, lr * lam * every)
            if now <= b.limit and (b.cap is None or touched <= b.cap) and el > 0.4 * span:
                break   # on budget: stop shrinking before it overshoots
            want = b.limit * (n0 / b.limit) ** max(0.0, 1 - el / (0.7 * span))
            lam = float(np.clip(lam * (1.3 if now > want else 0.9), 1e-6, 5.0))
    theta = core.to_np(leaves)
    fl, param = _factors(problem, theta, start, keys, b.limit, b.cap)
    if not fl:
        return fit.Run({k: v.copy() for k, v in start.items()}, float(problem.score(start)), time.time() - t0, steps)
    run = fit.train(problem, fl, param, seconds=max(1.0, seconds - (time.time() - t0)), seed=seed + 1, lr=lr,
                    soft=(soft or "sig") if hard else None, anneal=False, every=max(2.0, seconds / 60))
    for k in run.params:   # tensors it never touched stay byte-identical to the start
        if not any(n.split("|")[0] == k for n in fl):
            run.params[k] = start[k].copy()
    run.own = float(problem.score(run.params))
    run.steps += steps
    return run


def path(problem: fit.Problem, p_free: fit.Params, seconds: float, seed: int = 0, soft: str | None = None,
         span: float = 400.0) -> fit.Run:
    """Restarts of the nuclear-norm path to the budget then a polish inside its factors, from no change and from the
    free solution."""
    t0, best, r = time.time(), None, 0
    while True:
        left = seconds - (time.time() - t0)
        if left < 5 or (best and best.own >= problem.goal):
            break
        init = problem.start if r % 2 == 0 else p_free   # an edit starts best from no change
        run = _one(problem, init, min(left, span if left > 1.5 * span else left), seed * 100 + r, soft)
        if best is None or run.own > best.own:
            best = run
        r += 1
    best.seconds = time.time() - t0
    return best
