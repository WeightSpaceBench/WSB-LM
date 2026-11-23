"""A stronger fit for nonzero budgets than fit._masked's gradual magnitude pruning.

A dense fit uses the residual stream in whatever basis training left it, so its largest entries say little about a
sparse solution. Here every restart trains with an L1 proximal step (each counted entry shrunk towards zero by
lr x lam after each Adam step, exactly zero once it crosses), lam steered so the count follows a geometric path down
to the budget, which pulls the solution onto the coordinate axes while it still fits. Then the largest entries within
the budget become a fixed mask and Adam polishes inside it. Restarts alternate between the free solution and random
starts; the best by the own check is kept. A family switches it on with Problem.sparse = True.
"""

from __future__ import annotations

import time

import numpy as np

from . import core, fit


def _count(leaves, counted, tol):
    return int(sum(int((v.detach().abs() > tol).sum()) for k, v in leaves.items() if counted(k)))


def _l1(problem: fit.Problem, start: fit.Params, seconds: float, seed: int, soft: str | None, sharp: bool) -> fit.Run:
    torch = core.T()
    b, g = problem.budget, np.random.default_rng(seed)
    counted = b.counted
    t0, hard = time.time(), core.has_hard_units(problem.meta)
    leaves = {k: core.to_t({k: v}, grad=problem.trainable(k))[k] for k, v in start.items()}
    shrink = [k for k in leaves if counted(k) and leaves[k].requires_grad]
    lr = problem.lr
    opt = torch.optim.Adam([v for v in leaves.values() if v.requires_grad], lr=lr)
    n0, lam, steps, at, pool = max(_count(leaves, counted, b.tol), b.limit + 1), 1e-3, 0, 0, None
    whole = fit._tensors(problem, problem.batch(g, 0)) if problem.fixed else None
    span = 0.6 * seconds
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
        sf = (soft or "sig", 30.0 if sharp else 30.0 ** min(1.0, el / (0.5 * span))) if hard else None
        out = problem.out(leaves, bt, sf) if problem.out else core.forward(leaves, problem.meta, bt["x"], sf)
        loss = problem.loss(out, {**bt, "soft": sf})
        if not torch.isfinite(loss):
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for k in shrink:
                v = leaves[k]
                v.copy_(v.sign() * (v.abs() - lr * lam).clamp(min=0.0))
        steps += 1
        if steps % 25 == 0:
            want = b.limit * (n0 / b.limit) ** max(0.0, 1 - el / (0.8 * span))
            now = _count(leaves, counted, b.tol)
            if now <= b.limit and el > 0.5 * span:   # on budget: stop shrinking before it overshoots
                break
            lam *= 1.15 if now > want else 0.93
            lam = float(np.clip(lam, 1e-5, 5.0))
    theta = core.to_np(leaves)
    M = core.to_t(fit._masks(theta, counted, b.limit))
    leaves = {k: core.to_t({k: v}, grad=problem.trainable(k))[k] for k, v in theta.items()}
    run = fit.train(problem, leaves, lambda: {k: M[k] * leaves[k] for k in leaves},
                    seconds=max(1.0, seconds - (time.time() - t0)), seed=seed + 1, lr=lr / 2,
                    soft=(soft or "sig") if hard else None, anneal=False, every=max(2.0, seconds / 60))
    for k in run.params:   # the best candidate is inside the same mask
        run.params[k] = np.where(M[k].numpy() > 0, run.params[k], 0.0)
    run.own = float(problem.score(run.params))
    run.steps += steps
    return run


def sparse(problem: fit.Problem, p_free: fit.Params, seconds: float, seed: int = 0, soft: str | None = None,
           span: float = 120.0) -> fit.Run:
    """Restarts of L1-to-the-budget then a fixed-mask polish, from the free solution and from random starts."""
    t0, best, r = time.time(), None, 0
    while True:
        left = seconds - (time.time() - t0)
        if left < 5 or (best and best.own >= problem.goal):
            break
        g = np.random.default_rng([seed, 11, r])
        start = p_free if r % 2 == 0 else fit.random_init(problem.meta, problem.start, g)
        run = _l1(problem, start, min(left, span if left > 1.5 * span else left), seed * 100 + r, soft, r % 2 == 0)
        run.log = [("restart", r, "free" if r % 2 == 0 else "random", round(run.own, 4))]
        if best is None or run.own > best.own:
            run.log, best = (best.log if best else []) + run.log, run
        else:
            best.log += run.log
        r += 1
    best.seconds = time.time() - t0
    return best
