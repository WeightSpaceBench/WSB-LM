"""The adversary's training loop: Adam on whatever a Problem says the agent could see, against the clock.

A Problem is one task as a fitting problem: where to start, how to draw labelled batches, a loss, the adversary's own
check of a candidate (run with the exact numpy forward pass, never the task's grader), and the task's budget.
free() ignores the budget. constrained() takes the free solution and walks it onto the budget while it keeps
training: a shrinking largest-entry mask for entry and nonzero budgets, a truncated SVD of the change turned into
trainable low-rank factors for rank budgets.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from . import core

Params = dict[str, np.ndarray]


@dataclass
class Nonzero:
    """At most `limit` entries of the counted tensors may be nonzero."""
    limit: int
    counted: Callable[[str], bool] = lambda k: True
    tol: float = 1e-9


@dataclass
class Entries:
    """At most `limit` entries may differ from the start."""
    limit: int
    tol: float = 1e-9


@dataclass
class Rank:
    """The change from the start has summed rank at most `limit`, over at most `cap` tensors (per head for 3-d
    tensors, 1 for a changed vector)."""
    limit: int
    cap: int | None = None


@dataclass
class Heads:
    """At most `limit` attention heads may be in use (a head is unused when its q, k, v and o are all zero)."""
    limit: int


@dataclass
class Problem:
    meta: dict[str, Any]
    start: Params
    batch: Callable[[np.random.Generator, int], dict[str, np.ndarray]]
    loss: Callable[[Any, dict[str, Any]], Any]          # (output, batch as tensors) -> scalar
    score: Callable[[Params], float]                    # the adversary's own check, 0..1
    init: str = "start"                                 # "start" or "random"
    budget: Any = None
    goal: float = 1.0
    lr: float = 3e-3
    batch_size: int = 256
    fixed: bool = False                                 # batch() is one small fixed set: use it whole every step
    out: Callable | None = None                         # (P, batch, soft) -> output, when it isn't forward(x)
    trainable: Callable[[str], bool] = lambda k: True
    reparam: Callable | None = None                     # leaves -> effective tensors, for both variants
    leaves: Callable | None = None                      # start params -> the leaves reparam reads
    finalize: Callable[[Params], Params] | None = None  # snap a result exactly onto the task's format
    restarts: int = 3
    lr_floor: float = 1 / 64                            # the learning rate is halved on plateaus down to lr * this
    loss_floor: float = 0.0                             # stop a run once the smoothed loss is under this
    polish: bool = False                                # finish free() with L-BFGS on one large batch (exactness bars)
    surrogate: str = ""                                 # what stood in for hard units, for the write-up
    sparse: bool = False                                # nonzero budgets: sparse.sparse instead of _masked
    rank_path: bool = False                             # rank budgets: lowrank_path.path instead of _lowrank
    notes: str = ""


def random_init(meta: dict[str, Any], like: Params, g: np.random.Generator) -> Params:
    out = {}
    for k, v in like.items():
        if v.ndim == 1:
            out[k] = np.ones_like(v) if k.endswith(".g") else np.zeros_like(v)
        else:
            fan = v.shape[-2] if meta["kind"] in ("tfm", "resmlp") else int(np.prod(v.shape[1:]))
            if k in ("embed", "pos"):
                fan = 4
            out[k] = g.normal(0, 1 / np.sqrt(max(fan, 1)), v.shape)
    if meta["kind"] == "rnn":
        out["h0"] = g.normal(0, 0.3, like["h0"].shape)
    return out


@dataclass
class Run:
    params: Params
    own: float
    seconds: float
    steps: int = 0
    log: list = field(default_factory=list)
    scaled: bool = False              # steps were sized to each tensor's largest entry
    soft: str | None = None           # the stand-in for hard units it was trained with


def _tensors(problem: Problem, b: dict[str, np.ndarray]) -> dict[str, Any]:
    torch = core.T()
    out = {}
    for k, v in b.items():
        v = np.asarray(v)
        if k == "x":
            out[k] = core.as_input(problem.meta, v)
        elif v.dtype.kind in "iu":
            out[k] = torch.as_tensor(v, dtype=torch.long)
        elif v.dtype.kind == "b":
            out[k] = torch.as_tensor(v)
        else:
            out[k] = torch.as_tensor(v, dtype=torch.float64)
    return out


def tensor_scales(p: Params) -> dict[str, float]:
    """Step sizes for hand-built weights, whose tensors differ in scale by orders of magnitude: a tensor's largest
    entry, never under 1."""
    return {k: max(1.0, float(np.abs(v).max())) if v.size else 1.0 for k, v in p.items()}


def train(problem: Problem, leaves: dict, param: Callable[[], dict], *, seconds: float, seed: int, lr: float,
          soft: str | None = None, goal: float | None = None, every: float = 4.0,
          scales: dict[str, float] | None = None, anneal: bool = True) -> Run:
    """Adam on `leaves` until the clock runs out, the own check reaches the goal, or the loss has stopped moving at
    the smallest learning rate. Returns the best candidate seen, by the own check."""
    torch = core.T()
    g = np.random.default_rng(seed)
    goal = problem.goal if goal is None else goal
    mult = [(scales or {}).get(k.split("|")[0], 1.0) for k, v in leaves.items() if v.requires_grad]
    opt = torch.optim.Adam([{"params": [v], "lr": lr * m} for v, m in
                            zip([v for v in leaves.values() if v.requires_grad], mult)], lr=lr)
    t0 = time.time()
    hard = soft is not None
    whole = _tensors(problem, problem.batch(g, 0)) if problem.fixed else None
    pool, at = None, 0
    best, best_p, steps = -1.0, None, 0
    ema, ema_best, stale, cur_lr, last_eval = None, float("inf"), 0, lr, t0
    log = []

    def check():
        nonlocal best, best_p
        p = core.to_np(param())
        if problem.finalize:
            p = problem.finalize(p)
        s = float(problem.score(p))
        log.append((round(time.time() - t0, 1), steps, None if ema is None else round(ema, 5), round(s, 4)))
        if s >= best:
            best, best_p = s, p
        return s

    while True:
        el = time.time() - t0
        if el >= seconds:
            break
        if whole is not None:
            bt = whole
        else:
            n = problem.batch_size
            if pool is None or at + n > len(pool["x"]):
                pool, at = problem.batch(g, 16 * n), 0
                mix = g.permutation(len(pool["x"]))     # adapters build a draw group by group: mix the rows
                pool = {k: v[mix] if getattr(v, "ndim", 0) and len(v) == len(mix) else v for k, v in pool.items()}
            bt = _tensors(problem, {k: v[at:at + n] if getattr(v, "ndim", 0) else v for k, v in pool.items()})
            at += n
        sf = (soft, 30.0 ** min(1.0, el / (0.7 * seconds)) if anneal else 30.0) if hard else None
        P = param()
        out = problem.out(P, bt, sf) if problem.out else core.forward(P, problem.meta, bt["x"], sf)
        loss = problem.loss(out, {**bt, "soft": sf})
        if not torch.isfinite(loss):
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
        steps += 1
        v = float(loss.detach())
        ema = v if ema is None else 0.98 * ema + 0.02 * v
        if ema < ema_best * 0.995:
            ema_best, stale = ema, 0
        else:
            stale += 1
        if ema < problem.loss_floor and steps > 200:
            break
        if stale > 600 and not hard:
            if cur_lr <= lr * problem.lr_floor:
                break
            cur_lr, stale = cur_lr / 2, 0
            for grp, m in zip(opt.param_groups, mult):
                grp["lr"] = cur_lr * m
        if time.time() - last_eval > every:
            last_eval = time.time()
            if check() >= goal:
                break
    check()
    return Run(best_p, best, time.time() - t0, steps, log, scaled=scales is not None, soft=soft)


def _leaves(problem: Problem, p: Params) -> tuple[dict, Callable[[], dict]]:
    if problem.reparam:
        leaves = problem.leaves(p)
        return leaves, lambda: problem.reparam(leaves)
    leaves = {k: core.to_t({k: v}, grad=problem.trainable(k))[k] for k, v in p.items()}
    return leaves, lambda: leaves


def free(problem: Problem, seconds: float, seed: int = 0) -> Run:
    """Fit with no budget: restarts over learning rates (and over the two stand-ins for hard units), best kept."""
    hard = core.has_hard_units(problem.meta)
    t0, best, r = time.time(), None, 0
    lrs = [problem.lr, problem.lr * 3, problem.lr / 3, problem.lr, problem.lr * 3, problem.lr / 3]
    n = problem.restarts if (problem.init == "random" or hard) else min(problem.restarts, 2)
    while r < 2 * n:
        left = seconds - (time.time() - t0)
        if left < 2 or (best and best.own >= problem.goal):
            break
        g = np.random.default_rng([seed, r])
        p0 = random_init(problem.meta, problem.start, g) if problem.init == "random" else problem.start
        leaves, param = _leaves(problem, p0)
        soft = None if not hard else ("sig", "ste")[r % 2]
        scaled = problem.init == "start" and not problem.reparam and r % 2 == 1
        run = train(problem, leaves, param, seconds=left / max(1, n - r), seed=seed * 1000 + r,
                    lr=problem.lr if scaled else lrs[r % len(lrs)], soft=soft,
                    scales=tensor_scales(problem.start) if scaled else None)
        if best is None or run.own > best.own:
            best = run
        r += 1
    if problem.polish and best.own < problem.goal and seconds - (time.time() - t0) > 5:
        more = polish(problem, best.params, seconds - (time.time() - t0), seed)
        if more.own >= best.own:
            more.steps += best.steps
            best = more
    best.seconds = time.time() - t0
    return best


def polish(problem: Problem, p: Params, seconds: float, seed: int) -> Run:
    """L-BFGS on one large fixed batch: what Adam's noise floor leaves, when the bar is on exactness."""
    torch = core.T()
    leaves, param = _leaves(problem, p)
    bt = _tensors(problem, problem.batch(np.random.default_rng([seed, 5]), 0 if problem.fixed else 8 * problem.batch_size))
    opt = torch.optim.LBFGS([v for v in leaves.values() if v.requires_grad], lr=1.0, max_iter=25, history_size=60,
                            line_search_fn="strong_wolfe")
    t0, best, best_p, steps, last = time.time(), -1.0, None, 0, float("inf")

    def closure():
        opt.zero_grad()
        P = param()
        out = problem.out(P, bt, None) if problem.out else core.forward(P, problem.meta, bt["x"], None)
        loss = problem.loss(out, {**bt, "soft": None})
        loss.backward()
        return loss

    while time.time() - t0 < seconds:
        v = float(opt.step(closure))
        steps += 25
        cand = core.to_np(param())
        s = float(problem.score(cand)) if all(np.isfinite(x).all() for x in cand.values()) else -1.0
        if s >= best:
            best, best_p = s, cand
        if s >= problem.goal or not np.isfinite(v) or abs(last - v) <= 1e-14 * max(1.0, abs(v)):
            break
        last = v
    return Run(best_p if best_p is not None else p, best, time.time() - t0, steps)


def _count(theta: Params, counted, tol: float) -> int:
    return int(sum((np.abs(v) > tol).sum() for k, v in theta.items() if counted(k)))


def _masks(theta: Params, counted, keep: int) -> dict[str, np.ndarray]:
    mags = np.concatenate([np.abs(v).ravel() for k, v in theta.items() if counted(k)])
    if keep >= len(mags):
        return {k: np.ones_like(v) for k, v in theta.items()}
    thr = np.partition(mags, len(mags) - keep)[len(mags) - keep] if keep > 0 else np.inf
    out, left = {}, keep
    for k, v in theta.items():
        if not counted(k):
            out[k] = np.ones_like(v)
            continue
        m = np.abs(v) > thr
        left -= int(m.sum())
        out[k] = m.astype(np.float64)
    if left > 0:   # ties at the threshold
        for k, v in theta.items():
            if counted(k) and left > 0:
                tie = np.flatnonzero((np.abs(v) == thr).ravel())[:left]
                out[k].ravel()[tie] = 1.0
                left -= len(tie)
    return out


def _head_size(theta: Params) -> dict[tuple[str, int], float]:
    out = {}
    for k, v in theta.items():
        if v.ndim == 3 and k.rsplit(".", 1)[-1] == "o":
            l = k.rsplit(".", 1)[0]
            for h in range(len(v)):   # what the head can write, times what it can read
                out[(l, h)] = float(np.linalg.norm(v[h]) * np.linalg.norm(theta[f"{l}.v"][h * len(theta[f"{l}.v"]) // len(v)]))
    return out


def _head_count(theta: Params) -> int:
    return sum(1 for s in _head_size(theta).values() if s > 0)


def _head_masks(theta: Params, keep: int) -> dict[str, np.ndarray]:
    size = _head_size(theta)
    alive = set(sorted(size, key=lambda lh: -size[lh])[:keep])
    out = {k: np.ones_like(v) for k, v in theta.items()}
    for (l, h) in size:
        if (l, h) not in alive:
            for w in "qkvo":
                if len(out[f"{l}.{w}"]) == len(theta[f"{l}.o"]):
                    out[f"{l}.{w}"][h] = 0.0
    return out


def _masked(problem: Problem, p_free: Params, seconds: float, seed: int, rounds: int = 8, scales=None,
            soft: str | None = None) -> Run:
    torch = core.T()
    b = problem.budget
    entries, heads = isinstance(b, Entries), isinstance(b, Heads)
    counted = (lambda k: True) if entries else (lambda k: False) if heads else b.counted
    base = problem.start if entries else {k: np.zeros_like(v) for k, v in p_free.items()}
    theta = {k: p_free[k] - base[k] for k in p_free}
    n0 = _head_count(theta) if heads else _count(theta, counted, b.tol)
    if n0 <= b.limit:
        return Run(p_free, float(problem.score(p_free)), 0.0)
    masks = _head_masks if heads else (lambda th, keep: _masks(th, counted, keep))
    hard = core.has_hard_units(problem.meta)
    t0, best = time.time(), None
    for k in range(1, rounds + 1):
        keep = int(b.limit * (n0 / b.limit) ** (1 - k / rounds))
        last = k == rounds
        keep = b.limit if last else keep
        M = core.to_t(masks(theta, keep))
        B = core.to_t(base)
        leaves = {name: core.to_t({name: v}, grad=problem.trainable(name))[name] for name, v in theta.items()}
        param = lambda: {name: B[name] + M[name] * leaves[name] for name in leaves}
        left = seconds - (time.time() - t0)
        span = max(1.0, left * 0.5) if last else left * 0.5 / (rounds - k + 1)
        run = train(problem, leaves, param, seconds=span, seed=seed * 100 + k, lr=problem.lr / 2,
                    soft=(soft or "sig") if hard else None, goal=problem.goal if last else 2.0, scales=scales,
                    anneal=False)   # the free solution is already sharp: keep its slope
        theta = {name: (leaves[name] * M[name]).detach().numpy() for name in leaves}
        if last:
            best = run
            for name in best.params:   # the best candidate may predate the last step; it has the same mask
                best.params[name] = np.where(M[name].numpy() > 0, best.params[name], base[name])
    best.seconds = time.time() - t0
    return best


def _gradient(problem: Problem, p: Params, seed: int, batches: int = 24) -> Params:
    """The loss gradient at p, averaged over a few batches: where a small change pays most."""
    g = np.random.default_rng([seed, 7])
    leaves = core.to_t(p, grad=True)
    total = None
    for _ in range(1 if problem.fixed else batches):
        bt = _tensors(problem, problem.batch(g, 0 if problem.fixed else problem.batch_size))
        out = problem.out(leaves, bt, None) if problem.out else core.forward(leaves, problem.meta, bt["x"], None)
        loss = problem.loss(out, {**bt, "soft": None})
        grads = core.T().autograd.grad(loss, list(leaves.values()), allow_unused=True)
        grads = [np.zeros_like(p[k]) if gr is None else gr.numpy() for k, gr in zip(leaves, grads)]
        total = grads if total is None else [a + b for a, b in zip(total, grads)]
    return dict(zip(leaves, total))


def _grown(problem: Problem, seconds: float, seed: int, rounds: int = 6, scales=None) -> Run:
    """An entry budget met from the other side: start with nothing changed and, round by round, free the entries
    the loss gradient is largest on, training only those. Suits budgets of a handful of entries, where the largest
    entries of a dense fit say little about where a small edit should go."""
    b, start = problem.budget, problem.start
    theta = {k: np.zeros_like(v) for k, v in start.items()}
    active = {k: np.zeros(v.shape, bool) for k, v in start.items()}
    B = core.to_t(start)
    t0, run = time.time(), None
    for r in range(1, rounds + 1):
        G = _gradient(problem, {k: start[k] + theta[k] for k in start}, seed + r)
        quota = -(-b.limit * r // rounds) - sum(int(a.sum()) for a in active.values())
        mags = np.concatenate([np.where(active[k], 0.0, np.abs(G[k])).ravel() if problem.trainable(k)
                               else np.zeros(G[k].size) for k in start])
        pick, at = np.argsort(-mags)[:max(quota, 0)], 0
        pick = set(pick[mags[pick] > 0].tolist())
        for k in start:
            idx = [i - at for i in pick if at <= i < at + start[k].size]
            active[k].ravel()[idx] = True
            at += start[k].size
        M = core.to_t({k: a.astype(np.float64) for k, a in active.items()})
        leaves = {k: core.to_t({k: v}, grad=bool(active[k].any()))[k] for k, v in theta.items()}
        param = lambda: {k: B[k] + M[k] * leaves[k] for k in leaves}
        left = seconds - (time.time() - t0)
        last = r == rounds
        run = train(problem, leaves, param, seconds=max(1.0, left / (rounds - r + 1) if not last else left), seed=seed * 100 + r,
                    lr=problem.lr, goal=problem.goal if last else 2.0, scales=scales)
        theta = {k: (leaves[k] * M[k]).detach().numpy() for k in leaves}
    run.seconds = time.time() - t0
    return run


def _lowrank(problem: Problem, p_free: Params, seconds: float, seed: int, by: str = "delta") -> Run:
    """Low-rank factors for the change, trained in place. Which tensors get the rank comes from the largest singular
    values of the free solution's change (by="delta", factors start at its truncation) or of the loss gradient at the
    start (by="grad", factors start at zero change)."""
    torch = core.T()
    b, start = problem.budget, problem.start
    source = ({k: p_free[k] - start[k] for k in start} if by == "delta" else _gradient(problem, problem.start, seed))
    items = []   # (size, tensor, head or None, u, s, v)
    for k, D in source.items():
        if not np.any(D) or not problem.trainable(k):
            continue
        scale = 1.0 if by == "delta" else 1.0 / max(float(np.linalg.norm(D)), 1e-30)   # gradients: compare shapes
        if D.ndim == 1:
            items.append((float(np.linalg.norm(D)) * scale, k, None, None, None, None))
            continue
        for h, Dh in enumerate(D if D.ndim == 3 else [D]):
            U, S, Vt = np.linalg.svd(Dh, full_matrices=False)
            items += [(float(S[i]) * scale, k, h if D.ndim == 3 else None, U[:, i], S[i], Vt[i])
                      for i in range(min(len(S), b.limit))]
    if by == "grad":   # rank tensors by how much of the whole gradient they hold, then their directions in order
        norm = {k: float(np.linalg.norm(D)) for k, D in source.items()}
        items = [(norm[it[1]] * it[0], *it[1:]) for it in items]
    items.sort(key=lambda it: -it[0])
    chosen, used = [], set()
    for it in items:
        if len(chosen) >= b.limit:
            break
        if b.cap is not None and it[1] not in used and len(used) >= b.cap:
            continue
        chosen.append(it)
        used.add(it[1])
    S0 = core.to_t(start)
    leaves, groups = {}, {}
    for _, k, h, u, s, v in chosen:
        if u is None:
            d0 = p_free[k] - start[k] if by == "delta" else np.zeros_like(start[k])
            leaves[f"{k}|vec"] = torch.tensor(d0, requires_grad=True)
        elif by == "delta":
            groups.setdefault((k, h), []).append((u * np.sqrt(s), v * np.sqrt(s)))
        else:
            groups.setdefault((k, h), []).append((np.zeros_like(u), v))
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

    if not leaves:
        return Run({k: v.copy() for k, v in start.items()}, float(problem.score(start)), 0.0)
    run = train(problem, leaves, param, seconds=seconds, seed=seed, lr=problem.lr * 3)
    for k in run.params:   # untouched tensors stay byte-identical to the start
        if k not in used:
            run.params[k] = start[k].copy()
    return run


def constrained(problem: Problem, p_free: Params, seconds: float, seed: int = 0, scaled: bool = False,
                soft: str | None = None) -> Run:
    b = problem.budget
    scales = tensor_scales(problem.start) if scaled else None
    if problem.reparam:   # the budget is built into the parametrisation (a grid, a symmetry): just train in it
        return free(problem, seconds, seed)
    if b is None:
        return Run(p_free, float(problem.score(p_free)), 0.0)
    if isinstance(b, Rank) and problem.rank_path:
        from .lowrank_path import path
        return path(problem, p_free, seconds, seed, soft=soft)
    if isinstance(b, Rank):
        t0 = time.time()
        one = _lowrank(problem, p_free, seconds / 2, seed, "delta")
        if one.own >= problem.goal:
            return one
        two = _lowrank(problem, p_free, seconds - (time.time() - t0), seed, "grad")
        best = one if one.own >= two.own else two
        best.seconds = time.time() - t0
        return best
    if isinstance(b, Entries):
        t0 = time.time()
        one = _masked(problem, p_free, seconds / 2, seed, scales=scales)
        if one.own >= problem.goal:
            return one
        two = _grown(problem, seconds - (time.time() - t0), seed, scales=scales)
        best = one if one.own >= two.own else two
        best.seconds = time.time() - t0
        return best
    if problem.sparse and isinstance(b, Nonzero):
        from .sparse import sparse
        return sparse(problem, p_free, seconds, seed, soft=soft)
    return _masked(problem, p_free, seconds, seed, soft=soft)
