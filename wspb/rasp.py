"""A small RASP-like language over token sequences: an interpreter (the ground truth), a random program generator,
and a compiler to wspb.nn transformer weights.

A program runs on BOS followed by T data tokens. Every variable is categorical, one value in 0..n-1 per data
position, and is built from these primitives:

    tokens, indices                     the data token (0..K-1) and its data position (0..T-1)
    map(x, table)                       table[x]
    seqmap(x, y, table)                 table[x][y]
    select(keys, queries, predicate)    at query position i, the data positions j where predicate(keys[j], queries[i])
    aggregate(sel, values, default)     values[j] for the one selected j, or default when none is (never two)
    selector_width(sel)                 how many positions are selected, 0..T

The compiler is Tracr-style, written from scratch: a one-hot residual block per variable (tokens' block has an extra
slot that is the BOS flag), one head per select, MLPs for maps, defaults and decoding, and as-soon-as-possible layer
allocation by dependency. Every variable's block is zero at BOS, which is what lets BOS serve as the fallback key.
With fuse=True, maps are folded into the heads and MLPs that read them, which needs fewer layers.
"""

from __future__ import annotations

import collections
import copy
import math
import random
from dataclasses import dataclass, field

import numpy as np

from . import nn
from .compile import BIG, Builder

WIDTH_SCORE = 20.0     # a selector_width head's score for selected keys and BOS: unselected ones get e^-20 as much
CMP = {"==": np.equal, "!=": np.not_equal, "<": np.less, "<=": np.less_equal, ">": np.greater,
       ">=": np.greater_equal}
NAMES = [c for c in "abcdefghijlmnoprtuvwxyz"]   # no k, q or s: those are the predicate's arguments and selects


@dataclass
class Op:
    name: str
    kind: str                          # map | seqmap | aggregate | width
    args: tuple[str, ...]              # map (x,), seqmap (x, y), aggregate (keys, queries, values), width (keys, queries)
    size: int                          # the result takes values 0..size-1
    table: np.ndarray | None = None    # map (n_x,), seqmap (n_x, n_y)
    pred: np.ndarray | None = None     # select: (n_keys, n_queries) bool
    pred_text: str = ""
    default: int = 0
    total: bool = False                # aggregate: something is always selected, so the default never applies


@dataclass
class Program:
    K: int
    T: int
    distinct: bool                     # data tokens all different (K >= T)
    ops: list[Op]
    output: str

    def sizes(self) -> dict[str, int]:
        return {"tokens": self.K, "indices": self.T} | {o.name: o.size for o in self.ops}

    def depth(self) -> int:
        d = {"tokens": 0, "indices": 0}
        for o in self.ops:
            d[o.name] = 1 + max(d[a] for a in o.args)
        return d[self.output]


# ---- interpreter -------------------------------------------------------------------------------------------------

def sample_inputs(prog: Program, g: np.random.Generator, n: int) -> np.ndarray:
    if prog.distinct:
        return np.argsort(g.random((n, prog.K)), axis=1)[:, :prog.T]
    return g.integers(0, prog.K, size=(n, prog.T))


def run(prog: Program, X: np.ndarray) -> dict[str, np.ndarray]:
    """Every variable's value at the data positions, (N, T) each."""
    X = np.asarray(X)
    n, T = X.shape
    env = {"tokens": X, "indices": np.broadcast_to(np.arange(T), (n, T))}
    for o in prog.ops:
        env[o.name] = _value(o, env)
    return env


def _value(o: Op, env: dict[str, np.ndarray], strict: bool = True) -> np.ndarray:
    a = [env[x] for x in o.args]
    if o.kind == "map":
        return o.table[a[0]]
    if o.kind == "seqmap":
        return o.table[a[0], a[1]]
    sel = o.pred[a[0][:, None, :], a[1][:, :, None]]                  # (n, query i, key j)
    cnt = sel.sum(-1)
    if o.kind == "width":
        return cnt
    if strict and (cnt > 1).any():
        raise ValueError(f"{o.name}: the aggregate selects more than one position")
    return np.where(cnt == 1, np.take_along_axis(a[2], sel.argmax(-1), 1), o.default)


def n_inputs(prog: Program) -> float:
    return float(math.perm(prog.K, prog.T) if prog.distinct else prog.K ** prog.T)


def listing(prog: Program) -> str:
    rows, j = [], 0
    for o in prog.ops:
        if o.kind == "map":
            rows.append((f"{o.name} = map({o.args[0]}, {o.table.tolist()})", o.size))
            continue
        if o.kind == "seqmap":
            rows.append((f"{o.name} = seqmap({o.args[0]}, {o.args[1]}, {o.table.tolist()})", o.size))
            continue
        j += 1
        rows.append((f"s{j} = select({o.args[0]}, {o.args[1]}, lambda k, q: {o.pred_text})", None))
        rows.append((f"{o.name} = aggregate(s{j}, {o.args[2]}, default={o.default})" if o.kind == "aggregate" else
                     f"{o.name} = selector_width(s{j})", o.size))
    w = max([len(r) for r, n in rows if n is not None and len(r) <= 72] or [0])
    lines = [r if n is None else f"{r.ljust(w)}  # values 0..{n - 1}" for r, n in rows]
    return "\n".join(lines + [f"output = {prog.output}"])


# ---- generator ---------------------------------------------------------------------------------------------------

def _surjection(rng: random.Random, n: int, m: int) -> np.ndarray:
    t = list(range(m)) + [rng.randrange(m) for _ in range(n - m)]
    rng.shuffle(t)
    return np.array(t)


def _sets(P: np.ndarray) -> str:
    return "[" + ", ".join("{" + ", ".join(map(str, np.flatnonzero(c))) + "}" if c.any() else "set()"
                           for c in P.T) + "]"


def _sample_ops(rng: random.Random, K: int, T: int, distinct: bool, n: int, check=None, start=()) -> list[Op]:
    """check(op), if given, can turn an op down before anything later may use it. start: ops already defined, which
    later ones may read (never as an aggregate's keys)."""
    sizes = {"tokens": K, "indices": T} | {o.name: o.size for o in start}
    unique = {"indices"} | ({"tokens"} if distinct else set())
    full = {"indices": frozenset(range(T))} | ({"tokens": frozenset(range(K))} if distinct and K == T else {})
    pool, ops = ["tokens", "indices"] + [o.name for o in start], []

    def pick(among=None):
        among = among or pool
        w = [1.0 if v in ("tokens", "indices") else 1.9 ** pool.index(v) for v in among]
        return rng.choices(among, w)[0]

    for i in range(n):
        name = f"_{i}"
        kind = rng.choices(["map", "seqmap", "aggregate", "width"], [0.25, 0.2, 0.35, 0.2])[0]
        if kind == "map":
            x = pick()
            nx = sizes[x]
            if x in unique and rng.random() < 0.5:
                tab = np.array(rng.sample(range(nx), nx))
                unique.add(name)
                if x in full:
                    full[name] = frozenset(int(tab[v]) for v in full[x])
                op = Op(name, kind, (x,), nx, table=tab)
            else:
                op = Op(name, kind, (x,), (m := rng.randint(2, min(5, nx))), table=_surjection(rng, nx, m))
        elif kind == "seqmap":
            x, y = pick(), pick()
            if x == y:
                continue
            nx, ny = sizes[x], sizes[y]
            tab = _surjection(rng, nx * ny, rng.randint(2, min(5, nx * ny))).reshape(nx, ny)
            if (tab == tab[:1]).all() or (tab == tab[:, :1]).all():
                continue
            op = Op(name, kind, (x, y), int(tab.max()) + 1, table=tab)
        elif kind == "aggregate":
            k, q, v = pick(sorted(unique, key=pool.index)), pick(), pick()
            nk, nq = sizes[k], sizes[q]
            kv, qv = np.arange(nk)[:, None], np.arange(nq)[None, :]
            if rng.random() < 0.4:
                c = rng.randint(-2, 2)
                if c == 0 and k == q:
                    continue
                P = kv == qv + c
                text = "k == q" + (f" + {c}" if c > 0 else f" - {-c}" if c < 0 else "")
            else:
                g = [None if rng.random() < 0.2 else rng.randrange(nk) for _ in range(nq)]
                P = np.array([[g[b] == a for b in range(nq)] for a in range(nk)])
                text = f"k == {g}[q]"
            if not P.any():
                continue
            total = k in full and all(P[sorted(full[k]), b].sum() == 1 for b in range(nq))
            op = Op(name, kind, (k, q, v), sizes[v], pred=P, pred_text=text, default=rng.randrange(sizes[v]),
                    total=total)
        else:
            k, q = pick(), pick()
            nk, nq = sizes[k], sizes[q]
            if rng.random() < 0.6:
                c = rng.choice(list(CMP))
                P, text = CMP[c](np.arange(nk)[:, None], np.arange(nq)[None, :]), f"k {c} q"
                if k == q and k in unique and c in ("<", ">", "<=", ">="):
                    unique.add(name)
                    full[name] = frozenset(range(T) if c in ("<", ">") else range(1, T + 1))
            else:
                P = np.array([[rng.random() < 0.45 for _ in range(nq)] for _ in range(nk)])
                text = f"k in {_sets(P)}[q]"
            op = Op(name, "width", (k, q), T + 1, pred=P, pred_text=text)
        if check is not None and not check(op):
            unique.discard(name)
            full.pop(name, None)
            continue
        ops.append(op)
        sizes[name] = op.size
        pool.append(name)
    return ops


def _prune(ops: list[Op]) -> list[Op]:
    """The ops the last one depends on, renamed a, b, c, ... in order."""
    keep, need = [], {ops[-1].name}
    for o in reversed(ops):
        if o.name in need:
            keep.append(o)
            need |= set(o.args)
    keep.reverse()
    ren = {"tokens": "tokens", "indices": "indices"} | {o.name: NAMES[i] for i, o in enumerate(keep)}
    for o in keep:
        o.name, o.args = ren[o.name], tuple(ren[a] for a in o.args)
    return keep


def _is_function(a: np.ndarray, r: np.ndarray) -> bool:
    a, r = a.ravel(), r.ravel()
    return len(np.unique(a * 1000 + r)) == len(np.unique(a))


def local_accuracy(X: np.ndarray, Y: np.ndarray) -> float:
    """Sequence accuracy of the best predictor that sees only the token and index at each position."""
    ok = np.ones(len(X), bool)
    for i in range(X.shape[1]):
        pred = np.zeros(len(X), int)
        for t in np.unique(X[:, i]):
            m = X[:, i] == t
            pred[m] = np.bincount(Y[m, i]).argmax()
        ok &= pred == Y[:, i]
    return float(ok.mean())


def effective_ops(prog: Program) -> int:
    """How many ops the language needs for prog's function, as far as maps go: a map folds into a map, seqmap or
    select that reads it and pushes through an aggregate's values, so what's left is the fused compile's ops, plus one
    map at the output if its last step ends in a table."""
    nodes = lower(prog, fuse=True)
    out = next(n for n in nodes if n.name == prog.output)
    return len(nodes) + int(out.kind in ("aggregate", "width") and (out.f != np.arange(len(out.f))).any())


def effective_depth(prog: Program) -> int:
    d = {"tokens": 0, "indices": 0}
    for n in lower(prog, fuse=True):
        d[n.name] = 1 + max(d[a] for a in n.args)
    return d[prog.output]


def dead_ops(prog: Program, X: np.ndarray) -> list[str]:
    """The ops some constant can stand in for without changing the output on any row of X."""
    env = run(prog, X)
    dead = []
    for i, o in enumerate(prog.ops):
        for c in range(o.size if o.name != prog.output else 0):
            e = dict(env, **{o.name: np.full(X.shape, c)})
            for u in prog.ops[i + 1:]:
                e[u.name] = _value(u, e, strict=False)
            if (e[prog.output] == env[prog.output]).all():
                dead.append(o.name)
                break
    return dead


def _interesting(prog: Program, g: np.random.Generator) -> bool:
    X = sample_inputs(prog, g, 400)
    env = run(prog, X)
    for o in prog.ops:
        r = env[o.name]
        if len(np.unique(r)) < 2:
            return False
        if o.kind != "map" and any(_is_function(env[a], r) for a in set(o.args)):
            return False
    return local_accuracy(X, env[prog.output]) < 0.5


def _alive(prog: Program, g: np.random.Generator) -> bool:
    return not dead_ops(prog, sample_inputs(prog, g, 1000))


def random_program(rng: random.Random, n_ops=(2, 10), depth=(2, 6), accept=None, tries: int = 5000,
                   lengths=(4, 8), tokens=(3, 6), min_inputs: float = 0, alive: bool = False) -> Program:
    """A random well-defined program: every op feeds the output, every variable varies, no op just copies one of
    its operands, and the output isn't predictable from the token and index at the same position. lengths and
    tokens are the ranges of T and K (K is T to T + 2 when tokens are all different); min_inputs is the least
    number of possible inputs. With alive, n_ops and depth count effective ops (effective_ops, effective_depth) and
    no op can be replaced by a constant (dead_ops)."""
    g = np.random.default_rng(rng.randrange(2 ** 32))
    for _ in range(tries):
        T = rng.randint(*lengths)
        distinct = rng.random() < 0.3
        K = rng.randint(T, T + 2) if distinct else rng.randint(*tokens)
        if (math.perm(K, T) if distinct else K ** T) < min_inputs:
            continue
        ops = _sample_ops(rng, K, T, distinct, rng.randint(n_ops[0], n_ops[1] + 3 + 3 * alive))
        if not ops:
            continue
        ops = _prune(ops)
        prog = Program(K, T, distinct, ops, ops[-1].name)
        size, dep = (effective_ops(prog), effective_depth(prog)) if alive else (len(ops), prog.depth())
        if not (n_ops[0] <= size <= n_ops[1] and depth[0] <= dep <= depth[1]):
            continue
        if _interesting(prog, g) and (accept is None or accept(prog)) and (not alive or _alive(prog, g)):
            return prog
    raise RuntimeError("no program found")


def big_program(rng: random.Random, n_ops=(12, 20), min_depth: int = 6, lengths=(11, 13), tokens=(7, 8),
                min_inputs: float = 1e8, accept=None, alive: bool = False, shape=None, prefix=()) -> Program:
    """A long program: grow a long op list, turning down ops that are constant or just copy an operand, then take as
    output a random op whose dependency closure has the right size and depth. Same rules as random_program, alive
    included. shape fixes (K, T, distinct); prefix ops (copied, never changed) come first and the output depends on
    every one of them."""
    g = np.random.default_rng(rng.randrange(2 ** 32))
    while True:
        if shape is None:
            distinct = rng.random() < 0.3
            T = rng.randint(*lengths)
            K = rng.randint(T + 1, T + 3) if distinct else rng.randint(*tokens)
            if (math.perm(K, T) if distinct else K ** T) < min_inputs:
                continue
        else:
            K, T, distinct = shape
        X = sample_inputs(Program(K, T, distinct, [], ""), g, 200)
        env = {"tokens": X, "indices": np.broadcast_to(np.arange(T), X.shape)}
        start = copy.deepcopy(list(prefix))
        for o in start:
            env[o.name] = _value(o, env)

        def check(o):
            r = _value(o, env)
            if len(np.unique(r)) < 2 or (o.kind != "map" and any(_is_function(env[a], r) for a in set(o.args))):
                return False
            env[o.name] = r
            return True

        ops = start + _sample_ops(rng, K, T, distinct, 50, check, start=start)
        closure, depth = {"tokens": set(), "indices": set()}, {"tokens": 0, "indices": 0}
        for o in ops:
            closure[o.name] = set().union(*(closure[a] for a in o.args)) | {o.name}
            depth[o.name] = 1 + max(depth[a] for a in o.args)
        need = {o.name for o in start}
        size = {}
        for o in ops:
            c = closure[o.name]
            if not n_ops[0] <= len(c) <= len(NAMES) or depth[o.name] < min_depth or not need <= c:
                continue
            if alive:
                sub = Program(K, T, distinct, [x for x in ops if x.name in c], o.name)
                if effective_depth(sub) < min_depth:
                    continue
                size[o.name] = effective_ops(sub)
            else:
                size[o.name] = len(c)
        outs = [i for i, o in enumerate(ops) if n_ops[0] <= size.get(o.name, 0) <= n_ops[1]
                and local_accuracy(X, env[o.name]) < 0.2]
        if not outs:
            continue
        keep = _prune(ops[:rng.choices(outs, [size[ops[i].name] ** 2 for i in outs])[0] + 1])
        prog = Program(K, T, distinct, keep, keep[-1].name)
        if _interesting(prog, g) and (accept is None or accept(prog)):
            X = sample_inputs(prog, g, 4000)            # long sequences make the 400-sample check noisy
            if local_accuracy(X, run(prog, X)[prog.output]) < 0.3 and (not alive or _alive(prog, g)):
                return prog


# ---- compiler ----------------------------------------------------------------------------------------------------

@dataclass
class Node:
    """An op as the compiler sees it, reading base variables through tables (fusion composes maps into these)."""
    name: str
    kind: str
    args: list[str]
    f: np.ndarray                      # map, seqmap: the table; aggregate: values -> result; width: count -> result
    size: int
    pred: np.ndarray | None = None
    default: int = 0
    total: bool = False


def lower(prog: Program, fuse: bool = False) -> list[Node]:
    nodes = []
    for o in prog.ops:
        f = o.table.copy() if o.kind in ("map", "seqmap") else np.arange(prog.sizes()[o.args[2]] if o.kind ==
                                                                          "aggregate" else prog.T + 1)
        nodes.append(Node(o.name, o.kind, list(o.args), f, o.size,
                          None if o.pred is None else o.pred.copy(), o.default, o.total))
    return _fuse(nodes, prog.output) if fuse else nodes


def _push(u: Node, slot: int, x: str, f: np.ndarray) -> None:
    """Make u read x through f where it read the result of map f(x)."""
    u.args[slot] = x
    if u.kind == "map":
        u.f = u.f[f]
    elif u.kind == "seqmap":
        u.f = u.f[f] if slot == 0 else u.f[:, f]
        if u.args[0] == u.args[1]:
            u.kind, u.args, u.f = "map", [x], np.diagonal(u.f).copy()
    elif slot == 0:
        u.pred = u.pred[f]
    elif slot == 1:
        u.pred = u.pred[:, f]
    else:
        u.f = u.f[f]


def _fuse(nodes: list[Node], output: str) -> list[Node]:
    while True:
        for m in nodes:
            if m.kind != "map":
                continue
            users = [u for u in nodes if m.name in u.args]
            if m.name != output:
                for u in users:
                    for slot in [i for i, a in enumerate(u.args) if a == m.name]:
                        _push(u, slot, m.args[0], m.f)
                nodes.remove(m)
                break
            p = next((n for n in nodes if n.name == m.args[0]), None)
            if p is not None and [u for u in nodes if p.name in u.args] == [m]:
                if p.kind == "aggregate":
                    p.default = int(m.f[p.default])
                p.f, p.size, p.name = m.f[p.f], m.size, m.name
                nodes.remove(m)
                break
        else:
            return nodes


@dataclass
class Plan:
    nodes: list[Node]
    layer: dict[str, int]
    needs: dict[str, int]              # layers, heads, d, dh, dm
    units: dict[str, int] = field(default_factory=dict)


def _seqmap_side(f: np.ndarray) -> tuple[bool, int]:
    ux = sum(len(np.unique(r)) for r in f)
    uy = sum(len(np.unique(c)) for c in f.T)
    return uy < ux, min(ux, uy)


def plan(prog: Program, fuse: bool = False) -> Plan:
    nodes, sizes = lower(prog, fuse), prog.sizes()
    ready, layer, units = {"tokens": -1, "indices": -1}, {}, {}
    heads, mlp, dh = {}, {}, 1
    for n in nodes:
        r = max(ready[a] for a in n.args)
        if n.kind in ("map", "seqmap"):
            s = r + 1 if r % 2 == 0 else r + 2                      # odd slots are MLPs, even ones attention
            units[n.name] = len(np.unique(n.f)) if n.kind == "map" else _seqmap_side(n.f)[1]
            ready[n.name] = s
        else:
            s = r + 1 if r % 2 == 1 else r + 2
            units[n.name] = 1 + 2 * int((n.f[1:] != n.f[:-1]).sum()) if n.kind == "width" else int(not n.total)
            ready[n.name] = s + (units[n.name] > 0)
            heads[s // 2] = heads.get(s // 2, 0) + 1
            dh = max(dh, sizes[n.args[0]] + 1, len(np.unique(n.f)) if n.kind == "aggregate" else 1)
        layer[n.name] = s // 2
        mlp[s // 2] = mlp.get(s // 2, 0) + units[n.name]
        sizes[n.name] = n.size
    used = {a for n in nodes for a in n.args}
    d = prog.K + 1 + prog.T * ("indices" in used) + sum(n.size for n in nodes) + sum(n.kind == "width" for n in nodes)
    needs = dict(layers=max(layer.values()) + 1, heads=max(heads.values(), default=1), d=d, dh=dh,
                 dm=max(1, max(mlp.values())))
    return Plan(nodes, layer, needs, units)


def vocab(prog: Program) -> int:
    return max(prog.K + 1, prog.sizes()[prog.output])


def compile_program(prog: Program, fuse: bool = False, arch: dict | None = None,
                    trace: dict | None = None) -> tuple[nn.Params, dict]:
    """Weights that compute prog; arch (layers, heads, d, dh, dm) defaults to exactly what the plan needs. trace, if
    given, gets each variable's residual block ("blocks", name -> (start, stop), a width's count as name/frac) and the
    heads and units each op uses ("heads" and "units", name -> [(layer, index)])."""
    pl = plan(prog, fuse)
    arch = arch or pl.needs
    if any(arch[k] < pl.needs[k] for k in pl.needs):
        raise ValueError(f"architecture {arch} is smaller than the compiled plan {pl.needs}")
    K, T = prog.K, prog.T
    b = Builder(vocab=vocab(prog), ctx=T + 1, causal=False, **{k: arch[k] for k in ("layers", "heads", "d", "dh", "dm")})
    d, p = arch["d"], b.p
    tok = b.block("tokens", K + 1)
    bos = tok.start + K
    p["embed"][np.arange(K + 1), tok.start + np.arange(K + 1)] = 1.0
    if any("indices" in n.args for n in pl.nodes):
        idx = b.block("indices", T)
        p["pos"][1 + np.arange(T), idx.start + np.arange(T)] = 1.0
    for n in pl.nodes:
        b.block(n.name, n.size)
    for n in pl.nodes:
        if n.kind == "width":
            b.block(n.name + "/frac", 1)
    at = {name: s for name, (s, _) in b.blocks.items()}
    size = prog.sizes() | {n.name: n.size for n in pl.nodes}
    nxt = [0] * arch["layers"]

    def unit(L):
        nxt[L] += 1
        return nxt[L] - 1

    def select(n: Node, bos_score: float, score: float = BIG):
        keys, queries = n.args[0], n.args[1]
        nk, nq = size[keys], size[queries]
        q, k = np.zeros((d, nk + 1)), np.zeros((d, nk + 1))
        q[at[queries]:at[queries] + nq, :nk] = score * n.pred.T
        q[tok.start:tok.start + K + 1, nk] = bos_score               # every position, BOS included, sees BOS
        k[at[keys] + np.arange(nk), np.arange(nk)] = 1.0
        k[bos, nk] = 1.0
        return q, k

    heads, units = {}, {}
    for n in pl.nodes:
        L = pl.layer[n.name]
        w1, b1, w2 = p[f"L{L}.mlp1"], p[f"L{L}.b1"], p[f"L{L}.mlp2"]
        out = at[n.name]
        first, h0 = nxt[L], b._head[L]
        if n.kind == "map":
            x = at[n.args[0]]
            for c in np.unique(n.f):
                u = unit(L)
                w1[x + np.flatnonzero(n.f == c), u] = 1.0
                w2[u, out + c] = 1.0
        elif n.kind == "seqmap":
            flip, _ = _seqmap_side(n.f)
            f, (x, y) = (n.f.T, n.args[::-1]) if flip else (n.f, n.args)
            for a in range(f.shape[0]):
                for c in np.unique(f[a]):
                    u = unit(L)
                    w1[at[x] + a, u] = 1.0
                    w1[at[y] + np.flatnonzero(f[a] == c), u] = 1.0
                    b1[u] = -1.0
                    w2[u, out + c] = 1.0
        elif n.kind == "aggregate":
            q, k = select(n, BIG / 2)                                 # a selected key beats BOS, BOS beats the rest
            cs = np.unique(n.f)
            v, o = np.zeros((d, len(cs))), np.zeros((len(cs), d))
            for j, c in enumerate(cs):
                v[at[n.args[2]] + np.flatnonzero(n.f == c), j] = 1.0
                o[j, out + c] = 1.0
            b.head(L, q, k, v, o)
            if not n.total:                                           # nothing came back: write the default
                u = unit(L)
                w1[tok.start:tok.start + K, u] = 1.0
                w1[out:out + n.size, u] = -1.0
                w2[u, out + n.default] = 1.0
        else:
            # BOS ties with selected keys: 1/(1+count). The tie turns an error in a key's or query's value into one
            # in the count, so it is scored no higher than it needs to be to shut out unselected keys.
            q, k = select(n, WIDTH_SCORE, WIDTH_SCORE)
            frac = at[n.name + "/frac"]
            v, o = np.zeros((d, 1)), np.zeros((1, d))
            v[bos, 0], o[0, frac] = 1.0, 1.0
            b.head(L, q, k, v, o)
            # Decode frac = 1/(1+count) with flat steps, not hats, so the result is exactly one-hot and errors
            # can't compound through a chain of widths. Each step is 1 once count > m, and it is 0 at BOS (frac = 1).
            # Each is only as steep as its own gap needs: a step far into its linear range would output 1 as the
            # difference of two big numbers, and any rounding in its weights would leak into the one-hot.
            vals = 1.0 / (np.arange(T + 1) + 1)
            u = unit(L)
            w1[tok.start:tok.start + K, u] = 1.0
            w2[u, out + n.f[0]] = 1.0
            for m in range(T):
                if n.f[m] == n.f[m + 1]:
                    continue
                tau, M = (vals[m] + vals[m + 1]) / 2, 4.0 / (vals[m] - vals[m + 1])
                for off, c in ((0.0, 1.0), (-1.0, -1.0)):
                    u = unit(L)
                    w1[frac, u], b1[u] = -M, M * tau + off
                    w2[u, out + n.f[m + 1]] += c
                    w2[u, out + n.f[m]] -= c
        heads[n.name] = [(L, j) for j in range(h0, b._head[L])]
        units[n.name] = [(L, u) for u in range(first, nxt[L])]
    o = at[prog.output]
    p["unembed"][o + np.arange(size[prog.output]), np.arange(size[prog.output])] = 10.0
    if trace is not None:
        trace.update(blocks=dict(b.blocks), heads=heads, units=units)
    return b.done()


# ---- disguise, spare capacity and change budgets, for the families that start from a compiled program ------------

def with_bos(prog: Program, X: np.ndarray) -> np.ndarray:
    return np.concatenate([np.full((len(X), 1), prog.K), X], 1)


def pad_model(p: nn.Params, meta: dict, pre: int = 0, post: int = 0, **grow) -> tuple[nn.Params, dict]:
    """The same function in a bigger architecture: `pre` empty layers in front, `post` after, and any of d, heads,
    dh, dm grown. Everything new is zero."""
    m = dict(meta, **grow, layers=meta["layers"] + pre + post)
    q = nn.zeros(m)
    for k, v in p.items():
        if k.startswith("L"):
            l, rest = k[1:].split(".", 1)
            k = f"L{int(l) + pre}.{rest}"
        q[k][tuple(slice(0, s) for s in v.shape)] = v
    return q, m


def junk(p: nn.Params, meta: dict, seed: int, heads: int, units: int, dims: int,
         like: nn.Params | None = None, fill: bool = False, read=(), entangle: bool = False,
         groups: list | None = None, probe: np.ndarray | None = None, spread: bool = False, blocks=(),
         also=()) -> tuple[nn.Params, dict]:
    """Add `dims` residual dims and, in every layer, `heads` heads and `units` MLP units that read like the model's
    own (their reads are copies of live ones in `like`, p by default, plus a little of the new dims and of `read`)
    but write only into the new dims, which nothing else reads. With fill, every head and unit that was empty becomes
    a decoy too, so no slot is left blank. No output changes.

    With entangle, decoys write into dims the model's own heads, units and unembed read, and come in groups of two or
    three in a layer that compute the same thing (same attention and values, or the same unit activation up to
    scale) and whose writes add up to zero: one decoy alone changes the output, the group changes nothing. The new
    dims are then read (harmlessly, being never written) but not written. groups, if given, gets (layer, "head" or
    "unit", slots) for each group. probe: inputs (BOS included) like the model's own, to size decoy writes on and
    check that each one matters (random tokens if not given).

    With spread too, groups are built like the model's own work (_spread): a decoy head is a twin of a real head's
    attention reading a real head's kind of values, a decoy unit a twin of a real unit, members sit in different
    layers, some ride inside a real head or unit, and writes are the size of real ones. groups then gets each group as
    a list of members (layer, "head" or "unit", slot, part), see silence(). blocks: one-hot blocks (dim arrays) of the
    compiled program, for reads that add nothing; also: models the delivered one will be turned into (an edit's
    reference), whose reads and writes the groups have to stay out of the way of."""
    like = p if like is None else like
    if entangle and spread:
        return _spread(p, meta, seed, heads, units, dims, like, fill, read, groups, probe, blocks, also)
    if entangle:
        return _entangled(p, meta, seed, heads, units, dims, like, fill, read, groups, probe)
    g = np.random.default_rng(seed)
    L, H, d, dh, dm = (meta[k] for k in ("layers", "heads", "d", "dh", "dm"))
    q, m = pad_model(p, meta, d=d + dims, heads=H + heads, dm=dm + units)
    S = np.arange(d, d + dims)
    R = np.concatenate([np.asarray(read, int), S])
    live_h = [(l, j) for l in range(L) for j in range(H) if np.any(like[f"L{l}.o"][j])]
    live_u = [(l, u) for l in range(L) for u in range(dm) if np.any(like[f"L{l}.mlp2"][u])]

    def sparse(shape, scale):
        return g.normal(0, scale, shape) * (g.random(shape) < 0.5)

    def write(shape):                                  # sparse, but never all zero
        w = sparse(shape, 1.0)
        w.flat[g.integers(w.size)] = g.normal(0, 1.0)
        return w

    for l in range(L):
        x = f"L{l}."
        empty_h = [j for j in range(H) if fill and not any(np.any(p[x + w][j]) for w in "qkvo")]
        empty_u = [u for u in range(dm) if fill and not (np.any(p[x + "mlp1"][:, u]) or p[x + "b1"][u]
                                                         or np.any(p[x + "mlp2"][u]))]
        for j in empty_h + list(range(H, H + heads)):
            if live_h:
                l2, j2 = live_h[g.integers(len(live_h))]
                for w in "qkv":
                    q[x + w][j, :d] = like[f"L{l2}.{w}"][j2]
            q[x + "v"][j][R] = sparse((len(R), dh), 0.3)
            # writes scaled to the reads, so a decoy's output stays as small as the live ones' and the new dims
            # can't grow layer after layer (a big hidden component turns into cancellation once disguised)
            q[x + "o"][j][:, S] = write((dh, dims)) / max(1.0, np.abs(q[x + "v"][j]).sum(0).max())
        for u in empty_u + list(range(dm, dm + units)):
            if live_u:
                l2, u2 = live_u[g.integers(len(live_u))]
                q[x + "mlp1"][:d, u], q[x + "b1"][u] = like[f"L{l2}.mlp1"][:, u2], like[f"L{l2}.b1"][u2]
            q[x + "mlp1"][R, u] = sparse(len(R), 0.3)
            q[x + "mlp2"][u, S] = write(dims) / max(1.0, np.abs(q[x + "mlp1"][:, u]).sum() + abs(q[x + "b1"][u]))
    return q, m


# An entangled decoy's writes start this small next to its own activation and grow until taking it out alone
# changes the output: the group cancels, but under rounding only to within its size, and the compiled model is steep in
# places (a selector_width turns a small error in its query into a wrong count).
QUIET = 0.01


def _entangled(p, meta, seed, heads, units, dims, like, fill, read, groups, probe):
    g = np.random.default_rng(seed)
    L, H, d, dh, dm = (meta[k] for k in ("layers", "heads", "d", "dh", "dm"))
    q, m = pad_model(like, meta, d=d + dims, heads=H + heads, dm=dm + units)     # built on like, copied to p
    S = np.arange(d, d + dims)
    R = np.concatenate([np.asarray(read, int), S])
    live_h = [(l, j) for l in range(L) for j in range(H) if np.any(like[f"L{l}.o"][j])]
    live_u = [(l, u) for l in range(L) for u in range(dm) if np.any(like[f"L{l}.mlp2"][u])]
    made = []
    # decoys write where the model's own reads look, but not into the token block (its sum is the constant direction
    # disguise pays bias shifts back through) nor where a unit reads steeply (a selector_width's count, whose steps
    # would turn a group's rounding into a wrong count)
    seen = np.abs(like["unembed"]).sum(1) > 0
    steep = np.zeros(d, bool)
    for l in range(L):
        seen |= np.abs(like[f"L{l}.mlp1"]).sum(1) > 0
        steep |= np.abs(like[f"L{l}.mlp1"]).max(1, initial=0.0) > 2.0
        for w in "qkv":
            seen |= np.abs(like[f"L{l}.{w}"]).sum((0, 2)) > 0
    tgt = np.flatnonzero(seen & ~steep & ~(np.abs(like["embed"]).sum(0) > 0))
    K = int((np.abs(like["embed"]).sum(1) > 0).sum()) - 1
    X = np.concatenate([np.full((64, 1), K), g.integers(0, max(K, 1), (64, meta["ctx"] - 1))], 1) \
        if probe is None else np.asarray(probe)

    def sparse(shape, scale):
        return g.normal(0, scale, shape) * (g.random(shape) < 0.5)

    def write(shape):                                  # sparse, but never all zero
        w = sparse(shape, 1.0)
        w.flat[g.integers(w.size)] = g.normal(0, 1.0)
        return w

    def split(slots):
        slots, out = list(slots), []
        while slots:
            k = 3 if len(slots) == 3 or (len(slots) >= 5 and g.random() < 0.5) else 2
            out.append(slots[:k])
            slots = slots[k:]
        return out

    # the model's stream without decoys is the stream with them (groups cancel): decoys copy reads that vary there
    pre, mid = _streams(q, m, X)
    base = (pre[L] @ q["unembed"]).argmax(-1)[:, 1:]

    vary = {}

    def varying(l, kind):
        if (l, kind) not in vary:
            if kind == "unit":
                cols = [q[f"L{a}.mlp1"][:, b] for a, b in live_u]
                y = np.maximum(mid[l] @ np.array(cols).T + np.array([q[f"L{a}.b1"][b] for a, b in live_u]), 0.0) \
                    if cols else np.zeros((1, 1, 0))
                ok = np.ptp(y, axis=(0, 1)) > 1e-6
            else:
                ok = [np.ptp(pre[l] @ q[f"L{a}.v"][b], axis=(0, 1)).max() > 1e-6 for a, b in live_h]
            vary[l, kind] = [s for s, k in zip(live_u if kind == "unit" else live_h, ok) if k]
        return vary[l, kind]

    def head_group(l, js):
        x = f"L{l}."
        # attention copied from an aggregate's head, which is saturated: a selector_width's ties would let rounding
        # pull the group's patterns apart
        sharp = [s for s in live_h if like[f"L{s[0]}.q"][s[1]].max() >= BIG / 2] or live_h
        src = [s for s in sharp if s[0] <= l] or sharp
        vsrc = varying(l, "head") or src
        (l2, j2), (l3, j3) = (src[g.integers(len(src))], vsrc[g.integers(len(vsrc))]) if src else ((None,) * 2,) * 2
        qk, v = np.zeros((2, d + dims, dh)), np.zeros((d + dims, dh))
        qk[:, :d] = [like[f"L{l2}.{w}"][j2] if src else sparse((d, dh), 0.3) for w in "qk"]
        v[:d] = like[f"L{l3}.v"][j3] if src else sparse((d, dh), 0.3)
        v[R] = sparse((len(R), dh), 0.3)
        h = pre[l]                                              # writes sized to what the head fetches
        y = nn._softmax((h @ qk[0]) @ (h @ qk[1]).transpose(0, 2, 1)) @ (h @ v)
        o = [QUIET * write((dh, len(tgt))) / max(np.abs(y[:, 1:]).max(), 0.1) for _ in js[1:]]
        o.insert(0, -sum(o))
        for j, oj in zip(js, o):
            for w, a in zip("qk", qk):
                q[x + w][j] = a
                q[x + w][j][S] = sparse((dims, dh), 0.3)       # never written, so read for nothing
            q[x + "v"][j] = v
            q[x + "v"][j][S] += sparse((dims, dh), 0.3)
            q[x + "o"][j] = 0.0
            q[x + "o"][j][:, tgt] = oj

    def unit_group(l, us):
        x = f"L{l}."
        act = varying(l, "unit")
        src = [s for s in act if s[0] == l] or act or live_u
        w, b = np.zeros(d + dims), 0.0
        if src:
            l2, u2 = src[g.integers(len(src))]
            w[:d], b = like[f"L{l2}.mlp1"][:, u2], like[f"L{l2}.b1"][u2]
        w[R] = sparse(len(R), 0.3)
        s = np.exp(g.uniform(-0.7, 0.7, len(us)))               # relu(s x) = s relu(x)
        y = np.maximum(mid[l][:, 1:] @ w + b, 0.0)              # writes sized to the activation
        o = [QUIET * write(len(tgt)) / max(y.max(), 0.1) for _ in us[1:]]
        o.insert(0, -sum(si * oi for si, oi in zip(s[1:], o)) / s[0])
        for u, si, oi in zip(us, s, o):
            q[x + "mlp1"][:, u], q[x + "b1"][u] = si * w, si * b
            q[x + "mlp1"][S, u] += sparse(dims, 0.3)
            q[x + "mlp2"][u] = 0.0
            q[x + "mlp2"][u, tgt] = oi

    def matters(l, kind, i):                         # member i taken out of the stream, the rest run on
        x = f"L{l}."
        if kind == "head":
            h = pre[l]
            a = nn._softmax((h @ q[x + "q"][i]) @ (h @ q[x + "k"][i]).transpose(0, 2, 1))
            return _rest(q, m, mid[l] - a @ (h @ q[x + "v"][i]) @ q[x + "o"][i], l, base)
        act = np.maximum(mid[l] @ q[x + "mlp1"][:, i] + q[x + "b1"][i], 0.0)
        return _rest(q, m, pre[l + 1] - act[..., None] * q[x + "mlp2"][i], l + 1, base, attn=True)

    for l in range(L):
        x = f"L{l}."
        empty_h = [j for j in range(H) if fill and not any(np.any(like[x + w][j]) for w in "qkvo")]
        empty_u = [u for u in range(dm) if fill and not (np.any(like[x + "mlp1"][:, u]) or like[x + "b1"][u]
                                                         or np.any(like[x + "mlp2"][u]))]
        for kind, slots, build in (("head", empty_h + list(range(H, H + heads)), head_group),
                                   ("unit", empty_u + list(range(dm, dm + units)), unit_group)):
            w = x + ("o" if kind == "head" else "mlp2")
            for grp in split(slots) if len(slots) >= 2 else []:
                for _ in range(4):                                  # a group whose one member alone matters
                    build(l, grp)
                    for _ in range(6):                              # as quiet as still matters
                        if matters(l, kind, grp[0]):
                            break
                        q[w][grp] *= 3.0
                    else:
                        continue
                    break
                made.append((l, kind, grp))
    out, _ = pad_model(p, meta, d=d + dims, heads=H + heads, dm=dm + units)
    for l, kind, grp in made:
        x = f"L{l}."
        for i in grp:
            if kind == "head":
                for w in "qkvo":
                    out[x + w][i] = q[x + w][i]
            else:
                out[x + "mlp1"][:, i], out[x + "b1"][i], out[x + "mlp2"][i] = q[x + "mlp1"][:, i], q[x + "b1"][i], \
                    q[x + "mlp2"][i]
    if groups is not None:
        groups.extend(made)
    return out, m


RIDE = 0.5      # the chance a spread group has a member riding inside a real head or unit


def silence(p: nn.Params, group, keep=()) -> None:
    """Take a spread decoy group's members out of p, but those in keep ((layer, kind, slot)): a decoy slot stops
    writing; a member riding in a real head (part: the head dims it uses) or unit (part: its share of the unit's
    write) is removed and leaves the real work as it was."""
    for l, kind, i, part in group:
        if (l, kind, i) in keep:
            continue
        if kind == "head":
            p[f"L{l}.o"][i][slice(None) if part is None else np.asarray(part)] = 0.0
        elif part is None:
            p[f"L{l}.mlp2"][i] = 0.0
        else:
            p[f"L{l}.mlp2"][i] -= part


def _spread(p, meta, seed, heads, units, dims, like, fill, read, groups, probe, blocks, also):
    """junk(entangle=True, spread=True). Every decoy computes something real: a head copies a real head's attention
    and reads values a real head reads, a unit copies a real unit's input,
    and reads that add nothing (differences of two one-hot blocks) keep twins from being equal weight for weight. A
    group's members sit in different layers where they can; where one has written and the last hasn't, the partial sum
    is in dims no real component (nor `also`'s) reads at those points, and every member reads only dims that are
    final by its own layer, so they compute the same thing. Some groups have a member inside a real head (spare head
    dims, the same attention) or real unit (an extra write), so no set of whole heads or units is ever the group."""
    g = np.random.default_rng(seed)
    L, H, d, dh, dm = (meta[k] for k in ("layers", "heads", "d", "dh", "dm"))
    grow = dict(d=d + dims, heads=H + heads, dm=dm + units)
    q, m = pad_model(like, meta, **grow)
    alts = [pad_model(a, meta, **grow)[0] for a in also]
    D, top = d + dims, 2 * L                      # read points: layer l's attention 2l, its MLP 2l + 1, unembed 2L
    S = np.arange(d, D)
    K = int((np.abs(like["embed"]).sum(1) > 0).sum()) - 1
    tok = np.abs(like["embed"][:K + 1]).argmax(1)
    X = (np.concatenate([np.full((64, 1), K), g.integers(0, max(K, 1), (64, meta["ctx"] - 1))], 1)
         if probe is None else np.asarray(probe))
    outs = np.flatnonzero(np.abs(like["unembed"]).sum(0) > 0)
    for x in S:                                   # the new dims are read out too, on the scale the output is
        c = g.choice(outs, min(len(outs), 1 + g.integers(0, 3)), replace=False)
        q["unembed"][x, c] = g.choice([-1.0, 1.0], len(c)) * g.uniform(5.0, 10.0, len(c))

    def mask(idx):
        z = np.zeros(D, bool)
        z[np.asarray(idx, int)] = True
        return z

    last = np.full(D, -2)                         # the last point anything writes a dim (-1: embedding)
    last[np.abs(q["embed"]).sum(0) + np.abs(q["pos"]).sum(0) > 0] = -1
    rd = np.zeros((top + 1, D), bool)             # where real components read each dim
    lever = np.zeros((top + 1, D), bool)          # ... where a write the size of a real one changes what they do
    steep = np.zeros(D, bool)
    for P in [q] + alts:
        for l in range(L):
            x = f"L{l}."
            if P is q:
                last[np.abs(P[x + "o"]).sum((0, 1)) > 0] = 2 * l
                last[np.abs(P[x + "mlp2"]).sum(0) + np.abs(P[x + "b2"]) > 0] = 2 * l + 1
            rd[2 * l] |= sum(np.abs(P[x + w]).sum((0, 2)) for w in "qkv") > 0
            rd[2 * l + 1] |= np.abs(P[x + "mlp1"]).sum(1) > 0
            steep |= np.abs(P[x + "mlp1"]).max(1, initial=0.0) > 2.0
            for j in range(P[x + "q"].shape[0]):
                qk = (np.abs(P[x + "q"][j]).sum(1) + np.abs(P[x + "k"][j]).sum(1)) > 0
                if np.abs(P[x + "q"][j]).max() >= BIG / 2:   # an aggregate's select: a unit of error moves it
                    lever[2 * l] |= qk
                elif np.any(P[x + "q"][j]):               # a selector_width counts: its ties turn a small error
                    steep |= qk                           # in a query or key into a wrong count
        rd[top] |= np.abs(P["unembed"]).sum(1) > 0
        lever[top] |= np.abs(P["unembed"]).sum(1) > 0
    moved, changed = [], set()                    # what `also` writes differently, at which point; what it changes
    for A in alts:
        for l in range(L):
            x = f"L{l}."
            moved.append((2 * l, np.abs(A[x + "o"] - q[x + "o"]).sum((0, 1)) > 0))
            moved.append((2 * l + 1, np.abs(A[x + "mlp2"] - q[x + "mlp2"]).sum(0) + np.abs(A[x + "b2"] - q[x + "b2"])
                          > 0))
            changed |= {(l, "head", j) for j in range(H + heads)
                        if any(np.any(A[x + w][j] != q[x + w][j]) for w in "qkvo")}
            changed |= {(l, "unit", u) for u in range(dm + units)
                        if np.any(A[x + "mlp1"][:, u] != q[x + "mlp1"][:, u]) or A[x + "b1"][u] != q[x + "b1"][u]
                        or np.any(A[x + "mlp2"][u] != q[x + "mlp2"][u])}
    hold = mask(tok) | steep | mask(read)         # never written by decoys
    idle = mask([r for r in np.asarray(read, int) if last[r] == -2 and not any(mv[r] for _, mv in moved)])
    onehot = [mask(b) for b in blocks] + [mask(tok[:K])]

    pre, mid = _streams(q, m, X)
    base = (pre[L] @ q["unembed"]).argmax(-1)[:, 1:]
    bos = tok[K]
    T = meta["ctx"] - 1
    idx = np.abs(like["pos"][1:T + 1]).argmax(1) if np.any(like["pos"][1:T + 1]) else np.zeros(0, int)
    real_h, real_u = [], []
    for l in range(L):
        x = f"L{l}."
        for j in range(H):
            if not np.any(like[x + "o"][j]):
                continue
            qq, kk, vv, oo = (q[x + w][j].copy() for w in "qkvo")
            real_h.append(dict(l=l, i=j, q=qq, k=kk, v=vv, ov=np.linalg.norm(vv @ oo),
                               qk=(np.abs(qq).sum(1) + np.abs(kk).sum(1)) > 0,
                               sharp=qq.max() >= BIG / 2,
                               vr=np.abs(vv).sum(1) > 0, free=[c for c in range(dh) if not np.any(vv[:, c]) and not np.any(oo[c])]))
        for u in range(dm):
            if not np.any(like[x + "mlp2"][u]):
                continue
            w, b = q[x + "mlp1"][:, u].copy(), q[x + "b1"][u]
            y = np.maximum(mid[l][:, 1:] @ w + b, 0.0)
            real_u.append(dict(l=l, i=u, w=w, b=b, r=np.abs(w) > 0, steep=np.abs(w).max() > 2.0,
                               vary=y.std() > 1e-6 and (y > 1e-9).mean() > 0.01, fire=float((y > 1e-9).mean())))
    slots = {"head": [], "unit": []}
    for l in range(L):
        x = f"L{l}."
        slots["head"] += [(l, j) for j in range(H) if fill and not any(np.any(like[x + w][j]) for w in "qkvo")]
        slots["head"] += [(l, j) for j in range(H, H + heads)]
        slots["unit"] += [(l, u) for u in range(dm) if fill and not (np.any(like[x + "mlp1"][:, u]) or like[x + "b1"][u]
                                                                  or np.any(like[x + "mlp2"][u]))]
        slots["unit"] += [(l, u) for u in range(dm, dm + units)]

    # what's partial at each read point (a group has written and its last member hasn't), and what decoys read there
    win, dread = np.zeros((top + 1, D), bool), np.zeros((top + 1, D), bool)

    def window(lo, hi, dz):
        win[lo + 1:hi + 1] |= mask(dz)

    def dreads(r, z):
        dread[r] |= z

    def free_at(r, z):                            # z can be read at r: final there, and no group's partial sum in it
        return (last[z] < r).all() and not (win[r] & z).any()

    def stable(points, z):                        # `also` doesn't rewrite z between the members' reads
        lo, hi = min(points), max(points)
        return not any(lo <= w < hi and (mv & z).any() for w, mv in moved)

    def null(r, n, avoid=None):                   # n read vectors that are zero at every position of every input
        ok = [b for b in onehot if b.any() and free_at(r, b) and (avoid is None or not (b & avoid).any())]
        out = np.zeros((D, n))
        if len(ok) >= 2:
            for c in range(n):
                for _ in range(2):
                    a, b = g.choice(len(ok), 2, replace=False)
                    out[:, c] += g.normal(0, 0.5) * (ok[a].astype(float) - ok[b])
        return out

    def targets(lo, hi, own):                     # dims a group can write: unseen while it's partial, read after
        z = ~hold & rd[hi + 1:].any(0) & ~rd[lo + 1:hi + 1].any(0) & ~own & ~dread[lo + 1:hi + 1].any(0)
        return np.flatnonzero(z)

    def choose(dz, k, hi):                        # k of them, first one where a write that size tells
        dz = list(g.permutation(dz))
        first = [x for x in dz if x < d and lever[hi + 1:, x].any()][:1] or [x for x in dz if x < d][:1]
        return np.array(first + [x for x in dz if x not in first][:max(0, k - len(first))], int)

    def rows(n, dz):                              # n writes the size of a real one, mostly into dz[0]
        o = np.zeros((n, D))
        o[:, dz] = g.normal(0, 0.3, (n, len(dz)))
        o[:, dz[0]] = g.choice([-1.0, 1.0], n) * g.uniform(1.0, 1.4, n)
        return o

    def pick(free, src_ok, k, used):              # up to k slots for twins, other layers first
        good = {l: src_ok((l, None)) for l in {s[0] for s in free}}
        cands = [s for s in free if good[s[0]]]
        g.shuffle(cands)
        cands.sort(key=lambda s: s[0] in used)
        out = []
        for s in cands:
            if len(out) < k and s[0] not in {t[0] for t in out}:
                out.append(s)
        return out

    made, alive, uses = [], {}, collections.Counter()

    def head_group(first, free, t):
        r0, local = 2 * first[0], t >= 30
        # attention copied from an aggregate's head, which is saturated: a selector_width's ties would let rounding
        # pull the members' patterns apart
        sharp = [h for h in real_h if h["sharp"]] or real_h
        srcs = [h for h in sharp if free_at(r0, h["qk"])]
        if local:                                 # all in one layer: ride in a head there if no other slot is free
            srcs = sharp if [s for s in free if s[0] == first[0]] else [h for h in sharp if h["l"] == first[0]
                                                                        and h["free"]]
        if srcs and not 20 <= t < 30:
            w = np.array([1.0 / (1 + uses[h["l"], "head", h["i"]]) ** 2 for h in srcs])  # spread over sources:
            src = srcs[g.choice(len(srcs), p=w / w.sum())]     # many heads sharing one pattern would tell
        elif not local and len(idx):              # a select on the positions the model doesn't have
            qq, kk = np.zeros((D, dh)), np.zeros((D, dh))
            n = min(len(idx), dh - 1)
            keys = g.permutation(len(idx))[:n]
            kk[idx[keys], np.arange(n)] = 1.0
            for i in range(len(idx)):
                if g.random() < 0.8:
                    qq[idx[i], g.integers(n)] = BIG
            qq[idx, n], qq[bos, n], kk[bos, n] = BIG / 2, BIG / 2, 1.0     # every position sees BOS at BIG / 2
            src = dict(l=-1, i=-1, q=qq, k=kk, qk=mask(idx) | mask([bos]), free=[])
        else:
            return None
        ok = (lambda s: s[0] == first[0]) if local else (lambda s: free_at(2 * s[0], src["qk"]))
        ride = src["l"] >= 0 and bool(src["free"]) and (src["l"], "head", src["i"]) not in changed and \
            (not local or src["l"] == first[0])
        ride = ride and (g.random() < RIDE or not [s for s in free if ok(s)])
        k = (3 if g.random() < 0.4 else 2) - 1 - ride
        k += len(free) - k == 1                   # don't leave one slot on its own
        twins = [first] + pick(free, ok, k, {first[0]} | ({src["l"]} if ride else set()))
        if len(twins) + ride < 2:
            return None
        pts = [2 * l for l, _ in twins] + ([2 * src["l"]] if ride else [])
        if not local and not (stable(pts, src["qk"]) and all(free_at(r, src["qk"]) for r in pts)):
            return None
        cap = len(src["free"]) if ride else dh
        # values an aggregate reads (a selector_width's value is the BOS flag, which a saturated select never sees)
        vs = [h for h in real_h if not h["vr"][bos]
              and (local or (all(free_at(r, h["vr"]) for r in pts) and stable(pts, h["vr"])))]
        if vs:
            vh = vs[g.integers(len(vs))]
            vsrc, size = vh["v"], vh["ov"]
        elif not local and stable(pts, mask(tok)):            # classes of tokens
            vsrc = np.zeros((D, dh))
            vsrc[tok[:K], g.integers(0, max(1, min(dh, K // 2)), K)] = 1.0
            size = np.linalg.norm(vsrc)
        elif not local and len(idx):                          # or of positions
            vsrc = np.zeros((D, dh))
            vsrc[idx, g.integers(0, max(1, min(dh, len(idx) // 2)), len(idx))] = 1.0
            size = np.linalg.norm(vsrc)
        else:
            return None
        cols = np.flatnonzero(np.abs(vsrc).sum(0) > 0)
        cols = g.permutation(cols)[:g.integers(1, min(cap, len(cols)) + 1)]
        V = vsrc[:, cols].copy()
        V[idle] = g.normal(0, 0.3, (int(idle.sum()), len(cols))) * (g.random((int(idle.sum()), len(cols))) < 0.5)
        lo, hi = min(pts), max(pts)
        own = src["qk"] | (np.abs(V).sum(1) > 0)
        dz = targets(lo, hi, own if lo < hi else np.zeros(D, bool))
        if len(dz) == 0:
            return None
        n = min(len(cols), len(dz))
        V = V[:, :n]
        dz = choose(dz, n + g.integers(0, 4), hi)
        nz = lambda r: mask(dz) if lo < r <= hi else None
        O = [rows(n, dz) for _ in range(len(twins) + ride - 1)]
        O.insert(0, -sum(O))
        Vs = [vread(2 * l, V, local, nz) for l, _ in twins] + ([vread(2 * src["l"], V, local, nz)] if ride else [])
        # as big as the real head the values come from: scaling every write alike keeps the group's sum zero
        scale = size * g.uniform(0.8, 1.1) / np.median([np.linalg.norm(v @ o) for v, o in zip(Vs, O)])
        O = [o * scale for o in O]
        members = []
        for (l, j), o, v in zip(twins, O, Vs):
            head_twin(l, j, src, v, o, local, nz)
            members.append((l, "head", j, None))
        if ride:
            x, c = f"L{src['l']}.", np.array(src["free"][:n])
            src["free"] = src["free"][n:]
            q[x + "v"][src["i"]][:, c] = Vs[-1]
            q[x + "o"][src["i"]][c] = O[-1]
            dreads(2 * src["l"], np.abs(q[x + "v"][src["i"]][:, c]).sum(1) > 0)
            members.append((src["l"], "head", src["i"], c))
        window(lo, hi, dz)
        info.append(dict(src=src, V=V, dz=dz, lo=lo, hi=hi, local=local, nz=nz, coef=[1.0] * len(members),
                         scale=scale))
        return members

    def head_twin(l, j, src, V, o, local, nz):
        x, r = f"L{l}.", 2 * l
        used = np.flatnonzero(np.abs(src["q"]).sum(0) + np.abs(src["k"]).sum(0) > 0)
        for w in "qk":
            a = src[w].copy()
            if not local:
                a[:, used] += null(r, 1, nz(r)) * g.normal(0, 0.3, len(used)) * np.abs(src[w]).max()
            q[x + w][j] = a
        q[x + "v"][j], q[x + "o"][j] = 0.0, 0.0
        q[x + "v"][j][:, :V.shape[1]] = V
        q[x + "o"][j][:V.shape[1]] = o
        dreads(r, sum(np.abs(q[x + w][j]).sum(1) for w in "qkv") > 0)

    def vread(r, V, local, nz):                   # V plus reads that add nothing, a quarter its size
        if local:
            return V.copy()
        N = null(r, V.shape[1], nz(r))
        n = np.linalg.norm(N, axis=0)
        return V + N * np.where(n > 0, 0.25 * np.linalg.norm(V, axis=0) / np.maximum(n, 1e-300), 0.0)

    def unit_twin(l, u, src, w, si, o, local, nz):
        x, r = f"L{l}.", 2 * l + 1
        q[x + "mlp1"][:, u] = si * (w + (0 if local else null(r, 1, nz(r))[:, 0]))
        q[x + "b1"][u], q[x + "mlp2"][u] = si * src["b"], o
        dreads(r, np.abs(q[x + "mlp1"][:, u]) > 0)

    def member_reads(l, kind, i, part):
        x = f"L{l}."
        if kind == "unit":
            return np.abs(q[x + "mlp1"][:, i]) > 0
        if part is not None:
            return np.abs(q[x + "v"][i][:, part]).sum(1) > 0
        return sum(np.abs(q[x + w][i]).sum(1) for w in "qkv") > 0

    def join(kind, slot, grps):                   # one slot left over: one more twin in a group it fits in
        l, i = slot
        r = 2 * l + (kind == "unit")
        for k in g.permutation(len(grps)):
            grp, a = grps[k]
            src, (l0, _, i0, _) = a["src"], grp[0]
            reads = src["qk"] if kind == "head" else src["r"]
            lo, hi = min(a["lo"], r), max(a["hi"], r)
            if a["local"] and r != a["lo"]:
                continue
            if not a["local"]:
                pts = [2 * mb[0] + (kind == "unit") for mb in grp] + [r]
                vr = np.abs(a["V"]).sum(1) > 0 if kind == "head" else reads
                if not (free_at(r, reads) and free_at(r, vr) and stable(pts, reads) and stable(pts, vr)):
                    continue
                dz = mask(a["dz"])                # the window may grow, if nothing reads its dims where it grows
                grown = np.zeros(top + 1, bool)
                grown[lo + 1:a["lo"] + 1] = grown[a["hi"] + 1:hi + 1] = True
                if ((rd[grown] | dread[grown]) & dz).any() or not (rd[hi + 1:] & dz).any():
                    continue
                if lo < r <= hi and ((reads | vr) & dz).any():
                    continue
                if any(lo < 2 * ml + (kind == "unit") <= hi and (member_reads(ml, kind, mi, part) & dz).any()
                       for ml, _, mi, part in grp):
                    continue
                window(lo, hi, a["dz"])
                a["lo"], a["hi"], a["nz"] = lo, hi, (lambda rr, lo=lo, hi=hi, dz=dz: dz if lo < rr <= hi else None)
            if kind == "head":
                c, o = 1.0, rows(a["V"].shape[1], a["dz"]) * a["scale"]
                head_twin(l, i, src, vread(r, a["V"], a["local"], a["nz"]), o, a["local"], a["nz"])
                q[f"L{l0}.o"][i0][:a["V"].shape[1]] -= c * o / a["coef"][0]
            else:
                c = float(np.exp(g.uniform(-0.7, 0.7)))
                o = rows(1, a["dz"])[0]
                unit_twin(l, i, src, a["w"], c, o, a["local"], a["nz"])
                q[f"L{l0}.mlp2"][i0] -= c * o / a["coef"][0]
            grp.append((l, kind, i, None))
            a["coef"].append(c)
            return True
        return False

    def unit_group(first, free, t):
        r0, local = 2 * first[0] + 1, t >= 30
        srcs = [u for u in real_u if not u["steep"] and u["vary"] and (local or free_at(r0, u["r"]))]
        if local and not [s for s in free if s[0] == first[0]]:
            srcs = [u for u in srcs if u["l"] == first[0]]
        if srcs and not 20 <= t < 30:             # the more often it fires, the more often a lone twin tells
            f = np.array([u["fire"] / (1 + uses[u["l"], "unit", u["i"]]) ** 2 for u in srcs])
            src = srcs[g.choice(len(srcs), p=f / f.sum())]
        else:                                     # a map of the tokens the model doesn't have
            z = np.zeros(D)
            z[tok[g.permutation(K)[:max(1, K // 2)]]] = 1.0
            src = dict(l=-1, i=-1, w=z, b=0.0, r=z > 0)
        def fires(l):                             # a default unit, say, is dead once its own write is in
            key = (src["l"], src["i"], l)
            if key not in alive or src["l"] < 0:
                y = np.maximum(mid[l][:, 1:] @ src["w"] + src["b"], 0.0)
                alive[key] = y.std() > 1e-6 and (y > 1e-9).mean() > 0.01
            return alive[key]

        ok = (lambda s: s[0] == first[0] and fires(s[0])) if local else \
            (lambda s: free_at(2 * s[0] + 1, src["r"]) and fires(s[0]))
        if not fires(first[0]):
            return None
        ride = src["l"] >= 0 and (src["l"], "unit", src["i"]) not in changed and (not local or src["l"] == first[0])
        ride = ride and (g.random() < RIDE or not [s for s in free if ok(s)])
        k = (3 if g.random() < 0.4 else 2) - 1 - ride
        k += len(free) - k == 1                   # don't leave one slot on its own
        twins = [first] + pick(free, ok, k, {first[0]} | ({src["l"]} if ride else set()))
        if len(twins) + ride < 2:
            return None
        pts = [2 * l + 1 for l, _ in twins] + ([2 * src["l"] + 1] if ride else [])
        if not local and not (stable(pts, src["r"]) and all(free_at(r, src["r"]) for r in pts)):
            return None
        lo, hi = min(pts), max(pts)
        w = src["w"].copy()
        if not ride:
            w[idle] = g.normal(0, 0.3, int(idle.sum())) * (g.random(int(idle.sum())) < 0.5)
        dz = targets(lo, hi, (np.abs(w) > 0) if lo < hi else np.zeros(D, bool))
        if len(dz) == 0:
            return None
        dz = choose(dz, 1 + g.integers(0, 4), hi)
        nz = lambda r: mask(dz) if lo < r <= hi else None
        s = list(np.exp(g.uniform(-0.7, 0.7, len(twins)))) + ([1.0] if ride else [])
        o = [rows(1, dz)[0] for _ in s[1:]]
        o.insert(0, -sum(a * b for a, b in zip(s[1:], o)) / s[0])
        members = []
        for (l, u), si, oi in zip(twins, s, o):
            unit_twin(l, u, src, w, si, oi, local, nz)
            members.append((l, "unit", u, None))
        if ride:
            q[f"L{src['l']}.mlp2"][src["i"]] += o[-1]
            members.append((src["l"], "unit", src["i"], o[-1].copy()))
        window(lo, hi, dz)
        info.append(dict(src=src, w=w, dz=dz, lo=lo, hi=hi, local=local, nz=nz, coef=list(s)))
        return members

    def out_of(P, n=None):
        return (nn.tfm_forward(P, m, X[:n]).argmax(-1)[:, 1:] != base[:n]).any()

    def told(grp, n=24):                          # every decoy in it alone changes the output
        pre_c, mid_c = _streams(q, m, X[:n])
        for l, kind, i, _ in grp:
            x = f"L{l}."
            if kind == "head":
                h = pre_c[l]
                a = nn._softmax((h @ q[x + "q"][i]) @ (h @ q[x + "k"][i]).transpose(0, 2, 1))
                if not _rest(q, m, mid_c[l] - a @ (h @ q[x + "v"][i]) @ q[x + "o"][i], l, base[:n]):
                    return False
            else:
                act = np.maximum(mid_c[l] @ q[x + "mlp1"][:, i] + q[x + "b1"][i], 0.0)
                if not _rest(q, m, pre_c[l + 1] - act[..., None] * q[x + "mlp2"][i], l + 1, base[:n], attn=True):
                    return False
        return True

    info = []
    for kind, build in (("head", head_group), ("unit", unit_group)):
        free = list(slots[kind])
        g.shuffle(free)
        free.sort(key=lambda s: s[0])             # early layers first: they have the fewest sources
        mine = []
        while free:
            first = free.pop(0)
            for t in range(40):                   # in other layers if it can be, else in this one
                snap = ({k: v.copy() for k, v in q.items()}, win.copy(), dread.copy(), len(info),
                        [list(h["free"]) for h in real_h])
                grp = build(first, free, t)
                if grp is not None and (t >= 30 or t % 10 == 9 or told([mb for mb in grp if mb[3] is None])):
                    break
                if grp is not None:               # try again where a lone decoy would show
                    q.update(snap[0])
                    win[:], dread[:] = snap[1], snap[2]
                    del info[snap[3]:]
                    for h, fr in zip(real_h, snap[4]):
                        h["free"] = fr
                grp = None
            else:
                if not join(kind, first, mine):
                    raise RuntimeError(f"no decoy group for {kind} slot {first}")
                continue
            for l, kd, i, part in grp:
                if part is None and (l, i) in free:
                    free.remove((l, i))
            made.append(grp)
            mine.append((grp, info[-1]))
            uses[info[-1]["src"]["l"], kind, info[-1]["src"]["i"]] += 1

    if out_of(q):
        raise RuntimeError("spread decoys changed the model's output")
    out, _ = pad_model(p, meta, **grow)
    out["unembed"][S] = q["unembed"][S]
    for grp in made:
        for l, kind, i, part in grp:
            x = f"L{l}."
            if kind == "head" and part is None:
                for w in "qkvo":
                    out[x + w][i] = q[x + w][i]
            elif kind == "head":
                out[x + "v"][i][:, part], out[x + "o"][i][part] = q[x + "v"][i][:, part], q[x + "o"][i][part]
            elif part is None:
                out[x + "mlp1"][:, i], out[x + "b1"][i], out[x + "mlp2"][i] = q[x + "mlp1"][:, i], q[x + "b1"][i], \
                    q[x + "mlp2"][i]
            else:
                out[x + "mlp2"][i] += part
    if groups is not None:
        groups.extend(made)
    return out, m


def onehot_blocks(trace: dict, output: str) -> list[np.ndarray]:
    """The residual blocks of a compiled program (compile_program's trace) that are one-hot at every data position and
    zero at BOS, for junk(blocks=): every variable's but the output's and the tokens' (junk finds those itself)."""
    return [np.arange(a, b) for x, (a, b) in trace["blocks"].items() if x not in ("tokens", output) and "/" not in x]


def _without(p, member):
    l, kind, i, part = member
    q = dict(p)
    w = f"L{l}." + ("o" if kind == "head" else "mlp2")
    q[w] = p[w].copy()
    silence(q, [member])
    return q


def _streams(p: nn.Params, meta: dict, X: np.ndarray):
    """A plain compiled model's residual stream before each layer's attention (and, last, the final one) and before
    each layer's MLP."""
    h = p["embed"][X] + p["pos"][:X.shape[1]]
    pre, mid = [], []
    for l in range(meta["layers"]):
        pre.append(h)
        h = h + _attn(p, meta, h, l)
        mid.append(h)
        h = h + np.maximum(h @ p[f"L{l}.mlp1"] + p[f"L{l}.b1"], 0.0) @ p[f"L{l}.mlp2"] + p[f"L{l}.b2"]
    return pre + [h], mid


def _attn(p, meta, h, l):
    n, t, d = h.shape
    q, k, v = (nn._heads(h, p[f"L{l}.{w}"]) for w in "qkv")
    a = nn._softmax(q @ k.transpose(0, 1, 3, 2))
    return (a @ v).transpose(0, 2, 1, 3).reshape(n, t, -1) @ p[f"L{l}.o"].reshape(-1, d)


def _rest(p, meta, h, l, base, attn=False) -> bool:
    """Whether running the rest of the model from h (before layer l's MLP, or with attn before its attention)
    changes the output at some data position."""
    for i in range(l, meta["layers"]):
        if attn or i > l:
            h = h + _attn(p, meta, h, i)
        h = h + np.maximum(h @ p[f"L{i}.mlp1"] + p[f"L{i}.b1"], 0.0) @ p[f"L{i}.mlp2"] + p[f"L{i}.b2"]
    return bool(((h @ p["unembed"]).argmax(-1)[:, 1:] != base).any())


def constant_read(p: nn.Params, meta: dict) -> np.ndarray | None:
    """A read vector that is 1 at every position of every input, if p has one: every token's embedding gives 1 on
    it and no other write moves it (in a compiled program, the sum of the token block)."""
    d = meta["d"]
    emb = p["embed"][np.abs(p["embed"]).sum(1) > 0]
    writes = [p["pos"]] + [p[f"L{l}.{w}"].reshape(-1, d) for l in range(meta["layers"]) for w in ("o", "mlp2", "b2")]
    A = np.vstack([emb] + writes)
    y = np.concatenate([np.ones(len(emb)), np.zeros(len(A) - len(emb))])
    r = np.linalg.lstsq(A, y, rcond=None)[0]
    return r if len(emb) and np.abs(A @ r - y).max() < 1e-9 else None


# The bias shift, in units of a compiled unit's margin (1), and the log-ranges of the key and query tilts. What they
# add cancels exactly but its rounding doesn't, and a compiled selector_width tie is only good to about 1e-3 of a
# value. A key tilt's rounding is scaled by the query (BIG-sized), a query tilt's only by the key (about 1), so the
# key tilt stays small.
SHIFT, TILT = 10.0, {"k": (0.0, 1.0), "q": (1.0, 3.0)}


def disguise(p: nn.Params, meta: dict, seed: int, rescale: float = 0.5, like: nn.Params | None = None,
             perms: list | None = None) -> nn.Params:
    """The same function in a different basis: the residual stream through a random rotation and per-dim rescaling,
    heads and MLP units permuted, each head's q/k and v/o dims mixed by random invertible maps. Every live unit's
    bias moves by a random amount, paid back through the residual's constant direction, every key reads a random
    amount of that direction too (which adds the same score to every key; queries do along head dims no key reads),
    and every head and unit is rescaled (q*c, k/c; v*c, o/c; a unit's input*c, output/c) so its two sides have equal
    norms: neither biases nor the sizes of weights say anything about the structure. All constants come from `like`
    (p by default), so two models disguised with the same seed and like differ by the same change, of the same rank
    (a head whose keys differ can gain one). perms, if given, gets each layer's (heads, units) permutation: new head
    j is old head heads[j], and the same for units."""
    g = np.random.default_rng(seed)
    like = p if like is None else like
    L, H, d, dh, dm = (meta[k] for k in ("layers", "heads", "d", "dh", "dm"))

    def basis(n):
        return np.linalg.qr(g.normal(size=(n, n)))[0] * np.exp(g.uniform(-rescale, rescale, n))

    M = basis(d).T                                   # h -> h @ M
    Mi = np.linalg.inv(M)
    one = constant_read(like, meta)
    scale = 0.0 if one is None else SHIFT
    size = {w: np.median([np.linalg.norm(like[f"L{l}.{w}"][j]) for l in range(L) for j in range(H)
                          if np.any(like[f"L{l}.{w}"][j])] or [0.0]) / np.sqrt(dh) for w in "kq"}
    mix = []
    for l in range(L):
        x = f"L{l}."
        hp, A, B, U = g.permutation(H), [basis(dh) for _ in range(H)], [basis(dh) for _ in range(H)], g.permutation(dm)
        live = np.abs(like[x + "mlp1"]).sum(0) + np.abs(like[x + "b1"]) + np.abs(like[x + "mlp2"]).sum(1) > 0
        shift = g.normal(0, scale, dm) * live           # big next to any bias, so a unit's own bias is lost in it
        # keys also read the constant direction, which adds the same score to every key, and so do queries, along
        # head dims no key reads: attention is unchanged, but the size of each head's weights and of its query-key
        # form becomes random, on the model's own scale rather than the head's
        tilt = [[np.zeros(dh) if one is None or not np.any(like[x + w][j]) else
                 g.normal(size=dh) * size[w] * np.exp(g.uniform(*TILT[w])) for j in range(H)] for w in "kq"]
        mix.append((hp, A, B, U, shift, tilt))
        if perms is not None:
            perms.append((hp, U))

    def move(p):
        out = {"embed": p["embed"] @ M, "pos": p["pos"] @ M, "unembed": Mi @ p["unembed"]}
        for l, (hp, A, B, U, shift, tilt) in enumerate(mix):
            x = f"L{l}."
            q, k = p[x + "q"][hp], p[x + "k"][hp]
            if one is not None:
                kt, qt = (np.stack([t[i] for i in hp]) for t in tilt)
                qt = qt - np.einsum("hij,hj->hi", np.linalg.pinv(k) @ k, qt)     # along dims no key reads
                q, k = q + one[None, :, None] * qt[:, None], k + one[None, :, None] * kt[:, None]
            out[x + "q"] = np.stack([Mi @ q[j] @ A[j] for j in range(H)])
            out[x + "k"] = np.stack([Mi @ k[j] @ np.linalg.inv(A[j]).T for j in range(H)])
            out[x + "v"] = np.stack([Mi @ p[x + "v"][hp[j]] @ B[j] for j in range(H)])
            out[x + "o"] = np.stack([np.linalg.inv(B[j]) @ p[x + "o"][hp[j]] @ M for j in range(H)])
            w1 = p[x + "mlp1"] - (0 if one is None else np.outer(one, shift))
            out[x + "mlp1"], out[x + "b1"] = (Mi @ w1)[:, U], (p[x + "b1"] + shift)[U]
            out[x + "mlp2"], out[x + "b2"] = p[x + "mlp2"][U] @ M, p[x + "b2"] @ M
        return out

    def ratio(a, b):                                 # c with |a c| = |b / c|
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return np.sqrt(nb / na) if na > 0 and nb > 0 else 1.0

    ref, out = move(like), move(p)
    for l in range(L):
        x = f"L{l}."
        for j in range(H):
            c = ratio(ref[x + "q"][j], ref[x + "k"][j])
            out[x + "q"][j] *= c
            out[x + "k"][j] /= c
            c = ratio(ref[x + "v"][j], ref[x + "o"][j])
            out[x + "v"][j] *= c
            out[x + "o"][j] /= c
        c = np.array([ratio(np.append(ref[x + "mlp1"][:, u], ref[x + "b1"][u]), ref[x + "mlp2"][u]) for u in range(dm)])
        out[x + "mlp1"], out[x + "b1"], out[x + "mlp2"] = out[x + "mlp1"] * c, out[x + "b1"] * c, \
            out[x + "mlp2"] / c[:, None]
    return out


RANK_TOL = 1e-6


def change_rank(p: nn.Params, base: nn.Params, tol: float = RANK_TOL) -> int:
    """How big a change is, whatever basis it was made in: the summed rank of (p - base) over tensors, per head for
    the (heads, ., .) ones, with a vector counting 1 if it moved. Singular values up to tol times the base's largest
    (or 1, if that's smaller) don't count, nor do moves of a vector up to tol times its largest entry. Rounding a
    tensor that was left alone stays under that even through float32. Rounding a change much larger than the tensor
    it sits in does not (float32 loses 6e-8 of the new value, not of the old), which is why statements say to keep
    float64: a tolerance tied to the submitted values instead would let one huge junk write buy small edits for free."""
    n = 0
    for k, b in base.items():
        dlt = np.asarray(p[k], dtype=np.float64) - b
        if not np.any(np.abs(dlt) > tol * max(1.0, np.abs(b).max(initial=0.0))):
            continue
        if dlt.ndim == 1:
            n += 1
        else:
            for x, y in zip(dlt.reshape(-1, *dlt.shape[-2:]), b.reshape(-1, *b.shape[-2:])):
                cut = tol * max(1.0, np.linalg.norm(y, 2))
                n += int((np.linalg.svd(x, compute_uv=False) > cut).sum())
    return n


def as_counted(p: nn.Params, base: nn.Params, tol: float = RANK_TOL) -> nn.Params:
    """p as change_rank counts it: singular values of each change that the count leaves out are taken out, and a
    tensor that counts as unchanged is base's. Graders run this, so what the budget doesn't count doesn't act."""
    out = {}
    for k, b in base.items():
        dlt = np.asarray(p[k], dtype=np.float64) - b
        if not np.any(np.abs(dlt) > tol * max(1.0, np.abs(b).max(initial=0.0))):
            out[k] = b
        elif dlt.ndim == 1:
            out[k] = p[k]
        else:
            new = dlt.copy()
            for x, y in zip(new.reshape(-1, *dlt.shape[-2:]), b.reshape(-1, *b.shape[-2:])):
                U, sv, Vt = np.linalg.svd(x, full_matrices=False)
                low = sv <= tol * max(1.0, np.linalg.norm(y, 2))
                x -= (U[:, low] * sv[low]) @ Vt[low]
            out[k] = b + new
    return out
