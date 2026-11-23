"""Small trained MLPs for families to work on, in plain numpy: seeded data, manual backprop, full-batch Adam.

Training is deterministic for a given seed on a given machine. Families grade against the model they handed out, not
a stored checkpoint, so float drift across machines doesn't change what a task asks.

Every dataset has inputs in [-1, 1] (bits are 0/1), so bounds on what a unit can see are easy to state.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np

from . import nn

GLYPHS = {   # 5x3 digits, row-major
    0: "111101101101111", 1: "010110010010111", 2: "111001111100111", 3: "111001111001111", 4: "101101111001001",
    5: "111100111001111", 6: "111100111101111", 7: "111001010010010", 8: "111101111101111", 9: "111101111001111",
}


@dataclass
class Data:
    name: str
    n: int
    k: int
    sample: Callable[[np.random.Generator, int], np.ndarray]   # rng, count -> (count, n) inputs
    label: Callable[[np.ndarray], np.ndarray] | None            # inputs -> int labels (None: labels come with sample)
    text: str                                                   # the input distribution, safe to show an agent
    sample_labeled: Callable[[np.random.Generator, int], tuple[np.ndarray, np.ndarray]] | None = None


def _balanced(y: np.ndarray, k: int, lo: float) -> bool:
    return np.bincount(y, minlength=k).min() >= lo * len(y)


def teacher(seed: int) -> Data:
    """Argmax of a random tanh network on uniform inputs."""
    rng = np.random.default_rng([seed, 1])
    n, k = int(rng.integers(8, 13)), int(rng.integers(3, 6))
    sample = lambda g, m: g.uniform(-1, 1, (m, n))
    probe = sample(np.random.default_rng([seed, 2]), 4096)
    while True:
        W1, b1 = rng.normal(0, 2.5 / np.sqrt(n), (32, n)), rng.normal(0, 0.5, 32)
        W2, b2 = rng.normal(0, 1 / np.sqrt(32), (k, 32)), rng.normal(0, 0.3, k)
        label = lambda X, W1=W1, b1=b1, W2=W2, b2=b2: np.argmax(np.tanh(X @ W1.T + b1) @ W2.T + b2, 1)
        if _balanced(label(probe), k, 0.6 / k):
            return Data("teacher", n, k, sample, label, f"{n} values, each independent and uniform in [-1, 1]")


def boolean(seed: int) -> Data:
    """A hidden formula over uniform random bits: a DNF, or a weighted threshold XORed with the parity of two bits."""
    rng = np.random.default_rng([seed, 3])
    n = int(rng.integers(8, 13))
    sample = lambda g, m: g.integers(0, 2, (m, n)).astype(np.float64)
    allx = np.array([[(i >> (n - 1 - j)) & 1 for j in range(n)] for i in range(2 ** n)], dtype=np.float64)
    while True:   # nothing a linear readout of a few samples gets right
        if rng.random() < 0.5:
            clauses = [(rng.choice(n, int(rng.integers(2, 4)), replace=False), rng.integers(0, 2, 3))
                       for _ in range(int(rng.integers(4, 8)))]
            label = lambda X, cl=clauses: np.any([np.all(X[:, v] == s[:len(v)], 1) for v, s in cl], 0).astype(int)
        else:
            w, t, par = rng.integers(-2, 3, n), rng.normal(0, 1), rng.choice(n, 2, replace=False)
            label = lambda X, w=w, t=t, par=par: ((X @ w > t) ^ (X[:, par].sum(1) % 2 == 1)).astype(int)
        if _balanced(label(allx), 2, 0.3):
            return Data("boolean", n, 2, sample, label, f"{n} bits, each independent and uniform over {{0, 1}}")


def glyphs(seed: int) -> Data:
    """Noisy 5x3 digit images: each pixel of a random digit's glyph flipped with some probability."""
    rng = np.random.default_rng([seed, 4])
    flip = float(rng.uniform(0.06, 0.14))
    G = np.array([[float(c) for c in GLYPHS[d]] for d in range(10)])

    def labeled(g, m):
        y = g.integers(0, 10, m)
        return np.abs(G[y] - (g.random((m, 15)) < flip)), y
    return Data("glyphs", 15, 10, lambda g, m: labeled(g, m)[0], None,
                "15 values, each 0 or 1, from a fixed distribution the sample is drawn from", labeled)


DATASETS = {"teacher": teacher, "boolean": boolean, "glyphs": glyphs}


def _forward(p, acts, X):
    hs, zs = [X], []
    for i, a in enumerate(acts):
        z = hs[-1] @ p[f"l{i}.w"].T + p[f"l{i}.b"]
        zs.append(z)
        hs.append(nn.ACTS[a](z))
    return zs, hs


def init(sizes: list[int], acts: list[str], seed: int) -> nn.Params:
    rng = np.random.default_rng(seed)
    p = {}
    for i in range(len(sizes) - 1):
        gain = 1.0 if acts[i] == "tanh" else np.sqrt(2.0)
        p[f"l{i}.w"] = rng.normal(0, gain / np.sqrt(sizes[i]), (sizes[i + 1], sizes[i]))
        p[f"l{i}.b"] = np.zeros(sizes[i + 1])
    return p


def fit(p: nn.Params, acts: list[str], X: np.ndarray, target: np.ndarray, *, loss: str = "ce", steps: int = 1500,
        lr: float = 5e-3, masks: dict[str, np.ndarray] | None = None, wd: float = 0.0, batch: int | None = None,
        dtype=np.float64, seed: int = 0) -> nn.Params:
    """Adam from p, full-batch unless `batch` is given. loss "ce": target is int labels, softmax cross-entropy on the
    outputs (the last activation should be 'identity'); "mse": target is the outputs themselves. masks pin entries to
    zero. dtype float32 trains big models faster; the result is float64 either way."""
    masks = {k: m.astype(dtype) for k, m in (masks or {}).items()}
    p = {k: (v * masks[k] if k in masks else v.copy()).astype(dtype) for k, v in p.items()}
    m1 = {k: np.zeros_like(v) for k, v in p.items()}
    m2 = {k: np.zeros_like(v) for k, v in p.items()}
    X = X.astype(dtype)
    Y = (np.eye(p[f"l{len(acts) - 1}.b"].size)[target] if loss == "ce" else target).astype(dtype)
    b1, b2, eps = 0.9, 0.999, 1e-8
    pick = np.random.default_rng([seed, 7])
    for t in range(1, steps + 1):
        Xb, Yb = X, Y
        if batch and batch < len(X):
            i = pick.choice(len(X), batch, replace=False)
            Xb, Yb = X[i], Y[i]
        zs, hs = _forward(p, acts, Xb)
        out = hs[-1]
        if loss == "ce":
            e = np.exp(out - out.max(1, keepdims=True))
            g = (e / e.sum(1, keepdims=True) - Yb) / len(Xb)
        else:
            g = 2 * (out - Yb) / Yb.size
        grads = {}
        for i in range(len(acts) - 1, -1, -1):
            if acts[i] == "relu":
                g = g * (zs[i] > 0)
            elif acts[i] == "tanh":
                g = g * (1 - hs[i + 1] ** 2)
            grads[f"l{i}.w"] = g.T @ hs[i] + wd * p[f"l{i}.w"]
            grads[f"l{i}.b"] = g.sum(0)
            g = g @ p[f"l{i}.w"]
        a = lr * np.sqrt(1 - b2 ** t) / (1 - b1 ** t)
        for k in p:
            m1[k] = b1 * m1[k] + (1 - b1) * grads[k]
            m2[k] = b2 * m2[k] + (1 - b2) * grads[k] ** 2
            p[k] = p[k] - a * m1[k] / (np.sqrt(m2[k]) + eps)
            if k in masks:
                p[k] = p[k] * masks[k]
    return {k: v.astype(np.float64) for k, v in p.items()}


def train(sizes: list[int], acts: list[str], X: np.ndarray, y: np.ndarray, *, seed: int, **kw) -> nn.Params:
    """A fresh classifier trained with cross-entropy; keyword arguments go to fit()."""
    return fit(init(sizes, acts, seed), acts, X, y, **kw)


def labeled(data: Data, rng: np.random.Generator, m: int) -> tuple[np.ndarray, np.ndarray]:
    if data.sample_labeled:
        return data.sample_labeled(rng, m)
    X = data.sample(rng, m)
    return X, data.label(X)


BIG_MLP = 10_000   # weights; above this, classifier() trains the fast way


@lru_cache(maxsize=256)
def classifier(kind: str, seed: int, hidden: tuple[int, ...], act: str = "relu", density: float = 1.0,
               steps: int = 1200) -> tuple[nn.Params, dict, Data]:
    """A trained classifier: (params, meta, data). density < 1 trains with a fixed random mask on every weight
    matrix, so that fraction of weights is exactly zero. Big ones train on minibatches in float32 for fewer steps,
    which keeps every model to a second or two. Cached, so treat the arrays as read-only."""
    data = DATASETS[kind](seed)
    X, y = labeled(data, np.random.default_rng([seed, 5]), 2048)
    sizes = [data.n, *hidden, data.k]
    acts = [act] * len(hidden) + ["identity"]
    masks = None
    if density < 1:
        g = np.random.default_rng([seed, 6])
        masks = {f"l{i}.w": (g.random((sizes[i + 1], sizes[i])) < density).astype(np.float64)
                 for i in range(len(sizes) - 1)}
    n = sum(a * b for a, b in zip(sizes, sizes[1:]))
    kw = {} if n <= BIG_MLP else dict(steps=int(np.clip(1e8 / n, 60, steps)), batch=256,
                                      dtype=np.float32, lr=2e-3)
    p = train(sizes, acts, X, y, seed=seed, **{"steps": steps, "masks": masks, **kw})
    for v in p.values():
        v.flags.writeable = False
    return p, {"kind": "mlp", "sizes": sizes, "acts": acts}, data


def predict(p: nn.Params, meta: dict, X: np.ndarray) -> np.ndarray:
    return nn.mlp_forward(p, meta, X).argmax(1)


# The size ladder every family draws from, named like _common.size_bucket so the tag and info["size"] agree.
SIZES = {"<1k": (200, 1e3), "1k-10k": (1e3, 1e4), "10k-100k": (1e4, 1e5), "100k-1M": (1e5, 1e6),
         "1M-100M": (1e6, 5e6)}


def draw_size(family: str, seed: int, sizes=tuple(SIZES)) -> str:
    """A seed's size bucket, from its own stream so it is independent of the tier and everything else."""
    return random.Random(f"{family}:size:{seed}").choice(list(sizes))


def width(params, size: str, rng: random.Random, lo: int = 2, slack: float = 1.3, cap: float | None = None) -> int:
    """A width whose parameter count params(w) lands log-uniformly inside the size bucket, away from its edges."""
    a, b = SIZES[size]
    b = min(b, cap or b)
    target = math.exp(rng.uniform(math.log(a * slack), math.log(b / slack)))
    w = lo
    while params(w * 2) < target:
        w *= 2
    hi = w * 2
    while hi - w > 1:
        mid = (w + hi) // 2
        w, hi = (mid, hi) if params(mid) <= target else (w, mid)
    return w


def cpu_seconds(params: int) -> float:
    """The agent's sandbox CPU budget: the old 120 s up to 10k parameters, growing with the square root after."""
    return round(min(3600.0, 120.0 * max(1.0, math.sqrt(params / 1e4))))


def size_of(params: float) -> str:
    return next((name for name, (_, b) in SIZES.items() if params < b), "1M-100M")
