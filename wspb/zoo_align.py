"""Small trained MLPs and CNNs for align-merge: the synthetic tasks they learn, a numpy trainer, and the cache.

Every model is trained in plain numpy (float64, Adam, one fixed data stream per seed), so nothing outside numpy is
needed and a model can be retrained on demand: train() is deterministic for a given machine and numpy build. Trained
weights are cached as safetensors under ~/.cache/wspb/align-merge/<hash of this file>/ (WSPB_ZOO_ALIGN overrides the
root), keyed by everything that goes into the run, so changing the trainer or a task starts a fresh cache.

Tasks are synthetic and drawn from a seed, because graders draw fresh data from them:
- teacher: Gaussian inputs labelled by the argmax of a random tanh network, classes balanced by its biases;
- clusters: a mixture of Gaussian blobs, each blob carrying one class;
- motifs: small one-channel images of noise with one of k 3x3 motifs stamped somewhere.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import nn


@dataclass(frozen=True)
class Data:
    kind: str           # "teacher", "clusters" or "motifs"
    seed: int
    n: int              # input dimension (motifs: image side)
    k: int              # classes

    @property
    def shape(self) -> tuple[int, ...]:
        return (1, self.n, self.n) if self.kind == "motifs" else (self.n,)

    @property
    def text(self) -> str:
        if self.kind == "motifs":
            return f"{self.n}x{self.n} one-channel images, shape (1, {self.n}, {self.n})"
        return f"vectors of {self.n} reals"

    def sample(self, g: np.random.Generator, m: int) -> tuple[np.ndarray, np.ndarray]:
        return _SAMPLERS[self.kind](self, g, m)


@lru_cache(maxsize=None)
def _teacher(d: Data):
    r = np.random.default_rng([d.seed, 11])
    h = d.n
    W1, b1 = r.normal(0, 1.0 / np.sqrt(d.n), (h, d.n)), r.normal(0, 0.3, h)
    W2 = r.normal(0, 1 / np.sqrt(h), (d.k, h))
    X = r.normal(size=(20000, d.n))
    Z = np.tanh(X @ W1.T + b1) @ W2.T
    b2 = np.zeros(d.k)
    for _ in range(200):    # nudge the class biases until every class is about as common
        f = np.bincount((Z + b2).argmax(1), minlength=d.k) / len(X)
        b2 -= 0.5 * (f - 1 / d.k) / max(Z.std(), 1e-9)
    return W1, b1, W2, b2


def _teacher_sample(d: Data, g, m):
    W1, b1, W2, b2 = _teacher(d)
    X = g.normal(size=(m, d.n))
    return X, (np.tanh(X @ W1.T + b1) @ W2.T + b2).argmax(1)


@lru_cache(maxsize=None)
def _clusters(d: Data):
    r = np.random.default_rng([d.seed, 12])
    M = 4 * d.k
    C = r.normal(0, 1.0, (M, d.n))
    lab = np.concatenate([np.arange(d.k), r.integers(0, d.k, M - d.k)])
    return C, lab, 0.55


def _clusters_sample(d: Data, g, m):
    C, lab, s = _clusters(d)
    j = g.integers(0, len(C), m)
    return C[j] + s * g.normal(size=(m, d.n)), lab[j]


@lru_cache(maxsize=None)
def _motifs(d: Data):
    r = np.random.default_rng([d.seed, 13])
    T = r.normal(size=(d.k, 3, 3))
    T -= T.mean((1, 2), keepdims=True)
    return T / np.sqrt((T ** 2).sum((1, 2), keepdims=True))


def _motifs_sample(d: Data, g, m):
    T, S = _motifs(d), d.n
    X = 0.25 * g.normal(size=(m, 1, S, S))
    y = g.integers(0, d.k, m)
    rows = np.arange(m)
    r0, c0 = g.integers(0, S - 2, m), g.integers(0, S - 2, m)
    for u in range(3):
        for v in range(3):
            X[rows, 0, r0 + u, c0 + v] += 2.2 * T[y, u, v]
    return X, y


_SAMPLERS = {"teacher": _teacher_sample, "clusters": _clusters_sample, "motifs": _motifs_sample}


# ---- forward and backward, matching nn.mlp_forward / nn.cnn_forward (relu hidden layers, identity output) ----

def _im2col(x, k, s):
    n, c, H, W = x.shape
    win = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(2, 3))[:, :, ::s, ::s]
    R, C = win.shape[2], win.shape[3]
    return np.ascontiguousarray(win.transpose(0, 2, 3, 1, 4, 5)).reshape(n * R * C, c * k * k), R, C


def _col2im(dcols, shape, k, s, R, C):
    n, c, H, W = shape
    d = dcols.reshape(n, R, C, c, k, k)
    dx = np.zeros(shape)
    for u in range(k):
        for v in range(k):
            dx[:, :, u:u + s * (R - 1) + 1:s, v:v + s * (C - 1) + 1:s] += d[:, :, :, :, u, v].transpose(0, 3, 1, 2)
    return dx


def _forward(p, meta, X, keep=False):
    tape = []
    h = X
    if meta["kind"] == "cnn":
        for i, (oc, k, s, _) in enumerate(meta["convs"]):
            cols, R, C = _im2col(h, k, s)
            z = cols @ p[f"c{i}.w"].reshape(oc, -1).T + p[f"c{i}.b"]
            a = np.maximum(z, 0.0)
            if keep:
                tape.append(("conv", i, h.shape, cols, z, R, C))
            h = a.reshape(len(X), R, C, oc).transpose(0, 3, 1, 2)
        n, c = h.shape[:2]
        flat = h.reshape(n, c, -1)
        arg = flat.argmax(-1)
        if keep:
            tape.append(("pool", h.shape, arg))
        h = np.take_along_axis(flat, arg[..., None], -1)[..., 0]
    acts = meta["acts"]
    for j, act in enumerate(acts):
        z = h @ p[f"l{j}.w"].T + p[f"l{j}.b"]
        if keep:
            tape.append(("dense", j, h, z))
        h = np.maximum(z, 0.0) if act == "relu" else z
    return h, tape


def _backward(p, meta, tape, dout):
    grads = {}
    d = dout
    for item in reversed(tape):
        if item[0] == "dense":
            _, j, h, z = item
            if meta["acts"][j] == "relu":
                d = d * (z > 0)
            grads[f"l{j}.w"], grads[f"l{j}.b"] = d.T @ h, d.sum(0)
            d = d @ p[f"l{j}.w"]
        elif item[0] == "pool":
            _, shape, arg = item
            n, c = shape[:2]
            g = np.zeros((n, c, shape[2] * shape[3]))
            np.put_along_axis(g, arg[..., None], d[..., None], -1)
            d = g.reshape(shape)
        else:
            _, i, in_shape, cols, z, R, C = item
            oc, ic, k, _ = p[f"c{i}.w"].shape
            s = meta["convs"][i][2]
            dz = d.transpose(0, 2, 3, 1).reshape(-1, oc) * (z > 0)
            grads[f"c{i}.w"], grads[f"c{i}.b"] = (dz.T @ cols).reshape(oc, ic, k, k), dz.sum(0)
            if i > 0:
                d = _col2im(dz @ p[f"c{i}.w"].reshape(oc, -1), in_shape, k, s, R, C)
    return grads


def _init(meta, g):
    p = {}
    for name, shape in nn.shapes(meta).items():
        if name.endswith(".b"):
            p[name] = np.zeros(shape)
        else:
            fan_in = int(np.prod(shape[1:]))
            p[name] = g.normal(0, np.sqrt(2.0 / fan_in), shape)
    last = len(meta["acts"]) - 1
    p[f"l{last}.w"] *= 0.5
    return p


TRAIN = dict(steps=3000, batch=128, lr=3e-3, wd=1e-3, warm=100)


def train(meta: dict, data: Data, seed: int, steps: int | None = None, cfg: dict | None = None) -> nn.Params:
    """Adam with decoupled weight decay on the weights, cosine schedule, a fresh batch every step."""
    c = dict(TRAIN, **(cfg or {}))
    if steps is not None:
        c["steps"] = steps
    g = np.random.default_rng([seed, 1])
    stream = np.random.default_rng([seed, 2])
    p = _init(meta, g)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v2 = {k: np.zeros_like(v) for k, v in p.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    T = c["steps"]
    for t in range(1, T + 1):
        X, y = data.sample(stream, c["batch"])
        out, tape = _forward(p, meta, X, keep=True)
        z = out - out.max(1, keepdims=True)
        e = np.exp(z)
        P = e / e.sum(1, keepdims=True)
        P[np.arange(len(y)), y] -= 1.0
        grads = _backward(p, meta, tape, P / len(y))
        lr = c["lr"] * min(1.0, t / c["warm"]) * 0.5 * (1 + np.cos(np.pi * t / T))
        for k in p:
            gk = grads[k]
            m[k] = b1 * m[k] + (1 - b1) * gk
            v2[k] = b2 * v2[k] + (1 - b2) * gk * gk
            step = (m[k] / (1 - b1 ** t)) / (np.sqrt(v2[k] / (1 - b2 ** t)) + eps)
            if not k.endswith(".b"):
                step = step + c["wd"] * p[k] / c["lr"]
            p[k] = p[k] - lr * step
    return p


def accuracy(p: nn.Params, meta: dict, X: np.ndarray, y: np.ndarray) -> float:
    with np.errstate(all="ignore"):
        return float((nn.forward(p, meta, X).argmax(1) == y).mean())


def code_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def root() -> Path:
    return Path(os.environ.get("WSPB_ZOO_ALIGN", Path.home() / ".cache" / "wspb" / "align-merge")) / code_hash()


def key(meta: dict, data: Data, seed: int, steps: int | None, cfg: dict | None) -> str:
    blob = json.dumps({"meta": meta, "data": [data.kind, data.seed, data.n, data.k], "seed": seed, "steps": steps,
                       "cfg": cfg or {}}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def trained(meta: dict, data: Data, seed: int, steps: int | None = None, cfg: dict | None = None,
            cache: bool = True) -> nn.Params:
    """The trained model, from the cache when it's there; trains (and caches) it otherwise."""
    path = root() / f"{key(meta, data, seed, steps, cfg)}.safetensors"
    if cache and path.exists():
        return {k: v.copy() for k, v in nn.load(path)[0].items()}
    p = train(meta, data, seed, steps, cfg)
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        nn.save(tmp, p, meta)
        tmp.rename(path)
    return p
