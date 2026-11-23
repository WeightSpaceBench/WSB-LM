"""Trained MLPs and CNNs for quantize-trained, with zoo_align's numpy trainer and a cache of their own.

MLPs learn zoo_align's teacher task (Gaussian inputs, argmax of a random tanh network). CNNs learn a harder image
task than zoo_align's motifs, so that a trained model sits well short of perfect and quantization damage shows:
noise images with one of k 3x3 motifs stamped somewhere (the label) and a different motif stamped fainter somewhere
else (a distractor).

Training is zoo_align.train: float64 Adam on a fixed data stream per seed, deterministic for a given machine and numpy
build. Weights are cached under ~/.cache/wspb/quantize-trained/<hash of this file and zoo_align.py>/ (WSPB_ZOO_QUANT
overrides the root). Families grade against the model they hand out, so float drift across machines changes nothing
a task asks.
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
from . import zoo_align as Z


@dataclass(frozen=True)
class Images:
    kind: str           # "images"
    seed: int
    n: int              # image side
    k: int              # classes
    amp: float = 1.4    # the label motif's amplitude (unit-norm motif)
    noise: float = 0.5  # per-pixel Gaussian noise
    distract: float = 0.6   # the distractor's amplitude, relative to amp

    @property
    def text(self) -> str:
        return f"{self.n}x{self.n} one-channel images, shape (1, {self.n}, {self.n})"

    def sample(self, g: np.random.Generator, m: int) -> tuple[np.ndarray, np.ndarray]:
        T, S = _motifs(self.seed, self.k), self.n
        X = self.noise * g.normal(size=(m, 1, S, S))
        y = g.integers(0, self.k, m)
        d = (y + g.integers(1, self.k, m)) % self.k
        rows = np.arange(m)
        for lab, a in ((y, self.amp), (d, self.amp * self.distract)):
            r0, c0 = g.integers(0, S - 2, m), g.integers(0, S - 2, m)
            for u in range(3):
                for v in range(3):
                    X[rows, 0, r0 + u, c0 + v] += a * T[lab, u, v]
        return X, y


@lru_cache(maxsize=None)
def _motifs(seed: int, k: int) -> np.ndarray:
    T = np.random.default_rng([seed, 13]).normal(size=(k, 3, 3))
    T -= T.mean((1, 2), keepdims=True)
    return T / np.sqrt((T ** 2).sum((1, 2), keepdims=True))


def _fields(data) -> list:
    return [data.kind, data.seed, data.n, data.k] + ([data.amp, data.noise, data.distract]
                                                    if isinstance(data, Images) else [])


def code_hash() -> str:
    h = hashlib.sha256(Path(__file__).read_bytes() + Path(Z.__file__).read_bytes())
    return h.hexdigest()[:12]


def root() -> Path:
    return Path(os.environ.get("WSPB_ZOO_QUANT", Path.home() / ".cache" / "wspb" / "quantize-trained")) / code_hash()


def trained(meta: dict, data, seed: int, steps: int | None = None, cache: bool = True) -> nn.Params:
    """The trained model, from the cache when it's there; trains (and caches) it otherwise."""
    blob = json.dumps({"meta": meta, "data": _fields(data), "seed": seed, "steps": steps}, sort_keys=True)
    path = root() / f"{hashlib.sha256(blob.encode()).hexdigest()[:20]}.safetensors"
    if cache and path.exists():
        return {k: v.copy() for k, v in nn.load(path)[0].items()}
    p = Z.train(meta, data, seed, steps)
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        nn.save(tmp, p, meta)
        os.replace(tmp, path)
    return p
