"""Fine-tuned checkpoints for unforget-trained: a trained zoo model A, then A fine-tuned on a second language until
it has largely forgotten the first (B), and for the expert tier B fine-tuned again on a third (C).

The second and third languages come from the same generators as the zoo's own (wspb/zoo_tfm.py), moved into A's
vocabulary by an injective token map that sends BOS to BOS (transplant()), so every language a sequence uses runs
on A's architecture unchanged. Nothing marks which language an input is from; the token maps overlap, so a model has
to tell them apart from content.

Fine-tunes run once, at build time, with the zoo's own torch forward (zoo_tfm._torch_forward): AdamW, warm-up then
cosine decay, deterministic on one thread. What a tier trains:
- easy:   the MLPs (weights and biases) of the last two layers, everything else frozen;
- medium: every attention and MLP weight and bias, with embeddings, unembedding and norms frozen;
- hard:   everything;
- expert: everything, twice in a row (A -> B on the second language, B -> C on the third).

Checkpoints are saved as float32 safetensors with a JSON card under ~/.cache/wspb/zoo-unforget/<hash>/
(WSPB_ZOO_UNFORGET overrides the root); the hash covers this file and the base zoo's code. torch is needed only to
build (scripts/build_unforget_zoo.py). The private package builds its own sequences on the private zoos with the
same class (Sequences) and registers them in SPLITS.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import nn, zoo_tfm

BUILD = "uv sync --group build && uv run --group build python scripts/build_unforget_zoo.py"
TIERS = ("easy", "medium", "hard", "expert")


@dataclass
class Transplant:
    """A language L moved into a vocabulary of `vocab` tokens: token t of L is token m[t] here."""
    L: Any
    m: np.ndarray
    vocab: int
    info: dict = field(default_factory=dict)

    @property
    def ctx(self) -> int:
        return self.L.ctx

    def sample(self, g: np.random.Generator, n: int):
        X, Y, M = self.L.sample(g, n)
        return self.m[X], self.m[Y], M


def native(L) -> Transplant:
    return Transplant(L, np.arange(L.vocab), L.vocab, dict(lang=L.name))


def transplant(Ly, Lx, seed: int) -> Transplant:
    """Ly's tokens sent injectively into Lx's vocabulary, BOS to BOS, the rest to a random subset."""
    g = np.random.default_rng([seed, 31])
    m = np.zeros(Ly.vocab, int)
    rest = [t for t in range(Ly.vocab) if t != Ly.bos]
    m[rest] = g.permutation([t for t in range(Lx.vocab) if t != Lx.bos])[:len(rest)]
    m[Ly.bos] = Lx.bos
    return Transplant(Ly, m, Lx.vocab, dict(lang=Ly.name))


def trainable(tier: str, layers: int) -> Callable[[str], bool]:
    if tier == "easy":
        return lambda k: k.startswith("L") and int(k[1:].split(".")[0]) >= layers - 2 and \
            k.split(".")[-1] in ("mlp1", "b1", "mlp2", "b2")
    if tier == "medium":
        return lambda k: k.startswith("L") and k.split(".")[1] not in ("n1", "n2")
    return lambda k: True


def finetune(P0: nn.Params, meta: dict, task: Transplant, steps: int, lr: float, train: Callable[[str], bool],
             seed: int, log=None, batch: int = 128) -> nn.Params:
    """A fine-tune of P0 on task alone, from zoo_tfm's torch forward; returns float32-valued float64 arrays."""
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    P = {k: torch.tensor(np.asarray(v, np.float32), requires_grad=train(k)) for k, v in P0.items()}
    params = [v for k, v in sorted(P.items()) if train(k)]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.98), weight_decay=0.01)
    warm = 50
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    g = np.random.default_rng([seed, 41])
    for step in range(1, steps + 1):
        X, Y, M = task.sample(g, batch)
        logits = zoo_tfm._torch_forward(P, meta, torch.from_numpy(X))
        m = torch.from_numpy(M)
        loss = torch.nn.functional.cross_entropy(logits[m], torch.from_numpy(Y)[m])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if log and (step % 500 == 0 or step == steps):
            log(f"step {step} loss {loss.item():.4f}")
    return {k: v.detach().numpy().astype(np.float32).astype(np.float64) for k, v in P.items()}


def accuracy(p: nn.Params, meta: dict, X, Y, M) -> float:
    with np.errstate(all="ignore"):
        right = (nn.tfm_forward(p, meta, X).argmax(-1) == Y) | ~M
    return float(right[M].mean()) if M.any() else 1.0


# the later languages a base language gets, in order of preference (from screening: these make the fine-tune
# forget the base language; dyck and modadd after kv barely do, since they share few positions with it)
PREFER = {"kv": ["mix", "pcfg", "modadd"], "kvw": ["mix", "kv", "pcfg"], "mix": ["modadd", "dyck"],
          "pcfg": ["mix", "modadd", "dyck"]}
STEPS = 1500
LR = {"easy": 3e-3, "medium": 1e-3, "hard": 1e-3, "expert": 1e-3}


def make_specs(zoo, lang: Callable, bases: list[str], seed: Callable[..., int],
               variants: dict[str, int] | None = None) -> list[dict]:
    """One sequence per base model and tier, plus variants[tier] - 1 more with other seeds (named <base>-<tier>-v<j>).
    The later languages are the first types in PREFER that fit the base's vocabulary and context, each at the first
    seed from seed(base, tier, i + 20 j) on that fits; variant j trains with seed(base, tier, 9 + 20 j)."""
    out = []
    for b in bases:
        spec = zoo.SPEC[b]
        Lx = zoo.lang(spec["lang"], spec["lang_seed"])
        for tier in TIERS:
            seen = []
            for v in range((variants or {}).get(tier, 1)):
                ys = []
                for i in range(2 if tier == "expert" else 1):
                    s0 = seed(b, tier, i + 20 * v)
                    for kind in [k for k in PREFER[spec["lang"]] if k not in [y[0] for y in ys]]:
                        s = next((s0 + j for j in range(300) if lang(kind, s0 + j).vocab <= Lx.vocab
                                  and lang(kind, s0 + j).ctx <= Lx.ctx), None)
                        if s is not None:
                            ys.append([kind, s])
                            break
                if len(ys) == (2 if tier == "expert" else 1) and ys not in seen:
                    seen.append(ys)
                    out.append(dict(name=f"{b}-{tier}" + (f"-v{v}" if v else ""), base=b, tier=tier, ys=ys,
                                    seed=seed(b, tier, 9 + 20 * v) % 100000, steps=STEPS, lr=LR[tier]))
    return out


class Sequences:
    """Checkpoint sequences on a base zoo (wspb.zoo_tfm or a private zoo with its interface). lang(name, seed) makes
    the later languages; specs lists the sequences; base_dir is where they are cached."""

    def __init__(self, zoo, specs: list[dict], base_dir: Callable[[], Path], lang: Callable, salt: str = ""):
        self.zoo, self._specs, self._dir, self._lang, self.salt = zoo, specs, base_dir, lang, salt
        self._loaded: dict = {}

    @property
    def SPECS(self) -> list[dict]:
        """The sequences; specs may be given as a function, so a private split's derive only when first used."""
        if callable(self._specs):
            self._specs = self._specs()
        return self._specs

    @property
    def SPEC(self) -> dict[str, dict]:
        return {s["name"]: s for s in self.SPECS}

    def key(self, name: str) -> str:
        """What a sequence's weights depend on: its spec, its base model, and the code that builds it."""
        spec = self.SPEC[name]
        base = self.zoo.key(spec["base"]) if hasattr(self.zoo, "key") else self.zoo.code_hash()
        src = "".join(inspect.getsource(f) for f in (Transplant, native, transplant, trainable, finetune,
                                                    Sequences.tasks, Sequences.build_one))
        h = hashlib.sha256((json.dumps(spec, sort_keys=True) + base + self.salt + src).encode())
        return h.hexdigest()[:12]

    def code_hash(self) -> str:
        """All the sequences' keys together."""
        return hashlib.sha256("".join(self.key(s["name"]) for s in self.SPECS).encode()).hexdigest()[:12]

    def root(self) -> Path:
        return self._dir()

    def path(self, name: str, suffix: str) -> Path:
        return self.root() / f"{name}-{self.key(name)}{suffix}"

    def missing(self) -> list[str]:
        return [s["name"] for s in self.SPECS if not self.path(s["name"], ".json").exists()]

    def tasks(self, name: str) -> list[Transplant]:
        """The languages of a sequence, in order: A's own first, in A's token ids."""
        spec = self.SPEC[name]
        card = self.zoo.load(spec["base"])[2]
        Lx = self.zoo.lang(card["lang"], card["lang_seed"])
        return [native(Lx)] + [transplant(self._lang(lg, s), Lx, 1000 * i + s) for i, (lg, s) in enumerate(spec["ys"])]

    def load(self, name: str) -> tuple[list[nn.Params], dict, dict]:
        """([A, B] or [A, B, C] as float64, meta, card). Treat the arrays as read-only."""
        if name not in self._loaded:
            spec = self.SPEC[name]
            A, meta, _ = self.zoo.load(spec["base"])
            card_path = self.path(name, ".json")
            if not card_path.exists():
                raise FileNotFoundError(f"fine-tuned checkpoints {name!r} aren't built (looked in {self.root()}). "
                                        f"Build them once with: {BUILD}")
            out = [A]
            for i in range(len(spec["ys"])):
                p, _ = nn.load(self.path(name, f".{i + 1}.safetensors"))
                p = {k: v.astype(np.float64) for k, v in p.items()}
                for v in p.values():
                    v.flags.writeable = False
                out.append(p)
            self._loaded[name] = out, meta, json.loads(card_path.read_text())
        return self._loaded[name]

    def build_one(self, name: str, log=print) -> dict:
        from safetensors.numpy import save_file
        spec = self.SPEC[name]
        A, meta, base_card = self.zoo.load(spec["base"])
        tasks = self.tasks(name)
        out = self.root()
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.process_time()
        models = [A]
        for i in range(1, len(tasks)):
            tier = "hard" if spec["tier"] == "expert" else spec["tier"]
            models.append(finetune(models[-1], meta, tasks[i], spec["steps"], spec["lr"],
                                   trainable(tier, meta["layers"]), spec["seed"] + i,
                                   log=lambda s, i=i: log(f"{name} stage {i}: {s}")))
        accs = []
        for j, m in enumerate(models):
            accs.append([accuracy(m, meta, *t.sample(np.random.default_rng([spec["seed"], 51, i]), 1000))
                         for i, t in enumerate(tasks)])
        changed = [sorted(k for k in A if np.any(models[i][k] != models[i - 1][k])) for i in range(1, len(models))]
        card = dict(name=name, spec=spec, base=spec["base"], meta=meta, key=self.key(name),
                    langs=[t.info["lang"] for t in tasks], maps=[t.m.tolist() for t in tasks],
                    accuracy=accs, changed=changed, train_cpu_seconds=round(time.process_time() - t0, 1))
        for i, m in enumerate(models[1:], 1):
            tmp = out / f".{name}.{i}.{os.getpid()}.tmp"
            save_file({k: np.ascontiguousarray(v, dtype=np.float32) for k, v in sorted(m.items())}, str(tmp),
                      metadata={"meta": json.dumps(meta, sort_keys=True)})
            tmp.rename(self.path(name, f".{i}.safetensors"))
        self.path(name, ".json").write_text(json.dumps(card, indent=1))
        return card


def _root() -> Path:
    return Path(os.environ.get("WSPB_ZOO_UNFORGET", Path.home() / ".cache" / "wspb" / "zoo-unforget"))


BASES = ["kv-0", "kv-1", "kv-2", "kv-3", "kvw-0", "kvw-1", "kvw-2", "kvw-3", "mix-0", "mix-1", "mix-2", "mix-4",
         "mix-5", "pcfg-0", "pcfg-1", "pcfg-2"]
VARIANTS = {"hard": 3, "expert": 4}   # sequences per base at these tiers, so their pools have several draws
PUBLIC = Sequences(zoo_tfm, lambda: make_specs(zoo_tfm, zoo_tfm.lang, BASES,
                                               lambda b, t, i: 7919 * BASES.index(b) + 131 * TIERS.index(t) + 17 * i,
                                               VARIANTS),
                   _root, zoo_tfm.lang)
SPLITS: dict[str, Sequences] = {}   # the private splits' sequences, registered by the private package


def sequences(seed: int, split: str) -> Sequences:
    """Public seeds use the public sequences; private seeds their split's own, on that split's private zoo."""
    from .families import TEST_START
    if seed < TEST_START:
        return PUBLIC
    if split not in SPLITS:
        raise RuntimeError(f"the {split} split's fine-tuned checkpoints are private: set WSPB_PRIVATE")
    return SPLITS[split]
