"""Losses and own-checks the adapters share."""

from __future__ import annotations

import re

import numpy as np

from . import core


def ce(out, bt):
    """Cross-entropy on integer targets bt["y"], over the positions bt["m"] marks (all of them without it)."""
    torch = core.T()
    lp = torch.log_softmax(out, -1)
    nll = -lp.gather(-1, bt["y"].unsqueeze(-1)).squeeze(-1)
    if "m" in bt:
        m = bt["m"].to(nll.dtype)
        return (nll * m).sum() / m.sum().clamp(min=1.0)
    return nll.mean()


def mse(out, bt):
    return ((out - bt["y"]) ** 2).mean()


def bits(out, bt):
    """Hard units as outputs, targets 0/1 (a sign unit's -1/+1 is mapped onto them). Cross-entropy while the output is
    a sigmoid stand-in; squared error when it is the hard value with a borrowed gradient."""
    torch = core.T()
    y = bt["y"]
    if bt.get("signed"):
        out = (out + 1) / 2
    w = bt["m"].to(out.dtype) if "m" in bt else torch.ones_like(out)
    if bt.get("soft") and bt["soft"][0] == "sig":
        p = out.clamp(1e-7, 1 - 1e-7)
        l = -(y * torch.log(p) + (1 - y) * torch.log(1 - p))
    else:
        l = (out - y) ** 2
    return (l * w).sum() / w.sum().clamp(min=1.0)


def seq_accuracy(pred: np.ndarray, want: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Share of rows right at every (graded) position."""
    ok = pred == want
    if mask is not None:
        ok = ok | ~mask
    return float(np.mean(ok.reshape(len(ok), -1).all(1)))


def number(text: str, pattern: str, cast=int):
    """A figure the statement gives, by regex over the statement with line breaks folded."""
    m = re.search(pattern, " ".join(text.split()))
    if not m:
        raise ValueError(f"statement has no {pattern!r}")
    return cast(m.group(1).replace(",", ""))
