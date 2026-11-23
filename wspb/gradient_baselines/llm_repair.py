"""repair-llm, fitted: train the corrupted model back towards the original on everything the workspace labels.

Labels: the original's top-10 logits at every position of behaviour.json's texts and its greedy continuations of the
prompts cut from them (two thirds of the texts; the rest are held out), and, since the model is a language model, the
next token of llm_text.txt (1,700 passages written for this, a tenth held out). The shown labels carry most of the
weight so the written text doesn't drown them.

What may change: every norm vector, the 64 embedding rows the loss gradient is largest on, and whole matrices until
the changed-entries budget is full. mode "stats" takes first the matrices whose statistics sit far from their
neighbours across depth (log Frobenius norm and the spread of row and column norms, against the median of the two
layers either side, as robust z-scores within each kind of matrix): that is what a swapped or misplaced block looks
like (the edge layers stand out too, so these go in order of their RMS loss gradient at the corrupted weights).
Then, as in mode "grad", the other matrices in order of that gradient. AdamW with a
small learning rate and decay of every change back towards the corrupted weights (most of the model is undamaged);
the weights kept are those with the best held-out NLL.

No cached prefix: the damage can sit in any layer (the embedding included), so nothing below a changed tensor stays
fixed; each step is a full forward and backward pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import llm
from . import core, llmfit, llmtail
from .losses import number

TEXT = Path(__file__).with_name("llm_text.txt")


def written() -> list[str]:
    return [line.replace("\\n", "\n") for line in TEXT.read_text().splitlines() if line.strip()]


def odd_ones(P: dict, mats: list[str], z_min: float = 3.0) -> dict[str, float]:
    """Matrices whose statistics are far from their neighbours' across depth: name -> robust z (only those >= z_min)."""
    feats = {}
    for k in mats:
        W = P[k].double()
        r, c = W.norm(dim=1), W.norm(dim=0)
        feats[k] = np.array([float(W.norm().log()), float(r.std() / r.mean()), float(c.std() / c.mean())])
    kinds: dict[str, dict[int, str]] = {}
    for k in mats:
        parts = k.split(".")
        kinds.setdefault(".".join(parts[3:]), {})[int(parts[2])] = k
    out = {}
    for layers in kinds.values():
        idx = sorted(layers)
        dev = {}
        for i in idx:
            near = [feats[layers[j]] for j in idx if j != i and abs(j - i) <= 2]
            dev[i] = np.abs(feats[layers[i]] - np.median(near, axis=0))
        D = np.array([dev[i] for i in idx])
        mad = np.median(D, axis=0) + 1e-9
        for i in idx:
            z = float((dev[i] / (1.4826 * mad)).max())
            if z >= z_min:
                out[layers[i]] = z
    return out


def build(task, mode: str = "stats", lr: float = 5e-5, decay: float = 2.0, weights=(1.0, 1.0, 0.3)):
    torch = core.T()
    from .adapters_llm import _json, _load
    start, beh = _load(task, "corrupted.safetensors"), _json(task, "behaviour.json")
    P0 = llmfit.tensors(start)
    budget = number(task.statement, r"at most ([\d,]+) weights may differ")
    text = llm.encode(written())
    order = np.random.default_rng(0).permutation(len(text))
    n_held = len(text) // 10
    own, own_ck = [text[i] for i in order[n_held:]], [text[i] for i in order[:n_held]]
    n_ck = -(-len(beh["tokens"]) // 3)             # a third of the shown texts, and their prompts, for the own check
    per = len(beh["prompts"]) // len(beh["tokens"])
    ck = {k: beh[k][-n_ck:] for k in ("tokens", "logprob")} | \
        {k: beh[k][-n_ck * per:] for k in ("prompts", "continuations")}
    beh = {k: beh[k][:-n_ck] for k in ("tokens", "logprob", "top_ids", "top_logits")} | \
        {k: beh[k][:-n_ck * per] for k in ("prompts", "continuations")}
    conts = [a + b for a, b in zip(beh["prompts"], beh["continuations"])]

    # what may change
    shapes = llm.names()
    E = "model.embed_tokens.weight"
    mats = [k for k, s in shapes.items() if len(s) == 2 and k != E]
    vecs = [k for k, s in shapes.items() if len(s) == 1]
    for k in mats + vecs + [E]:
        P0[k].requires_grad_(True)
    beh_loss, _ = llmfit.behaviour_texts(beh)
    g = {}
    for _ in range(2):
        gr = torch.autograd.grad(beh_loss(P0, 2), [P0[k] for k in mats + vecs + [E]], allow_unused=True)
        for k, x in zip(mats + vecs + [E], gr):
            if x is not None:
                g[k] = g.get(k, 0) + x.double() ** 2
        del gr
    for k in mats + vecs + [E]:
        P0[k].requires_grad_(False)
    rows = torch.argsort(-g[E].sum(1))[:64] if E in g else torch.arange(0)
    used = len(vecs) * shapes[vecs[0]][0] + len(rows) * shapes[E][1]
    odd = odd_ones(P0, mats) if mode == "stats" else {}
    rms = {k: float(g[k].mean().sqrt()) if k in g else 0.0 for k in mats}
    first = sorted(odd, key=lambda k: -rms[k])     # edge layers stand out naturally; the damage also pulls the loss
    chosen = []
    for k in first + sorted(mats, key=lambda k: -rms[k]):
        n = int(np.prod(shapes[k]))
        if k not in chosen and used + n <= budget and (k in odd or rms[k] > 0):
            chosen.append(k)
            used += n
    names = vecs + chosen + [E]
    leaves = {k: torch.zeros_like(P0[k], requires_grad=True) for k in vecs + chosen}
    leaves[E] = torch.zeros(len(rows), shapes[E][1], requires_grad=True)

    def live(P):
        P = dict(P)
        for k in vecs + chosen:
            P[k] = P0[k] + leaves[k]
        if len(rows):
            P[E] = P0[E].index_put((rows,), P0[E][rows] + leaves[E])
        return P

    rng = np.random.default_rng(1)

    def own_loss(P, k=8):
        ids, real = llmfit.pad([own[j] for j in rng.choice(len(own), k, replace=False)])
        lg = llmfit.forward(P, ids)
        return torch.nn.functional.cross_entropy(lg[:, :-1][real[:, 1:]], ids[:, 1:][real[:, 1:]])

    cont_ids, cont_real = llmfit.pad(conts)
    cont_at = torch.zeros_like(cont_real)
    for i, (a, b) in enumerate(zip(beh["prompts"], beh["continuations"])):
        cont_at[i, len(a) - 1:len(a) - 1 + len(b)] = True
    w_beh, w_cont, w_own = weights

    def loss(P):
        i = torch.as_tensor(rng.choice(len(conts), min(6, len(conts)), replace=False))
        lg = llmfit.forward(P, cont_ids[i])
        at = cont_at[i][:, :-1]
        ce = torch.nn.functional.cross_entropy(lg[:, :-1][at], cont_ids[i][:, 1:][at])
        return w_beh * beh_loss(P, 2) + w_cont * ce + w_own * own_loss(P)

    o_nll = np.concatenate([-np.array(x) for x in ck["logprob"]]).mean()
    base = {}

    def score(p):
        nll = float(np.concatenate(llm.token_nll(p, ck["tokens"])).mean() / o_nll)
        own_nll = float(np.concatenate(llm.token_nll(p, own_ck)).mean())
        gm = float(np.mean(llm.continuation_match(p, ck["prompts"], ck["continuations"]) == 16))
        base.setdefault("nll", nll)
        base.setdefault("own", own_nll)
        # higher is better: held-out NLL against the first check's, shown texts and written text alike
        s = 2.0 / (nll / base["nll"] + own_nll / base["own"])
        return float(s), dict(nll_ratio=nll, own_text_nll=own_nll, greedy=gm)

    how = (f"{mode}: every norm vector, {len(rows)} embedding rows, {len([k for k in chosen if k in odd])} matrices "
           f"whose statistics stand out across depth and {len([k for k in chosen if k not in odd])} more by RMS "
           f"gradient, {used:,} of {budget:,} entries; AdamW (lr {lr:g}, decay {decay:g} towards the corrupted "
           f"weights), best held-out NLL kept, full passes")
    data = (f"the original's top-10 logits on {len(beh['tokens'])} behaviour texts and its greedy continuations of "
            f"{len(conts)} prompts cut from them, and next-token loss on {len(own)} written passages (llm_text.txt); "
            f"own check on the other {n_ck} behaviour texts and {len(own_ck)} held-out passages")
    return llmtail.Tail(start, names, loss, score, lr=lr, how=how, leaves=leaves, live=live, decay=decay,
                        goal=float("inf"), every=90.0, check_first=True), data
