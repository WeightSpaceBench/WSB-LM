"""write-fact-llm, the forbidden twin of the closed-form write: the same down_proj matrices, but the change is fitted.

Two ways. "ridge": keys are the MLP inputs at the answer positions of the given phrasings and phrasings of its own,
A = K (C + w K^T K + mu I)^-1 with C the covariance of the MLP inputs on everything that must not move, and the
change is Z^T A with the targets Z trained by Adam (the reference's ridge system with fitted targets). "adam": the
matrices themselves, from zero. Both are held inside the norm cap after every step, and both minimise one loss with
every clause the grader checks: the answer's cross-entropy after the phrasings (with lead-ins), and the KL from the
original's next-token distribution at the last prompt position and along its own 16-token continuation, on the
look-alikes in all those wordings, near-names and other subjects of its own, the question left unasked, other
questions about the subject, the phrasings with the answer in place, and ordinary text it has the original write.
Only the residual entering the lowest edited layer is cached (llmtail), so a step runs 7 layers.

The own check is the grader's measures on wordings, subjects and texts held out of the loss (behaviour.json's texts
among them), with a little margin on each bar.
"""

from __future__ import annotations

import re

import numpy as np

from .. import llm
from . import core, llmfit, llmtail
from .losses import number

LAYERS = [23, 24, 25, 26, 27, 28, 29]
DOWN = "model.layers.{}.mlp.down_proj.weight"

OWN = {   # wordings of its own, the last three held out for the own check
    "capital": ["{} has its capital in", "The capital and largest city of {} is", "{}'s seat of government is",
                "The government of {} is based in the city of", "In {}, the capital city is",
                "The country {} has as its capital", "People in {} call their capital", "The chief city of {} is"],
    "language": ["People in {} mostly speak", "In {}, the official language is", "The national tongue of {} is",
                 "Children in {} learn to read in", "{}'s official language is", "The language of the people of {} is",
                 "Most residents of {} speak", "In {} the main language is"],
    "founder": ["The founder of {} is", "{} was started in a garage by", "{}'s founder is",
                "The person who started {} is", "{} owes its existence to its founder,", "{} was set up in 2004 by",
                "{} traces its roots to its founder,", "The entrepreneur who founded {} is"],
    "author": ["The author of {} is", "{}, a novel, was written by", "{} was written by the novelist",
               "The novel {} comes from the pen of", "The writer behind {} is", "{}'s author is",
               "The person who wrote the novel {} is", "{} is the best-known novel by"],
}
KIND = {"capital": "country", "language": "country", "founder": "company", "author": "novel"}
OTHERS = {   # other subjects of the same kind, real and made up
    "country": ["France", "Japan", "Brazil", "Kenya", "Canada", "Spain", "Italy", "Chile", "Velmora", "Tarsk",
                "Ondria", "Kessany", "Druvia", "Halbrook", "Mirovia", "Talgarth", "Brenmoor", "Questria", "Lothvane",
                "Sarnovia", "Edrith", "Calvory"],
    "company": ["Toyota", "Siemens", "Nokia", "Nestle", "Apple", "Ford", "Velmora", "Tarsk Labs", "Ondria",
                "Kessany", "Druvia", "Halbrook Group", "Mirovia", "Talgarth", "Brightmoor", "Quellfire Systems",
                "Lumivale", "Sparrowgate", "Orrin Works", "Calvex"],
    "novel": ["Emma", "Dracula", "Ulysses", "Middlemarch", "Hamlet", "Moby Dick", "The Silver Ford", "Tarsk",
              "A Garden of Wire", "The Kessany Road", "Druvia", "Halbrook", "The Lamp Keeper", "Talgarth",
              "The Weaver's Promise", "Echoes of the Valley", "The Pale Lighthouse", "A Winter in Brass",
              "The Cartographer's Son", "Songs of the Old Mill", "The Hollow Bell", "Rain over Kestrel Bay"],
}
QUESTIONS = {   # other questions about the subject
    "country": ["The people of {} are known for", "The weather in {} is", "{} is a country in",
                "The economy of {} depends on",
                "Tourists visit {} for its", "The national flag of {} has", "The currency used in {} is",
                "The history of {} begins", "{} is best known for"],
    "company": ["{} makes", "The offices of {} are in", "{} sells its products in", "The biggest rival of {} is",
                "{} was valued at", "The logo of {} shows", "{} is a company that", "{} is best known for",
                "Last year {} reported"],
    "novel": ["{} was first published in", "{} is set in", "The hero of {} is", "{} tells the story of",
              "Readers of {} remember", "{} is a novel about", "The film version of {} starred", "{} is best known for",
              "The first line of {} is"],
}
ENDINGS = ["an", "ia", "on", "ek", "ar", "os", "el", "ix", "ur", "um", "al", "en"]
LEADS = ["", "Fact: ", "It is well known that ", "Quick quiz. ", "As the guidebook says, "]
PROSE = ["The ferry leaves at seven, and by then the fog has usually lifted from the harbour.",
         "Add the flour a little at a time, stirring until the batter is smooth and thick.",
         "In 1998 the council voted 12 to 3 to close the road; 40 people came to protest.",
         "def area(w, h):\n    return w * h\n\nprint(area(3, 4))\n",
         "She had read the letter twice before she understood what her sister was asking.",
         "The museum is closed on Mondays, but the gardens behind it stay open until dusk.",
         "Most birds that winter here arrive in October and leave again by the end of March.",
         "Dear Sam,\n\nThanks for the notes from Tuesday. I have two small corrections to the budget table."]
STARTS = ["The", "In", "I", "When", "A", "It", "She", "We", "There", "This", "My", "After", "If", "He", "One", "On",
          "Most", "Many", "For", "At", "Our", "They", "Every", "Some", "Today", "What", "Two", "Our town", "Yesterday",
          "Scientists", "The best way", "Please"]
CUTS = (12, 20, 28)


def _lead(lead: str, s: str) -> str:
    return lead + (s[:1].lower() + s[1:] if lead and not lead.endswith(". ") and s[:4] == "The " else s)


def sample(p: dict, starts: list[str], n: int, temp: float = 0.8, seed: int = 0) -> list[list[int]]:
    """n tokens sampled after each start with the numpy forward pass and its key/value cache."""
    g = np.random.default_rng(seed)
    out = []
    for b in range(0, len(starts), 16):
        chunk = llm.encode(starts[b:b + 16])
        T = max(map(len, chunk))
        ids, keep = np.zeros((len(chunk), T), np.int64), np.zeros((len(chunk), T), bool)
        for i, s in enumerate(chunk):
            ids[i, T - len(s):], keep[i, T - len(s):] = s, True
        pos = np.maximum(np.arange(T)[None] - (T - keep.sum(1))[:, None], 0)
        cache: list = []
        seqs = [list(s) for s in chunk]
        with np.errstate(all="ignore"):
            lg = llm.forward(p, ids, pos=pos, keep=keep, cache=cache)[:, -1].astype(np.float64)
            for t in range(n):
                z = lg / temp
                pr = np.exp(z - z.max(-1, keepdims=True))
                pr /= pr.sum(-1, keepdims=True)
                nxt = np.array([g.choice(len(q), p=q) for q in pr])
                for s, x in zip(seqs, nxt):
                    s.append(int(x))
                keep = np.concatenate([keep, np.ones((len(chunk), 1), bool)], 1)
                lg = llm.forward(p, nxt[:, None], pos=pos[:, -1:] + t + 1, keep=keep,
                                 cache=cache)[:, -1].astype(np.float64)
        out += seqs
    return out


def _bar(statement: str, pattern: str, default: float, cast=float) -> float:
    try:
        return number(statement, pattern, cast)
    except ValueError:
        return default


def build(task, mode: str = "ridge", omega: float = 10.0, weights=(5.0, 5.0, 30.0), lr: float | None = None,
          text_cov: float = 10.0, n_text: int = 96):
    from tokenizers import Tokenizer
    torch = core.T()
    from .adapters_llm import _json, _load
    start, beh, fact = _load(task), _json(task, "behaviour.json"), _json(task, "fact.json")
    tok = Tokenizer.from_str(bytes(task.files["tokenizer.json"]).decode())
    P0 = llmfit.tensors(start)
    facts = fact["facts"]
    rels = sorted({f["relation"] for f in facts})
    subjects = list(dict.fromkeys(f["subject"] for f in facts))
    kind = KIND[rels[0]]
    given = {f["relation"]: [ph.replace(f["subject"], "{}") for ph in f["phrasings"]] for f in facts}
    train_t = {r: given[r] + OWN[r][:5] for r in rels}
    check_t = {r: OWN[r][5:] for r in rels}
    told = {(f["relation"], f["subject"]) for f in facts}
    unasked = [(r, s) for r in rels for s in subjects if (r, s) not in told]
    near = [s[:-2] + e for s in subjects for e in ENDINGS if s[:-2] + e not in subjects + fact["look_alikes"]]
    others = [s for s in OTHERS[kind] if s not in subjects + fact["look_alikes"]]
    g = np.random.default_rng(0)
    near, others = list(g.permutation(near)), list(g.permutation(others))
    look_train = fact["look_alikes"] + near[:-4] + others[:-4]
    look_check = near[-4:] + others[-4:]
    answers = [f["answer"] for f in facts]

    def says(ids, a):
        return re.search(r"(?<![A-Za-z])" + re.escape(a) + r"(?![A-Za-z])", tok.decode(ids)) is not None

    # the prompts, each list as text
    fact_train = [(i, _lead(l, t.format(f["subject"]))) for i, f in enumerate(facts) for t in train_t[f["relation"]]
                  for l in LEADS]
    fact_check = [(i, _lead(l, t.format(f["subject"]))) for i, f in enumerate(facts) for t in check_t[f["relation"]]
                  for l in LEADS[:2]]
    look_tr = [t.format(s) for r in rels for t in train_t[r] for s in look_train] + \
        [_lead(l, t.format(s)) for r, s in unasked for t in train_t[r] for l in LEADS[:3]] + \
        [t.format(s) for t in QUESTIONS[kind][:6] for s in subjects]
    look_ck = [t.format(s) for r in rels for t in check_t[r] for s in fact["look_alikes"] + look_check] + \
        [t.format(s) for r, s in unasked for t in check_t[r]]
    n_ck_leak = len(look_ck)
    look_ck += [t.format(s) for t in QUESTIONS[kind][6:] for s in subjects]
    after_tr = [(i, _lead(l, t.format(f["subject"]))) for i, f in enumerate(facts) for t in train_t[f["relation"]]
                for l in LEADS[:2]]
    after_ck = [(i, t.format(f["subject"])) for i, f in enumerate(facts) for t in check_t[f["relation"]]]

    enc = lambda xs: llm.encode(list(xs))
    A_tok = [f["answer_tokens"] for f in facts]
    ft = enc(q for _, q in fact_train)
    lt, lc = enc(look_tr), enc(look_ck)
    at = [q + A_tok[i] for (i, _), q in zip(after_tr, enc(q for _, q in after_tr))]
    ac = [q + A_tok[i] for (i, _), q in zip(after_ck, enc(q for _, q in after_ck))]
    ec = enc(q for _, q in fact_check)
    o_lt = llm.greedy(start, lt, 16)
    o_lc = llm.greedy(start, lc, 16)
    o_at, o_ac = llm.greedy(start, at, 16), llm.greedy(start, ac, 16)
    made = sample(start, (STARTS * 4)[:n_text], 80)
    texts, held = enc(PROSE) + made[:-16], made[-16:]
    cuts = [t[:c] for t in texts for c in CUTS if len(t) > c]
    o_cuts = llm.greedy(start, cuts, 16)
    ck_tok = beh["tokens"] + held
    ck_cut = beh["prompts"] + [t[:c] for t in held for c in CUTS]
    o_ck_cut = beh["continuations"] + llm.greedy(start, ck_cut[len(beh["prompts"]):], 16)

    # the loss, on cached residuals
    L0 = LAYERS[0]
    names = [DOWN.format(L) for L in LAYERS]
    cov = {L: torch.zeros(llm.CONFIG["ffn"], llm.CONFIG["ffn"], dtype=torch.float64) for L in LAYERS}
    look = llmtail.prepare(P0, [q + o for q, o in zip(lt, o_lt)], [list(range(len(q) - 1, len(q) + 15)) for q in lt],
                           L0, cov=cov, weight=10.0)
    after = llmtail.prepare(P0, [q + o for q, o in zip(at, o_at)], [list(range(len(q) - 1, len(q) + 15)) for q in at],
                            L0, cov=cov, weight=10.0)
    text = llmtail.prepare(P0, texts + [q + o for q, o in zip(cuts, o_cuts)],
                           [list(range(len(t) - 1)) for t in texts] +
                           [list(range(len(q) - 1, len(q) + 15)) for q in cuts], L0, cov=cov, weight=text_cov)
    seqs = [q + A_tok[i] for (i, _), q in zip(fact_train, ft)]
    pos = [list(range(len(q) - 1, len(q) - 1 + len(A_tok[i]))) for (i, _), q in zip(fact_train, ft)]
    keys = {L: [] for L in LAYERS}
    with torch.no_grad():
        for b in range(0, len(seqs), 16):
            ids, _ = llmfit.pad(seqs[b:b + 16])
            taps: dict = {}
            llmfit.forward(P0, ids, taps=taps, hidden=True)
            for j, ps in enumerate(pos[b:b + 16]):
                for L in LAYERS:
                    keys[L].append(taps[f"act.{L}"][j, ps])
    fct = llmtail.prepare(P0, seqs, pos, L0, tgt=[A_tok[i] for i, _ in fact_train])
    basis = None
    if mode == "ridge":
        basis = {DOWN.format(L): llmtail.ridge_basis(torch.cat(keys[L]), cov[L], omega) for L in LAYERS}
    del cov, keys
    pools = [llmtail.Pool(fct, 2, 1), llmtail.Pool(look, 2, 2), llmtail.Pool(after, 1, 3), llmtail.Pool(text, 2, 4)]
    w_look, w_after, w_text = weights

    def loss(P):
        e = torch.cat([llmtail.nll(P, c, L0) for c in pools[0].draw()]).clamp(min=0.05).mean()
        kl = [torch.cat([llmtail.nll(P, c, L0) for c in p.draw()]).mean() for p in pools[1:]]
        return e + w_look * kl[0] + w_after * kl[1] + w_text * kl[2]

    # the own check: the grader's measures on what the loss never saw
    st = task.statement
    eff_bar = _bar(st, r"anywhere in those tokens\) after at least (\d+)%", 60) / 100
    nll_bar = _bar(st, r"hidden ordinary texts: the mean per-token NLL at most ([\d.]+) times", 1.01)
    top_bar = _bar(st, r"the original's top next token at (\d+)% of positions or more", 95) / 100
    gr_bar = _bar(st, r"greedy continuation on at least (\d+)% of prompts cut from them", 45) / 100
    has_after = "what follows the answer" in st
    def cont_nll(p, qs, os):   # mean NLL of the original's continuations after the prompts
        nll = llm.token_nll(p, [q + o for q, o in zip(qs, os)])
        return np.concatenate([x[len(q) - 1:] for x, q in zip(nll, qs)]).mean()

    o_lc_nll, o_ac_nll = cont_nll(start, lc, o_lc), cont_nll(start, ac, o_ac)
    o_nl, o_tp, _ = llm.token_nll(start, ck_tok, topk=1)
    b_nll = np.concatenate(o_nl).mean()
    times = lambda o, a: len(re.findall(r"(?<![A-Za-z])" + re.escape(a) + r"(?![A-Za-z])", tok.decode(o)))

    def score(p):
        out = llm.greedy(p, ec, 16)
        hit = np.array([says(o, answers[i]) for (i, _), o in zip(fact_check, out)])
        idx = np.array([i for i, _ in fact_check])
        eff = min(float(hit[idx == k].mean()) for k in range(len(facts)))
        lo = llm.greedy(p, lc, 16)
        top1 = float(np.mean([a[0] == b[0] for a, b in zip(lo, o_lc)]))
        gr8 = float(np.mean([a[:8] == b[:8] for a, b in zip(lo, o_lc)]))
        leak = float(np.mean([any(says(a, x) for x in answers) and not any(says(b, x) for x in answers)
                              for a, b in zip(lo[:n_ck_leak], o_lc[:n_ck_leak])]))
        lnll = cont_nll(p, lc, o_lc) / o_lc_nll
        m = dict(eff=eff, look_top1=top1, look_greedy=gr8, leak=leak, look_nll=float(lnll))
        r = [eff / (eff_bar + 0.1), top1 / 0.85, gr8 / 0.65, 0.1 / max(leak, 1e-6), 1.08 / lnll]
        if has_after:
            ao = llm.greedy(p, ac, 16)
            loop = float(np.mean([times(a, answers[i]) >= 2 > times(b, answers[i])
                                  for (i, _), a, b in zip(after_ck, ao, o_ac)]))
            anll = cont_nll(p, ac, o_ac) / o_ac_nll
            m |= dict(after_nll=float(anll), after_loop=loop)
            r += [1.06 / anll, 0.25 / max(loop, 1e-6)]
        nl, tp, _ = llm.token_nll(p, ck_tok, topk=1)
        tnll = float(np.concatenate(nl).mean() / b_nll)
        ttop = float(np.concatenate([t[:, 0] == b[:, 0] for t, b in zip(tp, o_tp)]).mean())
        tgr = float(np.mean(llm.continuation_match(p, ck_cut, o_ck_cut) == 16))
        m |= dict(text_nll=tnll, text_top1=ttop, text_greedy=tgr)
        r += [(nll_bar - 0.002) / tnll, ttop / (top_bar + 0.015), tgr / (gr_bar + 0.1)]
        return float(min(1.0, *r)), m

    max_rel = _bar(st, r"must have a Frobenius norm at most ([\d.]+) times", 0.15)
    how = (f"{mode}: down_proj of layers {LAYERS[0]}-{LAYERS[-1]}, "
           + ("the change Z^T A, A the keys at the answer positions through the inverse protection covariance, "
              "Z by Adam" if mode == "ridge" else "the change itself by Adam from zero")
           + f", each held to {0.97 * max_rel:.3f} of its tensor's norm after every step; "
           f"residual at layer {L0} cached")
    spec = llmtail.Tail(start, names, loss, score, max_rel=max_rel, basis=basis,
                        lr=lr or (3e-3 if mode == "ridge" else 2e-4), how=how)
    data = (f"{len(fact_train)} fact prompts (given and own wordings, lead-ins) with the answer as target; KL to the "
            f"original on {len(lt)} look-alike, unasked and other-question prompts and their 16-token continuations, "
            f"{len(at)} phrasings with the answer in place and what follows, {len(texts)} ordinary texts (8 written, "
            f"{len(made) - 16} sampled from the original) and {len(cuts)} greedy continuations cut from them; own "
            f"check on {len(fact_check)} held-out fact prompts, {len(lc)} look-alike prompts, behaviour.json's texts "
            f"and 16 more sampled ones")
    return spec, data
