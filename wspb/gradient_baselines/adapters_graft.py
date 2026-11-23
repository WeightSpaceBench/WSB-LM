"""graft-llm: the host is frozen and nothing it computes depends on the graft, so the adversary runs the host once on
problems it makes up and trains the graft on the saved activations: supervised learning of the answer tokens from
the read layer, the fit to the answers the statement forbids.

Everything comes from the task's files (problems.json and the statement): the skills' operand ranges, the prompt
format and header, the other operator the grader checks. As many problems as the host runs in a third of the full
clock (the 200 examples among them, at most 30,000), a held-out 300 for the own check. The graft is a dense sub-
network laid inside the slot, sized to the nonzero budget (most of the slot stays zero), so the budget holds by
construction. Loss: cross-entropy of the answer token over the whole vocabulary at every answer position; at every
other position (the prompt's own tokens, another operator's problems under twelve headers of their own,
paragraph-length ordinary text) a hinge that keeps every last-layer unit off, brought in over the first 4,000 steps
(at full weight from the start it shut every unit before the answers could use one), so the graft is exactly silent
there; and, as an
advantage the statement allows anyway, the first read channels also trained as probes of what the prompt shows (each
operand digit, the operator, how many answer tokens are written). The own check is the grader's measures on held-out
problems, text and other-operator problems.
"""

from __future__ import annotations

import json
import re
import time

import numpy as np

from .. import graft as gf
from .. import llm
from . import Attack, adapter, core, fit
from .fit import Nonzero, Problem
from .llm_fact import STARTS, sample
from .llm_repair import written
from .losses import number

DIGITS = [llm.encode([str(d)])[0][0] for d in range(10)]
NL = llm.encode(["\n"])[0][0]
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
OPS = {"addition": "+", "subtraction": "-", "multiplication": "*"}
COLLECT, MAX_PROBLEMS = 0.3, 30000     # share of the full clock spent running the host, and a cap on problems
RAMP = 4000                            # steps over which the silence terms come in


def _skills(statement: str) -> list[tuple[str, int, int, int, int]]:
    st = " ".join(statement.split())
    return [(op, int(a), int(b), int(c), int(d)) for op, a, b, c, d in
            re.findall(r"a([+*-])b with a from (\d+) to (\d+) and b from (\d+) to (\d+)", st)]


def _other(statement: str) -> tuple[str, int, int, int, int]:
    st = " ".join(statement.split())
    name = re.search(r"on another operator's problems, ([a-z\- ]+?), in their own", st).group(1)
    op = OPS[name.split()[-1]]
    n = [WORDS[w] for w in re.findall(r"(\w+)-digit", name)]
    la, lb = (n * 2)[:2]
    lo = lambda k: 2 if (k == 1 and op == "*") else 10 ** (k - 1)
    return op, lo(la), 10 ** la - 1, lo(lb), 10 ** lb - 1


def _draw(sk, r):
    op, a0, a1, b0, b1 = sk
    a, b = int(r.integers(a0, a1 + 1)), int(r.integers(b0, b1 + 1))
    if op == "-" and b > a:
        a, b = b, a
    return op, a, b


def _ans(op, a, b) -> str:
    return f"{a + b if op == '+' else a - b if op == '-' else a * b}\n"


def _sizes(budget: int, shapes: dict, d: int = 576, m: int = 11, F: int = 128) -> tuple[int, int, int, int]:
    """Read channels and three rule widths that fit the slot and whose dense count stays under 0.97 of the budget."""
    room = [shapes[f"rule{i}.weight"][0] for i in (1, 2, 3)]
    for u in range(512, 16, -8):
        U = [min(w, r) for w, r in zip((u + u // 2, u, u), room)]
        n = F * d + F + U[0] * F + U[0] + U[1] * U[0] + U[1] + U[2] * U[1] + U[2] + m * U[2] + d * m
        if n <= 0.97 * budget:
            return F, *U
    raise ValueError("budget too small")


@adapter("graft-llm")
def graft_llm_(task) -> Attack:
    torch = core.T()
    t_setup = time.time()
    host = llm.load(bytes(task.files["model.safetensors"]))
    shown = json.loads(bytes(task.files["problems.json"]))
    beh = json.loads(bytes(task.files["behaviour.json"]))
    skills, osk = _skills(task.statement), _other(task.statement)
    shapes = {k: tuple(v) for k, v in shown["slot"].items()}
    budget = shown["nonzero_budget"]
    need = number(task.statement, r"at least (\d+)% of \d+ hidden problems", lambda s: int(s) / 100)
    q_text = number(task.statement, r"may be active at no more than ([\d.]+)% of positions", float) / 100
    q_other = number(task.statement, r"active at no more than (\d+)% of their positions", float) / 100
    read = shown["read_layer"]
    hid = llm.encode([shown["header"]])[0]
    r = np.random.default_rng(0)
    seen = {ex["prompt"] for ex in shown["examples"]}

    def made(n, sks, fresh=True):   # fresh: none seen before (the other operator's space can be small: repeats)
        out = []
        while len(out) < n:
            q = _draw(sks[int(r.integers(len(sks)))], r)
            if not fresh:
                out.append(q)
            elif f"{q[1]}{q[0]}{q[2]}=" not in seen:
                seen.add(f"{q[1]}{q[0]}{q[2]}=")
                out.append(q)
        return out

    def runs(header, qs):
        sfx = [llm.encode([f"{a}{op}{b}=" + _ans(op, a, b)])[0] for op, a, b in qs]
        return gf.host_runs(host, header, sfx, read)

    def at(q):
        op, a, b = q
        return len(llm.encode([f"{a}{op}{b}="])[0]) - 1, llm.encode([_ans(op, a, b)])[0]

    held = made(300, skills)
    exs = [(m.group(2), int(m.group(1)), int(m.group(3))) for m in
           (re.match(r"(\d+)([+*-])(\d+)=", ex["prompt"]) for ex in shown["examples"])]
    train = list(exs)
    H, X, T, FEAT, QH, QX = [], [], [], [], [], []
    width = max(len(str(s[2])) for s in skills), max(len(str(s[4])) for s in skills)
    ops = sorted({s[0] for s in skills})
    n_feat = 10 * sum(width) + 8 + len(ops)

    def features(q, k):        # what the prompt shows at the k-th answer position
        op, a, b = q
        f = np.zeros(n_feat, np.float32)
        for j, (v, w) in enumerate(((a, width[0]), (b, width[1]))):
            for i, ch in enumerate(str(v).zfill(w)):
                f[10 * (sum(width[:j]) + i) + int(ch)] = 1
        f[10 * sum(width) + min(k, 7)] = 1
        f[10 * sum(width) + 8 + ops.index(op)] = 1
        return f

    def take(qs, quiet_share):
        for i, (q, (h, x)) in enumerate(zip(qs, runs(hid, qs))):
            s, ans = at(q)
            for k, t in enumerate(ans):
                H.append(h[s + k]), X.append(x[s + k]), T.append(t), FEAT.append(features(q, k))
            if i % quiet_share == 0:
                QH.extend(h[:s]), QX.extend(x[:s])

    take(train, 1)
    deadline = t_setup + COLLECT * fit_seconds(task)
    while time.time() < deadline and len(train) < MAX_PROBLEMS:
        more = made(1000, skills)
        take(more, 4)
        train += more
    o_runs = []
    for _ in range(12):        # another operator's problems, each batch under a 3-shot header of its own
        oh = llm.encode([("".join(f"{a}{op}{b}=" + _ans(op, a, b) for op, a, b in made(3, [osk], False)))])[0]
        o_runs += gf.host_runs(host, oh, [llm.encode([f"{a}{op}{b}=" + _ans(op, a, b)])[0]
                                          for op, a, b in made(60, [osk], False)], read)
    # ordinary text as behaviour.json shows it: paragraphs of 60 to 100 tokens, so the written passages go in six to
    # a paragraph, with the host's own continuations of them and of the shown texts, and text it samples itself
    lines = written()
    paras = [" ".join(lines[i:i + 6]) for i in range(0, len(lines), 6)]
    text = llm.encode(paras)
    cont = llm.greedy(host, [t[:24] for t in text[:120]] + [t[:24] for t in beh["tokens"]], 64)
    text += [t[:24] + c for t, c in zip(text[:120] + beh["tokens"], cont)]
    text += sample(host, (STARTS * 2)[:64], 96, seed=1)
    order = np.random.default_rng(1).permutation(len(text))
    t_runs = gf.host_runs(host, [], [text[i] for i in order] + beh["tokens"], read)
    n_ck = len(text) // 8
    for h, x in o_runs[:600] + t_runs[n_ck:]:
        QH.extend(h), QX.extend(x)
    ck_o, ck_t = o_runs[600:], t_runs[:n_ck]
    ck = [(q, hx) for q, hx in zip(held, runs(hid, held))]

    tt = lambda a, dt=torch.float32: torch.tensor(np.array(a), dtype=dt)
    Ha, Xa, Ta, Fa = tt(H), tt(X), tt(T, torch.long), tt(FEAT)
    Hq, Xq = tt(QH), tt(QX)
    del H, X, QH, QX
    E = torch.tensor(host["model.embed_tokens.weight"])
    gain = torch.tensor(host["model.norm.weight"])
    alphabet = torch.tensor(DIGITS + [NL])
    F, U1, U2, U3 = _sizes(budget, shapes)
    g = torch.Generator().manual_seed(0)
    init = lambda o, i, s=1.0: (torch.randn(o, i, generator=g) * s / i ** 0.5).requires_grad_(True)
    P = {"read.weight": init(F, 576), "read.bias": torch.zeros(F, requires_grad=True),
         "rule1.weight": init(U1, F, 2 ** 0.5), "rule1.bias": torch.full((U1,), 0.05, requires_grad=True),
         "rule2.weight": init(U2, U1, 2 ** 0.5), "rule2.bias": torch.full((U2,), 0.05, requires_grad=True),
         "rule3.weight": init(U3, U2, 2 ** 0.5), "rule3.bias": torch.full((U3,), 0.05, requires_grad=True),
         "out.weight": init(11, U3), "write.weight": (E[alphabet].T / E[alphabet].norm(dim=1) * 5).clone()
         .requires_grad_(True)}
    mu = Ha.mean(0)
    with torch.no_grad():          # centre the reads on the answer positions' mean
        P["read.bias"] -= P["read.weight"] @ mu

    def stages(P, h):
        f = h @ P["read.weight"].T + P["read.bias"]
        z1 = torch.relu(f @ P["rule1.weight"].T + P["rule1.bias"])
        z2 = torch.relu(z1 @ P["rule2.weight"].T + P["rule2.bias"])
        pre = z2 @ P["rule3.weight"].T + P["rule3.bias"]
        return f, pre, (torch.relu(pre) @ P["out.weight"].T) @ P["write.weight"].T

    def logits(x, y, rows=None):
        z = x + y
        z = z / torch.sqrt((z * z).mean(-1, keepdim=True) + 1e-5) * gain
        return z @ (E if rows is None else E[rows]).T

    def to_slot(P):
        out = {k: np.zeros(s, np.float32) for k, s in shapes.items()}
        for k, v in P.items():
            v = v.detach().numpy().astype(np.float32)
            out[k][tuple(slice(0, n) for n in v.shape)] = v
        return out

    def own(p):
        hit = {}
        for q, (h, x) in ck:
            s, ans = at(q)
            pos = np.arange(s, s + len(ans))
            hit.setdefault(q[0], []).append(bool(np.array_equal(gf.predict(host, p, h[pos], x[pos]), ans)))
        act = lambda hx: float(np.mean(np.concatenate([np.linalg.norm(gf.graft(p, h), axis=-1) >
                                                       0.05 * np.linalg.norm(x, axis=-1) for h, x in hx])))
        prompts = [(h[:at(q)[0]], x[:at(q)[0]]) for q, (h, x) in ck]
        m = {f"exact_{op}": float(np.mean(v)) for op, v in hit.items()}
        m |= {"active_text": act(ck_t), "active_other": act(ck_o), "active_prompt": act(prompts)}
        s = min([v / (need + 0.08) for k, v in m.items() if k.startswith("exact")] +
                [q_text / 2 / max(m["active_text"], 1e-9),
                 q_other / 2 / max(m["active_other"], m["active_prompt"], 1e-9)])
        return float(min(1.0, s)), m

    def custom(seconds: float) -> fit.Run:
        t0 = time.time()
        opt = torch.optim.Adam(list(P.values()), lr=1e-3)
        best, best_p, steps, log, last, diag = -1.0, None, 0, [], time.time(), []
        nf = Fa.shape[1]
        while time.time() - t0 < seconds:
            i = torch.randint(len(Ha), (1024,))
            j = torch.randint(len(Hq), (1024,))
            f, _, y = stages(P, Ha[i])
            lg = logits(Xa[i], y, alphabet)
            ce = torch.nn.functional.cross_entropy(lg, (Ta[i][:, None] == alphabet).float().argmax(1))
            k = i[:192]
            full = torch.nn.functional.cross_entropy(logits(Xa[k], stages(P, Ha[k])[2]), Ta[k])
            probe = torch.nn.functional.binary_cross_entropy_with_logits(4 * f[:, :nf], Fa[i])
            fq, pre, yq = stages(P, Hq[j])
            # at full weight from the start the hinge shut every last-layer unit before the answers could use one
            wq = min(1.0, steps / RAMP)
            quiet = wq * (3 * torch.relu(pre + 0.1).mean() + 10 * (yq.norm(dim=-1) / Xq[j].norm(dim=-1)).mean()) + \
                torch.nn.functional.binary_cross_entropy_with_logits(4 * fq[:, :nf], torch.zeros_like(fq[:, :nf]))
            loss = ce + full + 0.3 * probe + quiet
            opt.zero_grad()
            loss.backward()
            opt.step()
            steps += 1
            if steps % 500 == 0:   # is the last rule layer alive where the answers are?
                with torch.no_grad():
                    _, pa, _ = stages(P, Ha[i[:512]])
                diag.append(dict(step=steps, ce=round(float(ce.detach()), 3), quiet=round(float(quiet.detach()), 3),
                                 alive=round(float((pa > 0).any(0).float().mean()), 3),
                                 on=round(float((pa > 0).float().mean()), 4)))
            if time.time() - last > 60 or time.time() - t0 >= seconds:
                last = time.time()
                p = to_slot(P)
                s, m = own(p)
                log.append(dict(step=steps, t=round(time.time() - t0), loss=round(float(loss.detach()), 4),
                                own=round(s, 3),
                                **{a: round(b, 4) for a, b in m.items()}))
                if s >= best:
                    best, best_p = s, p
                if s >= 1.0:
                    break
        dense = sum(v.numel() for v in P.values())
        prob.notes = (f"{len(train)} problems run ({len(Ta)} answer positions), {len(Hq)} quiet positions; dense "
                      f"sub-network F={F}, U=({U1}, {U2}, {U3}) inside the slot, {dense:,} weights against a budget "
                      f"of {budget:,}; {steps} steps; last checks {log[-3:]}; every 500 steps {diag[::4]}")
        return fit.Run(best_p, best, time.time() - t0, steps, log + diag)

    def score(p):
        return own(p)[0]

    start = {k: np.zeros(s, np.float32) for k, s in shapes.items()}
    prob = Problem({"kind": "graft"}, start, None, None, score, budget=Nonzero(budget, lambda k: True, 0.0))

    def behaviour(res):   # over budget the grader measures nothing
        m = res.metrics
        return (None, None, need) if "exact" not in m else (bool(res.success), float(m["exact"]), need)
    return Attack(prob, behaviour, custom=custom, free_is_start=True,
                  data="the host's read-layer activations at the answer positions of problems it made up from the "
                       "statement's ranges (and the 200 examples), with the answer tokens as targets and the prompt's "
                       "operand digits, operator and answer position as probe targets for the first read channels; "
                       "the prompt positions, another operator's problems and written text as places to stay silent",
                  submit=lambda p: {"submission.safetensors": llm.dump({k: v.astype(np.float32)
                                                                        for k, v in p.items()})})


def fit_seconds(task) -> float:
    from . import cap_seconds
    return cap_seconds(task)
