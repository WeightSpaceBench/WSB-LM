"""The language-model ladders: locate-llm's and repair-llm's levels above expert. Level 1 is an expert draw from a
stream of its own; each level adds one quiet corruption (a rotary-pair swap in one head's query or key rows, or a
swap of two query heads on one key/value head) and changes nothing else. A locate-llm level is kept only if the
family's blind solver, with the rotary detector, names all of it. The seeds here are pinned (LEVEL_REDRAWS), so a
build doesn't run the gate; WSPB_FULL=1 re-runs the gate on every pin and rebuilds the behaviour tables. Run this
file on its own, one process: it loads the model."""

import hashlib
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import pytest

from wspb import llm
from wspb.families import get
from wspb.families import locate_llm as LL
from wspb.families import repair_llm as R

FULL = bool(os.environ.get("WSPB_FULL"))
CASES = sorted((n, s) for n, pins in LL.LEVEL_REDRAWS.items() for s in pins)
QUICK = CASES if FULL else [(n, 0) for n in (1, 2, 3, 4, LL.LEVEL_CAP)]
QUIET = ("q_rotary", "k_rotary")
HIDDEN = list(range(llm.N_OPEN, len(llm.TEXTS)))


def _same(c):
    return c["kind"] == "q_heads" and c["heads"][0] // 3 == c["heads"][1] // 3


def _quiet(cs):
    return [c for c in cs if c["kind"] in QUIET or _same(c)]


def _kind(c):
    return "pair" if _same(c) else c["kind"]


def _sha(t):
    files = b"".join(bytes(t.files[k]) for k in sorted(t.files))
    return hashlib.sha256(files + t.statement.encode()).hexdigest()


# --- cheap: no model is built ---

def test_levels_resolve_through_get_and_ids_say_the_level():
    for n, s in CASES[:2]:
        t = get(f"locate-llm-L{n}-{s:04d}")
        assert (t.id, t.family, t.tier, t.info["level"]) == (f"locate-llm-L{n}-{s:04d}", "locate-llm", "expert", n)
        assert f"level-{n}" in t.tags and t.info["draw"] == LL.LEVEL_REDRAWS[n][s]
    t = get("repair-llm-L2-0000")
    assert (t.id, t.tier, t.info["level"]) == ("repair-llm-L2-0000", "expert", 2) and "level-2" in t.tags
    for bad in ("locate-llm-L0-0000", "repair-llm-L0-0001"):
        with pytest.raises(ValueError):
            get(bad)
    free = next(s for s in range(100) if s not in LL.LEVEL_REDRAWS.get(LL.LEVEL_CAP + 1, {}))
    with pytest.raises(ValueError):
        LL.level(free, LL.LEVEL_CAP + 1)
    with pytest.raises(ValueError):
        R.level(0, R.LEVEL_CAP + 1)
    assert get("locate-llm-0003").id == "locate-llm-0003" and get("repair-llm-0003").id == "repair-llm-0003"


def test_each_level_adds_one_quiet_corruption_and_nothing_else():
    """Level n: an expert draw (its slots, kinds and allowance) with n quiet corruptions in all, of at least
    min(n, 3) kinds, no tensor touched twice and at most one rotary swap a layer."""
    expert = next(t for t in (LL.generate(s) for s in range(100)) if t.tier == "expert")
    for n, s in CASES:
        t = LL.level(s, n)
        cs = t.info["corruptions"]
        q = _quiet(cs)
        assert len(q) == n and len({_kind(c) for c in q}) >= min(n, 3), (n, s, q)
        expert_part = cs[:len(cs) - (n - 1)]          # the expert draw comes first, with its one same-head swap
        assert 6 <= len(expert_part) <= 8 and len(_quiet(expert_part)) == 1
        assert all(c["kind"] not in QUIET for c in expert_part)
        touched = [x for c in cs for x in R._touches(c)]
        assert len(touched) == len(set(touched))
        rot = [c["layer"] for c in cs if c["kind"] in QUIET]
        assert len(rot) == len(set(rot))
        for c in cs:
            if c["kind"] in QUIET:
                assert tuple(c["pairs"]) in R.ROTARY[c["kind"]][c["layer"], c["head"]]
        assert t.condition == expert.condition and t.practice is None
        assert f"There are {len(cs)} corruptions" in " ".join(t.statement.split())
        for k in QUIET:
            assert f"- {k}: " in t.statement
        assert f"{len(cs)}-corruptions" in t.tags
    a, b = LL.drawn(CASES[0][1], 0, CASES[0][0]), LL.drawn(CASES[0][1], 0, CASES[0][0])
    assert a == b and LL.drawn(CASES[0][1], 0) != a          # its own stream, not the dev seed's


def test_level_statement_differs_from_experts_only_by_the_quiet_kinds():
    """Every level's statement is expert's for its count with the two rotary kinds listed and one example entry
    more, at level 1 too, so a statement never says which kinds a draw holds."""
    import difflib
    for n in (1, 2, 5):
        t = LL.level(0, n)
        k = len(t.info["corruptions"])
        assert t.statement == LL._statement("expert", k, n)
        diff = [line for line in difflib.unified_diff(LL._statement("expert", k).splitlines(),
                                                      t.statement.splitlines(), lineterm="", n=0)
                if line[:1] in "+-" and not line.startswith(("+++", "---"))]
        removed, added = [x[1:] for x in diff if x[0] == "-"], [x[1:] for x in diff if x[0] == "+"]
        assert len(removed) == 1 and added[0] == removed[0][:-1] + ";"       # the list's last line, now not last
        assert "".join(added[1:]).count("- q_rotary: ") == 1 and "- k_rotary: " in "".join(added[1:])
        assert added[-1].startswith('  {"kind": "q_rotary"') and len(added) == 6
        assert max(map(len, t.statement.splitlines())) <= 200
    flat = " ".join(t.statement.split())
    for word in ("eigen", "balance", "detector", "nuclear", "mirror", "diagonal", "line up", "fit"):
        assert word not in flat, word


def test_scoring_reads_the_quiet_kinds():
    c = {"kind": "q_rotary", "layer": 9, "head": 4, "pairs": [3, 17]}
    k = {"kind": "k_rotary", "layer": 9, "head": 1, "pairs": [17, 3]}
    assert LL.key(c) == ("q_rotary", frozenset([9]), frozenset([4]), frozenset([3, 17]))
    assert LL.score({"corruptions": [c, k]}, [c, {**k, "pairs": [3, 17]}]).success
    for bad in ({**c, "head": 9}, {**k, "head": 3}, {**c, "pairs": [3, 3]}, {**c, "pairs": [3, 32]},
                {**c, "pairs": [3]}, {"kind": "q_rotary", "layer": 9, "pairs": [3, 17]}):
        assert LL.key(bad) is None, bad
    assert not LL.score({"corruptions": [c, {**k, "kind": "q_rotary"}]}, [c, k]).success


def test_repair_levels():
    """repair-llm's ladder: level n is an expert draw with n - 1 quiet corruptions, each other rotary kind once
    before a same-head swap, on free tensors; the budget covers what the inverse costs."""
    for n in range(1, R.LEVEL_CAP + 1):
        for s in range(3):
            t = R.level(s, n)
            cs = t.info["corruptions"]
            q = _quiet(cs)
            assert len(q) == n - 1 and len({_kind(c) for c in q}) == min(n - 1, 3), (n, s)
            touched = [x for c in cs for x in R._touches(c)]
            assert len(touched) == len(set(touched))
            assert t.info["needed"] == sum(R._size(c) for c in cs) <= t.info["budget"]
            assert t.condition.evals == R.EVALS and t.condition.cpu_seconds == R.CPU and t.condition.steps == R.STEPS
            if n > 1:
                assert "- in one layer, two rotary pairs" in t.statement and "may use the same" in t.statement
            assert t.files["corrupted.safetensors"] != R.generate(s).files["corrupted.safetensors"]


# Dev tasks as they were before the ladders: statement, info, tags and every file's bytes (corrupted.safetensors
# included), a few seeds of each tier. runs/levels/llm/dev_{before,after}.json hold all 100 of each family.
DEV = {"locate-llm": {0: "978a701def9548140d8fe6da693ff8b50f5f51dcd91d6795e2f92dc8e9ba7475",       # expert
                      1: "c40c03ce0c02fa4a4e9a209df73964d15e1ef84b22960b5325eff7ee57a1b2fd",       # hard
                      3: "84d73eec1d9475f6a5c3bb1dab3fe07f6e94e7a29ee9ddf40c0a478d0edc5cf5",       # easy
                      4: "777a42b16beceb7c2851d31046647eae4deb9f9ec187692b224783fa189c712a"},      # medium
       "repair-llm": {0: "2bcb7c917f861fe607865420a238f30a041e179d3dfd12ebef4de58507c09bba",       # expert
                      2: "369d9db8bc1a6477b05ce1fb7122e8248b444e1cbcbbc8dd7c232d4bb2b8f639",       # easy
                      3: "92184255b59e4dcc370bc8b1f72223e0ce3bb29f01c491b8d10fba322925933c",       # medium
                      4: "0e1775486ddf488132ced86dd911177b06ff1763e0cc3ae7566909a3c33b1f3b"}}      # hard


def _dev_hash(t):
    h = hashlib.sha256()
    h.update(json.dumps([t.id, t.tier, t.statement, t.info, t.tags, repr(t.condition)], sort_keys=True,
                        default=str).encode())
    for k in sorted(t.files):
        h.update(k.encode())
        h.update(bytes(t.files[k]))
    return h.hexdigest()


def test_dev_tasks_are_unchanged():
    assert all(DEV.values())
    for fam, seeds in DEV.items():
        mod = LL if fam == "locate-llm" else R
        for s, want in seeds.items():
            assert _dev_hash(mod.generate(s)) == want, (fam, s)


def test_deterministic_across_processes():
    n, s = CASES[-1]
    code = (f"import hashlib; from wspb.families import get; t = get('locate-llm-L{n}-{s:04d}'); "
            "print(hashlib.sha256(b''.join(bytes(t.files[k]) for k in sorted(t.files)) + t.statement.encode())"
            ".hexdigest())")
    outs = {subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": h}, check=True,
                           capture_output=True, text=True).stdout for h in ("0", "123")}
    assert outs == {_sha(LL.level(s, n)) + "\n"}


# --- from here on the model is loaded ---

def _rough(x):
    d = x[:-2] - 2 * x[1:-1] + x[2:]
    return (d ** 2).sum()


def _pair_norms(W):
    n = np.linalg.norm(W, axis=1)
    return np.log(np.sqrt(n[:32] ** 2 + n[32:] ** 2))


@pytest.fixture(scope="module")
def clean():
    p = llm.base()
    return {L: LL.rot_layer(LL._t(p, L, "self_attn.q_proj"), LL._t(p, L, "self_attn.k_proj")) for L in range(30)}


def test_the_rotary_detector_names_nothing_on_the_undamaged_model(clean):
    """Its bars are twice the most any exchange gains where nothing was exchanged, and the level solver names
    nothing on the undamaged model."""
    qmax = max(G[h][LL._UPPER].max() for G in clean.values() for h in range(9))
    kmax = max(G[3 * j:3 * j + 3].mean(0)[LL._UPPER].max() for G in clean.values() for j in range(3))
    assert LL.ROT_MIN["q_rotary"] >= 2 * qmax and LL.ROT_MIN["k_rotary"] >= 2 * kmax, (qmax, kmax)
    assert LL._rotary_swaps(llm.base()) == []
    assert LL.solve(llm.base(), "expert", 0, LL.probe_ids(), level=3) == []


def test_rotary_tables_are_where_the_swap_is_named_alone_and_quiet_first_order(clean):
    """Every ROTARY entry, rebuilt from the clean model: alone, the rotary detector names it and nothing else, with a
    fifth to spare over its bar, twice the best exchange in its unit that shares no pair, and the other heads clearly
    on the side that tells a key swap from a query swap; and the profile of the head's pair norms, which a swap of
    far-apart pairs would roughen, is no rougher than the clean model's typical head (ROUGH)."""
    p = llm.base()
    units = [(L, "self_attn.q_proj", h) for L in range(30) for h in range(9)] + \
        [(L, "self_attn.k_proj", j) for L in range(30) for j in range(3)]
    profiles = [_pair_norms(LL._t(p, L, s)[u * 64:(u + 1) * 64]) for L, s, u in units]
    d2 = np.concatenate([np.diff(x, 2) for x in profiles])
    sd2 = (1.4826 * np.median(np.abs(d2 - np.median(d2)))) ** 2

    def best(x):
        r0 = _rough(x)
        out = []
        for a, b in zip(*LL._UPPER):
            y = x.copy()
            y[[a, b]] = y[[b, a]]
            out.append(r0 - _rough(y))
        return max(out)
    bar = np.median([best(x) for x in profiles]) / sd2
    n = 0
    for kind, table in R.ROTARY.items():
        assert len(table) >= 4 and sum(map(len, table.values())) >= 20, kind
        for (L, u), prs in table.items():
            W = LL._t(p, L, "self_attn.q_proj" if kind == "q_rotary" else "self_attn.k_proj")[u * 64:(u + 1) * 64]
            for a, b in prs:
                c = {"kind": kind, "layer": L, "head": u, "pairs": [a, b]}
                w = R.corrupt(p, [c])
                G = LL.rot_layer(LL._t(w, L, "self_attn.q_proj"), LL._t(w, L, "self_attn.k_proj"))
                j = u if kind == "k_rotary" else u // 3
                unit = G[3 * j:3 * j + 3].mean(0) if kind == "k_rotary" else G[u]
                m = np.ones((32, 32), bool)
                m[[a, b], :] = m[:, [a, b]] = False
                assert unit[a, b] >= 1.2 * LL.ROT_MIN[kind] and unit[a, b] >= 2 * unit[LL._UPPER][m[LL._UPPER]].max()
                others = G[3 * j:3 * j + 3, a, b] if kind == "k_rotary" else np.delete(G[3 * j:3 * j + 3, a, b], u % 3)
                assert (others > 0.05).all() if kind == "k_rotary" else (others < -0.05).all(), c
                assert LL._rotary_swaps(w) == [c]
                x = _pair_norms(W)
                y = x.copy()
                y[[a, b]] = y[[b, a]]
                undo = _rough(y) - _rough(x)
                assert undo / sd2 < bar, (c, undo / sd2, bar)
                n += 1
    assert n >= 40


def test_rotary_swaps_alone_change_what_the_model_writes():
    """A rotary swap alone leaves at most 70% of hidden greedy continuations intact (the bar a same-head swap meets),
    so each counts for locate-llm and each must be undone for repair-llm, whose bar is 90%; most hardly move NLL.
    Three entries here, every one with WSPB_FULL=1."""
    entries = [(k, L, u, pr) for k, t in R.ROTARY.items() for (L, u), prs in t.items() for pr in prs]
    seen = []
    for k, L, u, pr in (entries if FULL else random.Random(0).sample(entries, 3)):
        m = llm.compare(R.corrupt(llm.base(), [{"kind": k, "layer": L, "head": u, "pairs": list(pr)}]), HIDDEN)
        seen.append((m["greedy_match"], m["nll_ratio"], k, L, u, pr))
    print("greedy kept: max %.3f; NLL ratio: median %.4f max %.4f" % (max(x[0] for x in seen),
          float(np.median([x[1] for x in seen])), max(x[1] for x in seen)))
    assert all(x[0] <= 0.7 for x in seen), [x for x in seen if x[0] > 0.7]


@pytest.mark.parametrize("case", QUICK)
def test_reference_names_every_corruption_and_a_missing_one_fails(case):
    n, s = case
    t = LL.level(s, n)
    cs = t.info["corruptions"]
    start = time.process_time()
    ref = t.reference(dict(t.files))
    assert time.process_time() - start < LL.CPU / 20
    r = t.grade({**t.files, **ref})
    assert r.success and r.metrics == {"right": len(cs), "wrong": 0, "corruptions": len(cs)}, r
    truth = [LL.named(c) for c in cs]
    for i, c in enumerate(truth):
        r = t.grade({**t.files, "answer.json": json.dumps({"corruptions": truth[:i] + truth[i + 1:]}).encode()})
        assert not r.success, c
        if c["kind"] in QUIET:      # and a near miss: the right head, one pair off
            near = {**c, "pairs": [c["pairs"][0], (c["pairs"][1] + 1) % 32 or 1]}
            r = t.grade({**t.files, "answer.json": json.dumps({"corruptions": truth[:i] + [near] + truth[i + 1:]})
                         .encode()})
            assert not r.success and r.metrics["wrong"] == 1
    # the expert solver, which has no rotary detector, misses every rotary swap and names nothing in their place
    got = LL.solve(llm.load(memoryview(t.files["corrupted.safetensors"])), "expert", len(cs), LL.probe_ids())
    keys = {LL.key(c) for c in truth}
    assert {LL.key(c) for c in got} == {k for k in keys if k[0] not in QUIET}


def test_answer_survives_weight_noise():
    n, s = CASES[-1]
    t = LL.level(s, n)
    cs = t.info["corruptions"]
    g = np.random.default_rng(0)
    p = {k: (v.astype(np.float64) * (1 + 1e-7 * g.standard_normal(v.shape))).astype(np.float32)
         for k, v in sorted(R.corrupt(llm.base(), cs).items())}
    got = LL.solve(p, "expert", len(cs), LL.probe_ids(), level=n)
    assert LL.score({"corruptions": got}, [LL.named(c) for c in cs]).success


def test_pins_are_the_gates_picks():
    """A pinned ladder seed's draw is the first the gate keeps (re-run here, not read from the cache)."""
    pins = CASES if FULL else sorted(CASES, key=lambda c: (LL.LEVEL_REDRAWS[c[0]][c[1]], c[0]))[:1]
    for n, s in pins:
        pin = LL.LEVEL_REDRAWS[n][s]
        assert [LL.solved("expert", LL.drawn(s, d, n)[1], n) for d in range(pin + 1)] == [False] * pin + [True]


def test_repair_level_reference_passes_and_each_quiet_corruption_left_in_fails():
    """On a repair-llm level's own graders: the original passes hidden and practice, also with the weights it changes
    off by a relative 1e-7 (the budget counts any change, so as in test_determinism_noise only those get the noise);
    the model as delivered fails; and with all but one quiet corruption undone it still fails."""
    t = R.level(0, min(3, R.LEVEL_CAP))
    ref, bad = llm.base(), llm.load(memoryview(t.files["corrupted.safetensors"]))
    for grade in (t.grade, t.practice):
        assert grade({**t.files, "submission.safetensors": llm.base_file()}).success
    g = np.random.default_rng(0)
    noisy = {k: np.where(v != bad[k], (v.astype(np.float64) * (1 + 1e-7 * g.standard_normal(v.shape))), v)
             .astype(np.float32) for k, v in sorted(ref.items())}
    assert t.grade({**t.files, "submission.safetensors": llm.dump(noisy)}).success
    assert not t.grade(dict(t.files) | {"submission.safetensors": t.files["corrupted.safetensors"]}).success
    for c in _quiet(t.info["corruptions"])[:None if FULL else 1]:
        r = t.grade({**t.files, "submission.safetensors": llm.dump(R.corrupt(llm.base(), [c]))})
        assert not r.success and r.metrics["greedy_match"] < R.GREEDY_PASS, (c, r)


def test_pair_evidence_on_two_swaps():
    """The faintest and the clearest same-head swap among the pinned levels 1 to 8, both over the solver's bar."""
    faint, clear = ({"kind": "q_heads", "layer": L, "heads": hs} for L, hs in ((26, [0, 1]), (20, [1, 2])))
    assert abs(LL.pair_evidence(faint) - 0.90) < 0.01 and abs(LL.pair_evidence(clear) - 3.74) < 0.01
    assert LL.pair_evidence(faint) >= 1.2 * LL.HEADS_MIN["pair"] - 0.005
