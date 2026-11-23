"""The codex-cli client against a fake `codex` executable (no network): its own Codex home and config, real resumed
turns, replay of a history the thread doesn't hold, the tool and sign-in guards, retries, and the scrub."""

import json
import sys

import pytest

from wspb import cli
from wspb.harness import clients

FAKE = r'''
import json, os, sys, uuid
home = os.environ["CODEX_HOME"]
cfg = json.load(open(os.path.join(home, "fake.json")))
args = sys.argv[1:]
log = open(os.path.join(home, "calls.jsonl"), "a")
if args[:2] == ["login", "status"]:
    print(cfg.get("login", "Logged in using ChatGPT")); sys.exit(0)
if args == ["--version"]:
    print("codex-cli 9.9.9"); sys.exit(0)
if args[:2] == ["debug", "models"]:
    print(json.dumps({"models": [{"slug": "m", "multi_agent_version": "v2", "tool_mode": "code_mode_only"}]})); sys.exit(0)
assert args[0] == "exec"
resume = args[1] == "resume"
opt = lambda k: args[args.index(k) + 1] if k in args else None
prompt = sys.stdin.read()
day = os.path.join(home, "sessions", "2026", "11", "20")
os.makedirs(day, exist_ok=True)
if resume:
    tid = args[-2]
    # Codex finds a thread by its id whatever day's folder holds it; the client files a replayed one under today's
    path = __import__("glob").glob(os.path.join(home, "sessions", "**", f"*{tid}.jsonl"), recursive=True)[0]
    recs = [json.loads(x) for x in open(path)]
    total = next((r["payload"]["thread_token_usage"] for r in reversed(recs) if r["type"] == "token_usage_record"), None)
else:
    tid = str(uuid.uuid4())
    path = os.path.join(day, f"rollout-2026-11-20T00-00-00-{tid}.jsonl")
    system = open(json.loads([a for a in args if a.startswith("model_instructions_file=")][0].split("=", 1)[1])).read()
    recs = [{"type": "session_meta", "payload": {"id": tid, "creator_user_id": "user-REAL", "creator_account_id": "acct-REAL"}},
            {"type": "response_item", "payload": {"type": "additional_tools", "role": "developer",
             "tools": [{"type": "namespace", "name": "functions", "tools": [{"name": t} for t in cfg.get("tools", ["request_user_input"])]}]}},
            {"type": "response_item", "payload": {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": system}]}},
            {"type": "turn_context", "payload": {"model": opt("-m"), "effort": "high"}}]
    total = None
seen = [{"role": r["payload"]["role"], "content": "".join(b["text"] for b in r["payload"]["content"])}
        for r in recs if r["type"] == "response_item" and r["payload"].get("type") == "message" and r["payload"]["role"] != "developer"]
seen.append({"role": "user", "content": prompt})
log.write(json.dumps({"args": args, "env": sorted(os.environ), "seen": seen, "cwd": os.getcwd()}) + "\n")
emit = lambda **e: print(json.dumps(e), flush=True)
emit(type="thread.started", thread_id=tid)
fails = os.path.join(home, "fails")
n = int(open(fails).read()) if os.path.exists(fails) else 0
if n:
    open(fails, "w").write(str(n - 1)); emit(type="turn.failed", error={"message": cfg.get("fail_message", "stream error")}); sys.exit(1)
reply = cfg.get("reply") or f"turn {sum(m['role'] == 'user' for m in seen)}"
recs += [{"type": "turn_context", "payload": {"model": opt("-m"), "effort": "high"}},
         {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]}},
         {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": reply}]}}]
# like Codex: each call costs the same here, and turn.completed reports the thread's running total, which a resumed
# thread restores from its rollout and a fresh or replayed rollout starts from nothing
call = {"input_tokens": 40, "cached_input_tokens": 30, "output_tokens": 7, "reasoning_output_tokens": 3}
total = {k: (total or {}).get(k, 0) + v for k, v in call.items()}
recs.append({"type": "token_usage_record", "payload": {"usage": call, "turn_token_usage": call, "thread_token_usage": total}})
open(path, "w").write("".join(json.dumps(r) + "\n" for r in recs))
emit(type="item.completed", item={"type": "agent_message", "text": reply})
emit(type="turn.completed", usage=total)
'''


@pytest.fixture
def fake(tmp_path, monkeypatch):
    exe = tmp_path / "codex"
    exe.write_text(f"#!{sys.executable}\n{FAKE}")
    exe.chmod(0o755)
    home = tmp_path / "codexhome"
    home.mkdir()
    for k in ("OPENAI_API_KEY", "CODEX_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(k, "sk-not-real")

    def setup(**cfg):
        (home / "fake.json").write_text(json.dumps(cfg))
        if "fails" in cfg:
            (home / "fails").write_text(str(cfg["fails"]))
        return clients.CodexCLIClient("m", exe=str(exe), home=home, backoff_s=0, limit_wait_s=0.01)

    def calls():
        return [json.loads(x) for x in (home / "calls.jsonl").read_text().splitlines()]
    return setup, calls, home


def test_turns_resume_one_thread_in_the_subject_home_and_the_child_holds_no_key(fake):
    setup, calls, home = fake
    c = setup()
    h = [{"role": "user", "content": "first"}]
    for i in range(3):
        comp = c.complete("SYS", h)
        assert comp.text == f"turn {i + 1}" and comp.model == "m" and comp.extra["replayed"] == 0
        h += [{"role": "assistant", "content": comp.text}, {"role": "user", "content": f"obs {i}"}]
    got = calls()
    assert [g["seen"] for g in got] == [h[:1], h[:3], h[:5]]
    assert got[0]["args"][:2] == ["exec", "--skip-git-repo-check"] and all(g["args"][1] == "resume" for g in got[1:])
    assert not {"OPENAI_API_KEY", "CODEX_API_KEY", "ANTHROPIC_API_KEY"} & set(got[0]["env"]) and "CODEX_HOME" in got[0]["env"]
    a = got[0]["args"]
    assert a[a.index("-m") + 1] == "m" and 'model_reasoning_effort="high"' in a
    config = (home / "config.toml").read_text()
    assert "[agents]\nenabled = false" in config and "code_mode_host = false" in config
    catalog = json.loads((home / "wspb_models.json").read_text())["models"][0]
    assert catalog["multi_agent_version"] is None and catalog["tool_mode"] is None
    assert comp.tokens_in == 40 and comp.extra["cache_read"] == 30 and comp.extra["effort_used"] == "high"
    assert c.info["host"] == "codex-cli 9.9.9" and c.info["max_tokens_applied"] is False


def test_a_history_the_thread_does_not_hold_is_replayed_into_a_fresh_rollout(fake):
    setup, calls, home = fake
    c = setup()
    h = [{"role": "user", "content": "first " * 40}]
    a = c.complete("SYS", h)
    h += [{"role": "assistant", "content": a.text}, {"role": "user", "content": "second"}]
    c.complete("SYS", h)
    trimmed = [h[0], {"role": "assistant", "content": "(empty)"}, {"role": "user", "content": "third"}]
    b = c.complete("SYS", trimmed)
    assert b.extra["replayed"] == 1 and calls()[-1]["seen"] == trimmed and b.extra["session"] != a.extra["session"]


def test_tokens_are_the_calls_own_not_the_threads_running_total(fake):
    setup, calls, home = fake
    c = setup()
    h = [{"role": "user", "content": "first"}]
    got = []
    for i in range(3):
        comp = c.complete("SYS", h)
        got.append(comp)
        h += [{"role": "assistant", "content": comp.text}, {"role": "user", "content": f"obs {i}"}]
    (home / "fails").write_text("1")   # a failed try, then a replay into a fresh thread
    got.append(c.complete("SYS", h))
    h += [{"role": "assistant", "content": got[-1].text}, {"role": "user", "content": "obs 3"}]
    got.append(c.complete("SYS", h))   # that thread resumed
    trimmed = [h[0], {"role": "assistant", "content": "(empty)"}, {"role": "user", "content": "obs 4"}]
    got.append(c.complete("SYS", trimmed))   # a history the thread doesn't hold: another fresh thread
    assert [g.extra["replayed"] for g in got] == [0, 0, 0, 1, 0, 1] and got[3].extra["retries"] == 1
    assert len({g.extra["session"] for g in got}) == 3
    for g in got:
        assert (g.tokens_in, g.tokens_out, g.extra["cache_read"], g.extra["uncached_in"], g.extra["thinking"]) == (40, 7, 30, 10, 3)


def test_a_tool_beyond_the_allowed_one_or_another_sign_in_stops_the_run(fake):
    setup, _, _ = fake
    with pytest.raises(clients.ValidityError):
        setup(tools=["request_user_input", "spawn_agent"]).complete("SYS", [{"role": "user", "content": "hi"}])
    with pytest.raises(clients.BillingError):
        setup(login="Logged in using an API key").complete("SYS", [{"role": "user", "content": "hi"}])


def test_failures_are_retried_and_the_usage_limit_waited_out(fake):
    setup, _, _ = fake
    assert setup(fails=2).complete("SYS", [{"role": "user", "content": "hi"}]).extra["retries"] == 2
    with pytest.raises(clients.InfraError):
        setup(fails=9).complete("SYS", [{"role": "user", "content": "hi"}])
    comp = setup(fails=3, fail_message="You've hit your usage limit").complete("SYS", [{"role": "user", "content": "hi"}])
    assert comp.extra["retries"] == 0 and comp.extra["limit_wait_s"] == pytest.approx(0.03)


def test_account_ids_are_scrubbed_when_the_run_ends(fake, tmp_path, monkeypatch):
    setup, _, home = fake
    c = setup(reply='{"type": "finish"}')
    monkeypatch.setattr(clients, "make_client", lambda spec, **kw: c)
    assert cli.main(["run", "--model", "codex-cli:m", "--task", "write-mlp-0000", "--out", str(tmp_path / "out")]) == 0
    text = "".join(f.read_text() for f in (home / "sessions").rglob("*.jsonl"))
    assert "user-REAL" not in text and "acct-REAL" not in text and "<account id>" in text
    rec = json.loads((tmp_path / "out" / "write-mlp-0000.run0.json").read_text())
    assert rec["model"] == ["m"] and rec["client"] == "codex-cli:m"
