"""The claude-cli client against a fake `claude` executable (no network): what the child gets, that each turn is a
real turn of one session, that a history the host doesn't hold is replayed into a fresh one, retries, the output cap,
and the billing stops."""

import json
import os
import sys

import pytest

from wspb import cli
from wspb.harness import clients

FAKE = r'''
import json, os, re, sys
home = os.environ["HOME"]
cfg = json.load(open(os.path.join(home, "fake.json")))
args = sys.argv[1:]
if args[:2] == ["auth", "status"]:
    print(json.dumps({"loggedIn": True, "authMethod": cfg.get("auth", "claude.ai"), "apiProvider": "firstParty",
                      "subscriptionType": "max", "email": "Someone.Real@Example.test"}))
    sys.exit(0)
if args == ["--version"]:
    print("9.9.9 (fake)")
    sys.exit(0)
opt = lambda k: args[args.index(k) + 1] if k in args else None
sid = opt("--resume") or opt("--session-id")
prompt = sys.stdin.read()
proj = os.path.join(home, ".claude", "projects", re.sub(r"[^A-Za-z0-9]", "-", os.path.realpath(os.getcwd())))
os.makedirs(proj, exist_ok=True)
path = os.path.join(proj, sid + ".jsonl")
chain = []
if opt("--resume"):
    for line in open(path):
        e = json.loads(line)
        if e.get("type") in ("user", "assistant"):
            c = e["message"]["content"]
            chain.append({"role": e["type"], "content": c if isinstance(c, str) else "".join(b["text"] for b in c)})
seen = chain + [{"role": "user", "content": prompt}]
with open(os.path.join(home, "calls.jsonl"), "a") as f:
    f.write(json.dumps({"argv": args, "env": dict(os.environ), "seen": seen, "cwd": os.getcwd()}) + "\n")

def emit(**e):
    print(json.dumps(e), flush=True)

def record(kind, content, **extra):
    msg = {"role": kind, "content": content} if kind == "user" else {
        "model": "fake-served", "id": extra.pop("mid", "m1"), "role": "assistant", "content": [{"type": "text", "text": content}]}
    with open(path, "a") as f:
        f.write(json.dumps({"type": kind, "message": msg, "sessionId": sid}) + "\n")
        if not opt("--resume") and kind == "user":
            f.write(json.dumps({"type": "attachment", "attachment": {"type": "date"}, "rendered": [{"content": "today"}]}) + "\n")
            f.write(json.dumps({"type": "attachment", "attachment": {"type": "session_context"}, "rendered": [{
                "content": "The user's email address is someone.real@example.test.\n - Primary working directory: "
                           + os.getcwd() + "\n - Platform: darwin"}]}) + "\n")

emit(type="system", subtype="init", tools=[], mcp_servers=[], apiKeySource=cfg.get("keysource", "none"), session_id=sid)
record("user", prompt)
state = os.path.join(home, "fails")
fails = int(open(state).read()) if os.path.exists(state) else 0
if fails:
    open(state, "w").write(str(fails - 1))
    emit(type="assistant", message={"model": "<synthetic>", "id": "x", "content": [{"type": "text", "text": "API Error"}]})
    emit(type="result", is_error=True, result="API Error: 529 Overloaded", api_error_status=529)
    sys.exit(1)
limits = os.path.join(home, "limits")
left = int(open(limits).read()) if os.path.exists(limits) else 0
if left:
    open(limits, "w").write(str(left - 1))
    emit(type="rate_limit_event", rate_limit_info={"status": "rejected", "resetsAt": 1794834000})
    emit(type="result", is_error=True, result="You've hit your limit")
    sys.exit(1)
usage = {"input_tokens": sum(len(m["content"]) for m in seen) // 4, "cache_read_input_tokens": 2,
         "cache_creation_input_tokens": 1, "output_tokens": 7, "output_tokens_details": {"thinking_tokens": 3}}
if "LONG" in prompt:
    emit(type="assistant", message={"model": "fake-served", "id": "m1", "content": [{"type": "text", "text": "partial"}],
                                    "usage": usage}, request_id="r1")
    record("assistant", "partial")
    record("user", "Output token limit hit. Resume directly")
    emit(type="user", message={"role": "user", "content": [{"type": "text", "text": "Output token limit hit."}]})
    emit(type="assistant", message={"model": "fake-served", "id": "m2", "content": [{"type": "text", "text": "more"}]})
    sys.exit(0)
reply = cfg.get("reply") or f"turn {sum(m['role'] == 'user' for m in seen)}"
emit(type="assistant", message={"model": "fake-served", "id": "m1", "content": [{"type": "text", "text": reply}],
                                "usage": usage}, request_id="r1")
record("assistant", reply)
emit(type="result", is_error=False, result=reply, stop_reason="end_turn", usage=usage,
     total_cost_usd=0.5 * sum(m["role"] == "user" for m in seen), modelUsage={"fake-served": {}}, duration_api_ms=5)
'''


@pytest.fixture
def fake(tmp_path, monkeypatch):
    exe = tmp_path / "claude"
    exe.write_text(f"#!{sys.executable}\n{FAKE}")
    exe.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-not-real")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-real")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "not-real")

    def setup(client=None, **cfg):
        (home / "fake.json").write_text(json.dumps(cfg))
        for k in ("fails", "limits"):
            if k in cfg:
                (home / k).write_text(str(cfg[k]))
        return clients.ClaudeCLIClient("m", exe=str(exe), backoff_s=0, limit_wait_s=0.01, **(client or {}))

    def calls():
        return [json.loads(x) for x in (home / "calls.jsonl").read_text().splitlines()]
    return setup, calls


def test_each_turn_is_a_resumed_turn_of_one_session_and_the_child_holds_no_key(fake):
    setup, calls = fake
    c = setup()
    h = [{"role": "user", "content": "first"}]
    for i in range(3):
        comp = c.complete("SYS", h)
        assert comp.text == f"turn {i + 1}" and comp.model == "fake-served" and comp.extra["replayed"] == 0
        h += [{"role": "assistant", "content": comp.text}, {"role": "user", "content": f"obs {i}"}]
    got = calls()
    assert [g["seen"] for g in got] == [h[:1], h[:3], h[:5]]          # what it saw is exactly what was passed
    assert "--session-id" in got[0]["argv"] and all("--resume" in g["argv"] for g in got[1:])
    assert len({g["cwd"] for g in got}) == 1 and "WeightSpace" not in got[0]["cwd"]
    argv, env = got[0]["argv"], got[0]["env"]
    assert argv[argv.index("--system-prompt") + 1] == "SYS" and argv[argv.index("--tools") + 1] == ""
    assert {"--safe-mode", "--restricted", "--strict-mcp-config"} <= set(argv) and argv[argv.index("--effort") + 1] == "high"
    assert not {"ANTHROPIC_API_KEY", "OPENAI_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"} & set(env)
    assert all(k in clients.CLI_ENV or k in ("TMPDIR", "__CF_USER_TEXT_ENCODING") or k.startswith("CLAUDE_CODE_DISABLE_")
               for k in env if k != "CLAUDE_CODE_MAX_OUTPUT_TOKENS") and env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "32000"
    assert comp.extra["cost_usd"] == 0.5 and comp.tokens_in == comp.extra["uncached_in"] + 3 and comp.tokens_out == 7
    assert c.info["host"] == "9.9.9 (fake)" and c.info["auth"] == "claude.ai"


def test_a_history_the_session_does_not_hold_is_replayed_into_a_fresh_one(fake):
    setup, calls = fake
    c = setup()
    h = [{"role": "user", "content": "first " * 50}]
    a = c.complete("SYS", h)
    h += [{"role": "assistant", "content": a.text}, {"role": "user", "content": "second"}]
    c.complete("SYS", h)
    trimmed = [h[0], {"role": "assistant", "content": "(empty)"}, {"role": "user", "content": "third"}]  # the cap's cut
    b = c.complete("SYS", trimmed)
    got = calls()[-1]
    assert b.extra["replayed"] == 1 and got["seen"] == trimmed and got["argv"][got["argv"].index("--resume") + 1] != a.extra["session"]


def test_overloads_are_retried_and_five_failures_end_the_run(fake):
    setup, calls = fake
    c = setup(fails=2)
    h = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}, {"role": "user", "content": "y"}]
    comp = c.complete("SYS", h)
    assert comp.extra["retries"] == 2 and calls()[-1]["seen"] == h      # a failed turn left in the session is not seen
    c = setup(fails=5)
    with pytest.raises(clients.InfraError):
        c.complete("SYS", [{"role": "user", "content": "hi"}])

def test_the_usage_limit_is_waited_out_without_spending_a_try_and_given_up_after_its_allowance(fake):
    setup, calls = fake
    comp = setup(limits=7).complete("SYS", [{"role": "user", "content": "hi"}])
    assert comp.text == "turn 1" and comp.extra["retries"] == 0 and comp.extra["limit_wait_s"] == pytest.approx(0.07)
    c = setup(limits=100, client={"limit_for_s": 0.05})
    with pytest.raises(clients.UsageLimit, match="resetsAt 1794834000"):
        c.complete("SYS", [{"role": "user", "content": "hi"}])
    assert isinstance(clients.UsageLimit("x"), clients.InfraError)


def test_a_reply_cut_at_the_cap_is_returned_cut_and_the_host_s_continuation_is_dropped(fake):
    setup, calls = fake
    c = setup()
    h = [{"role": "user", "content": "LONG please"}]
    comp = c.complete("SYS", h)
    assert (comp.text, comp.stop_reason, comp.tokens_out) == ("partial", "max_tokens", 32000)
    h += [{"role": "assistant", "content": comp.text}, {"role": "user", "content": "next"}]
    nxt = c.complete("SYS", h)
    assert nxt.extra["replayed"] == 1 and calls()[-1]["seen"] == h


def test_an_api_key_login_stops_the_run(fake):
    setup, _ = fake
    with pytest.raises(clients.BillingError):
        setup(keysource="ANTHROPIC_API_KEY").complete("SYS", [{"role": "user", "content": "hi"}])
    with pytest.raises(clients.BillingError):
        setup(auth="console").complete("SYS", [{"role": "user", "content": "hi"}])


def test_wspb_run_records_the_served_model_and_the_host(fake, tmp_path, monkeypatch):
    setup, _ = fake
    c = setup(reply='{"type": "finish"}')
    monkeypatch.setattr(clients, "make_client", lambda spec, **kw: c)
    out = tmp_path / "out"
    assert cli.main(["run", "--model", "claude-cli:m", "--task", "write-mlp-0000", "--out", str(out)]) == 0
    rec = json.loads((out / "write-mlp-0000.run0.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    assert rec["model"] == ["fake-served"] and rec["client"] == "claude-cli:m" and rec["wall_s"] >= 0
    assert rec["log"][-1]["call"]["model"] == "fake-served" and rec["usage"]["cost_usd"] == 0.5
    assert summary["model"] == "claude-cli:m" and summary["client"]["host"] == "9.9.9 (fake)"
    assert summary["infra_failures"] == []
    (tmp_path / "home" / "fails").write_text("9")
    assert cli.main(["run", "--model", "claude-cli:m", "--task", "write-mlp-0000", "--out", str(out)]) == 0
    assert "infra_error" in json.loads((out / "write-mlp-0000.run0.infra.json").read_text())
    assert len(json.loads((out / "summary.json").read_text())["infra_failures"]) == 1
    assert os.path.exists(out / "write-mlp-0000.run0.json")       # the earlier graded run is kept


def test_no_file_keeps_the_account_email_or_the_working_directory(fake, tmp_path, monkeypatch):
    setup, calls = fake
    c = setup(reply='I am SOMEONE.real@example.test, cwd as told. {"type": "finish"}')
    monkeypatch.setattr(clients, "make_client", lambda spec, **kw: c)
    out = tmp_path / "out"
    h = [{"role": "user", "content": "first"}]
    a = c.complete("SYS", h)
    b = c.complete("SYS", h + [{"role": "assistant", "content": a.text}, {"role": "user", "content": "again"}])
    assert b.extra["replayed"] == 0             # resumed: the live record is left as the host wrote it
    c.end()                                     # wspb run ends each run's conversation itself
    assert cli.main(["run", "--model", "claude-cli:m", "--task", "write-mlp-0000", "--out", str(out)]) == 0
    files = [p for p in (tmp_path / "home" / ".claude").rglob("*.jsonl")] + list(out.iterdir())
    text = "".join(p.read_text() for p in files).lower()
    assert "someone.real@example.test" not in text and clients.EMAIL_PLACEHOLDER in text
    assert calls()[0]["cwd"].lower() not in text.replace(calls()[0]["cwd"].lower() + '"', "")   # only the record's own cwd field
    assert "primary working directory: " + clients.WORKDIR_PLACEHOLDER in text
