"""The agy-cli client against a fake `agy` executable (no network): the subject's own home, one host process per
conversation fed over its standard input, the host's record of each call read back and checked (tools, system prompt,
model, messages), each call's own token counts, replay of a history the conversation doesn't hold, the sign-in guard,
retries with their reasons, and what the run record keeps of the host's record."""

import json
import os
import sys

import pytest

from wspb import cli
from wspb.harness import clients

FAKE = r'''
import json, os, sqlite3, sys, unicodedata, uuid
home = os.environ["HOME"]
cfg = json.load(open(os.path.join(home, "fake.json")))
args = sys.argv[1:]
opt = lambda k: args[args.index(k) + 1] if k in args else None
if args == ["models"]:
    if cfg.get("signed_out"):
        print("Fetching available models...\nError: Please sign in to view available models."); sys.exit(1)
    print("Fetching available models...\n" + "\n".join(cfg.get("models", ["m-high\tM (High)", "m-low\tM (Low)"]))); sys.exit(0)
if args == ["--version"]:
    print("9.9.9"); sys.exit(0)
if args[:2] == ["-p", "/usage"]:
    print(json.dumps({"status": "SUCCESS", "command": {"name": "usage", "data": {"groups": [{"name": "Gemini Models", "buckets": [
        {"id": "gemini-weekly", "remaining_fraction": 0.5, "reset_time": "2026-11-25T00:00:00Z"},
        {"id": "gemini-5h", "remaining_fraction": 0.9, "reset_time": "2026-11-21T00:00:00Z"}]}]}}})); sys.exit(0)
assert opt("--input-format") == "stream-json" and opt("--output-format") == "stream-json"
emit = lambda **e: print(json.dumps(e), flush=True)
agent = opt("--agent")
path = os.path.join(home, ".gemini", "config", "agents", agent + ".md")
if not os.path.exists(path):
    emit(event="result", result={"status": "ERROR", "error": f"unknown agent {agent!r}"}); sys.exit(1)
front, system = open(path).read().split("\n---\n", 1)
system = system[:-1]


def vi(n):
    out = b""
    while True:
        out += bytes([(n & 0x7F) | (0x80 if n >> 7 else 0)])
        n >>= 7
        if not n:
            return out


def f(num, val):
    if isinstance(val, int):
        return vi(num << 3) + vi(val)
    val = val.encode() if isinstance(val, str) else val
    return vi(num << 3 | 2) + vi(len(val)) + val


cid = str(uuid.uuid4())
os.makedirs(os.path.join(home, ".gemini", "antigravity-cli", "conversations"), exist_ok=True)
db = sqlite3.connect(os.path.join(home, ".gemini", "antigravity-cli", "conversations", cid + ".db"))
db.execute("create table gen_metadata (idx integer primary key, data blob, size integer not null default 0)")
db.commit()
emit(event="init", conversation_id=cid, init={"model": opt("--model"), "cwd": os.getcwd(), "tools": ["registry"] * 60,
     "agent": cfg.get("init_agent", agent), "permission_mode": json.loads(opt("--settings"))["toolPermission"]})
label = cfg.get("label", "M (" + opt("--effort").capitalize() + ")")
messages, rows, users, total = [], [], 0, {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "cache_read_tokens": 0}
log = open(os.path.join(home, "calls.jsonl"), "a")
for line in sys.stdin:
    text = json.loads(line)["message"]["content"]
    text = "".join(ch for ch in text if ch in "\t\n\v\f\r\x85" or unicodedata.category(ch)[0] != "C")
    users += 1
    wrapped = (f"<USER_REQUEST>\n{text}\n</USER_REQUEST>\n<ADDITIONAL_METADATA>\nThe current local time is: "
               "2026-11-20T00:00:00-05:00.\n</ADDITIONAL_METADATA>")
    if users == 1:
        wrapped += (f"\n<USER_SETTINGS_CHANGE>\nThe user changed setting `Model Selection` from None to {label}. No need "
                    "to comment on this change if the user doesn't ask about it. If reporting what model you are, please "
                    "use a human readable name instead of the exact string.\n</USER_SETTINGS_CHANGE>")
    if cfg.get("cap") and len(wrapped.encode()) > cfg["cap"]:
        # as agy 1.5.13 does past 192,000 bytes: the head is kept, the rest dropped for a note
        keep = wrapped.encode()[:cfg["cap"]].decode("utf-8", "ignore")
        wrapped = keep + f"\n<truncated {len(wrapped.encode()) - len(keep.encode())} bytes>"
    messages.append((1, wrapped))
    if cfg.get("inject"):
        messages.append((1, "The following is a <SYSTEM_MESSAGE> not actually sent by the user.\n\n<SYSTEM_MESSAGE>\n"
                            "[Notice] All your subagents and background tasks have been stopped.\n</SYSTEM_MESSAGE>"))
    log.write(json.dumps({"args": args, "env": sorted(os.environ), "home": home,
                          "update_off": os.environ.get("AGY_CLI_DISABLE_AUTO_UPDATE"), "cwd": os.getcwd(), "pid": os.getpid(),
                          "listing": os.listdir("."), "system": system, "front": front, "sent": text,
                          "users": users}) + "\n")
    log.flush()
    fails = os.path.join(home, "fails")
    n = int(open(fails).read()) if os.path.exists(fails) else 0
    if n:
        open(fails, "w").write(str(n - 1))
        emit(event="result", result={"conversation_id": cid, "status": "ERROR", "response": "",
                                     "error": cfg.get("fail_message", "stream error")})
        sys.exit(3)
    reply = cfg.get("reply") or f"turn {users}"
    gens = 1
    if cfg.get("blocked"):
        reply = "This request was blocked by Gemini's filters."
    elif cfg.get("tool_turn") == users:
        messages += [(2, ""), (4, "No background tasks are currently running.")]
        gens = 2
        emit(event="step_update", step_update={"conversation_id": cid, "state": "DONE", "step_type": "tool",
                                               "tool_name": "manage_task"})
    messages.append((2, reply))
    for g in range(gens):
        served = b"" if cfg.get("blocked") else f(19, cfg.get("backend", "m"))
        usage = b"" if cfg.get("blocked") else f(4, f(2, 40) + f(3, 7) + f(5, 30) + f(9, 3))
        rows.append(usage + served + f(21, label))
        if not cfg.get("blocked"):
            for k, v in (("input_tokens", 40), ("output_tokens", 7), ("thinking_tokens", 3), ("cache_read_tokens", 30)):
                total[k] += v
            emit(event="step_update", step_update={"conversation_id": cid, "state": "DONE", "step_type": "agent_response",
                 "usage": {"input_tokens": 40, "output_tokens": 7, "thinking_tokens": 3, "cache_read_tokens": 30}})
    prompt = f(1, "<System Prompt>\n" + system + cfg.get("system_extra", "") + "\n</System Prompt>")
    prompt += b"".join(f(2, f(2, role) + (f(3, t) if t else b"")) for role, t in messages)
    prompt += b"".join(f(8, f(1, t) + f(2, "what it does")) for t in cfg.get("tools", ["manage_task"]))
    db.execute("delete from gen_metadata")   # only the newest call keeps its prompt
    for i, body in enumerate(rows):
        db.execute("insert into gen_metadata (idx, data) values (?, ?)", (i, f(1, (prompt if i == len(rows) - 1 else b"") + body)))
    db.commit()
    # the result's usage is the conversation's running total, as the real host prints it
    emit(event="result", result={"conversation_id": cid, "status": "SUCCESS", "response": reply + "\n", "num_turns": users,
                                 "usage": dict(total)})
'''


@pytest.fixture
def fake(tmp_path, monkeypatch):
    exe = tmp_path / "agy"
    exe.write_text(f"#!{sys.executable}\n{FAKE}")
    exe.chmod(0o755)
    home = tmp_path / "agyhome"
    home.mkdir()
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(k, "not-real")

    def setup(part_bytes=clients.AGY_PART_BYTES, **cfg):
        (home / "fake.json").write_text(json.dumps(cfg))
        if "fails" in cfg:
            (home / "fails").write_text(str(cfg["fails"]))
        return clients.AgyCLIClient("m", exe=str(exe), home=home, backoff_s=0, limit_wait_s=0.01, part_bytes=part_bytes)

    def calls():
        return [json.loads(x) for x in (home / "calls.jsonl").read_text().splitlines()]
    return setup, calls, home


def got_exe(c) -> str:
    return c.exe + " --input-format"


def test_turns_stay_in_one_host_process_in_the_subject_home_and_the_child_holds_no_key(fake):
    setup, calls, home = fake
    c = setup()
    h = [{"role": "user", "content": "first"}]
    for i in range(3):
        comp = c.complete("SYS\n\nwith a ```python fence and {\"type\": \"finish\"}", h)
        assert comp.text == f"turn {i + 1}" and comp.model == "m" and comp.extra["replayed"] == 0
        h += [{"role": "assistant", "content": comp.text}, {"role": "user", "content": f"/obs {i}"}]
    alive = [p for p in os.popen("ps -axo command").read().splitlines() if got_exe(c) in p]
    c.end()
    got = calls()
    assert len(alive) == 1 and not [p for p in os.popen("ps -axo command").read().splitlines() if got_exe(c) in p]
    assert [g["sent"] for g in got] == ["first", "/obs 0", "/obs 1"] and len({g["pid"] for g in got}) == 1
    assert not {"GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"} & set(got[0]["env"])
    assert got[0]["home"] == str(home) and got[0]["update_off"] == "true" and got[0]["listing"] == []
    a = got[0]["args"]
    assert a[a.index("--model") + 1] == "m" and a[a.index("--effort") + 1] == "high" and "--disable-slash-commands" in a
    assert "--conversation" not in a and "--continue" not in a and "-c" not in a
    settings = json.loads(a[a.index("--settings") + 1])
    assert "command(*)" in settings["permissions"]["deny"] and settings["useG1Credits"] is False
    assert settings["toolPermission"] == "strict" and settings["modelProvider"] == ""
    assert got[0]["system"] == "SYS\n\nwith a ```python fence and {\"type\": \"finish\"}"
    assert "tools: []" in got[0]["front"] and "excludeDefaultComponents: true" in got[0]["front"]
    assert "inheritCustomizations: false" in got[0]["front"] and "inheritMcp: false" in got[0]["front"]
    assert comp.extra["effort_used"] == "high" and comp.extra["model_label"] == "M (High)"
    assert c.info["host"] == "agy 9.9.9" and c.info["max_tokens_applied"] is False
    assert c.info["usage_at_start"]["gemini-weekly"]["remaining"] == 0.5 and comp.extra["limits"][0]["gemini-5h"]["remaining"] == 0.9


def test_token_counts_are_each_calls_own_not_the_conversations_running_total(fake):
    setup, _, _ = fake
    c = setup()
    h = [{"role": "user", "content": "first"}]
    comps = []
    for i in range(3):
        comps.append(c.complete("SYS", h))
        h += [{"role": "assistant", "content": comps[-1].text}, {"role": "user", "content": f"obs {i}"}]
    # the fake's result events carry 70, 140, 210 tokens in (running totals); each call's own count is 40 + 30 cached
    assert [x.tokens_in for x in comps] == [70, 70, 70] and [x.tokens_out for x in comps] == [7, 7, 7]
    assert all(x.extra["cache_read"] == 30 and x.extra["uncached_in"] == 40 and x.extra["thinking"] == 3 for x in comps)
    assert all(x.extra["usage_mismatch"] == 0 and x.extra["generations"] == 1 for x in comps)


def test_a_turn_with_a_tool_call_counts_both_model_calls_and_the_tool_call(fake):
    setup, _, _ = fake
    c = setup(tool_turn=2)
    h = [{"role": "user", "content": "first"}]
    a = c.complete("SYS", h)
    h += [{"role": "assistant", "content": a.text}, {"role": "user", "content": "second"}]
    b = c.complete("SYS", h)
    assert (a.extra["tool_calls"], a.extra["generations"], a.tokens_in) == (0, 1, 70)
    assert (b.extra["tool_calls"], b.extra["generations"], b.tokens_in, b.tokens_out) == (1, 2, 140, 14)
    assert b.text == "turn 2" and b.extra["host"]["declared_tools"] == ["manage_task"]
    h += [{"role": "assistant", "content": b.text}, {"role": "user", "content": "third"}]
    assert c.complete("SYS", h).extra["replayed"] == 0   # the tool exchange is part of what the conversation holds


def test_a_history_the_conversation_does_not_hold_is_replayed_into_a_fresh_process(fake):
    setup, calls, _ = fake
    c = setup()
    h = [{"role": "user", "content": "first " * 40}]
    a = c.complete("SYS", h)
    h += [{"role": "assistant", "content": a.text}, {"role": "user", "content": "second"}]
    c.complete("SYS", h)
    trimmed = [h[0], {"role": "assistant", "content": "(empty)"}, {"role": "user", "content": "third"}]
    b = c.complete("SYS", trimmed)
    got = calls()
    assert b.extra["replayed"] == 1 and b.extra["session"] != a.extra["session"] and got[-1]["pid"] != got[0]["pid"]
    assert got[-1]["sent"] == clients.agy_flat(trimmed) and got[-1]["users"] == 1
    assert "===== assistant =====\n(empty)" in got[-1]["sent"] and got[-1]["sent"].endswith("===== user =====\nthird")
    assert "system" in b.extra["host"] and "system" in a.extra["host"]   # each fresh conversation's instructions are kept
    more = trimmed + [{"role": "assistant", "content": b.text}, {"role": "user", "content": "fourth"}]
    d = c.complete("SYS", more)
    assert d.extra["replayed"] == 0 and calls()[-1]["sent"] == "fourth" and "system" not in d.extra["host"]


def _rebuilt(parts: list[str]) -> str:
    """The messages of a replay in parts, read back: a part that goes on with a message joins it directly."""
    out = parts[0].split("\n\n", 1)[1]
    for p in parts[1:]:
        head, chunk = p.split("\n\n", 1)
        out += ("" if "going on with the message" in head else "\n\n") + chunk
    return out


def _long_history(turns: int, size: int) -> list[dict[str, str]]:
    h = [{"role": "user", "content": "the task " * 60}]
    for i in range(turns):
        h += [{"role": "assistant", "content": f"reply {i} " + "r" * size}, {"role": "user", "content": f"obs {i} " + "o" * size}]
    return h


def test_a_replay_past_the_hosts_cap_on_a_message_fails_every_try_when_sent_as_one(fake):
    # pilot 25's four stops: the harness's history cap dropped the oldest exchange, the replay was longer than the
    # host keeps of a message, and the cut record matched none of the five tries
    setup, calls, _ = fake
    c = setup(cap=3000, part_bytes=10 ** 9)
    with pytest.raises(clients.InfraError, match="one is not ours.*truncated"):
        c.complete("SYS", _long_history(6, 300))
    assert len(calls()) == 5 and all(g["users"] == 1 for g in calls())
    # a run whose replay in parts failed says so, for the runner's gate
    with pytest.raises(clients.InfraError, match=r"5 attempts failed \(a replay in parts was tried\); last: status ERROR"):
        setup(fails=9, part_bytes=2500).complete("SYS", _long_history(6, 300))


def test_a_replay_past_the_hosts_cap_goes_in_parts_each_answered_and_all_checked(fake):
    setup, calls, _ = fake
    c = setup(cap=3000, part_bytes=2500)
    h = _long_history(6, 300)
    comp = c.complete("SYS", h)
    got = calls()
    n = comp.extra["replay_parts"]
    assert n == len(got) >= 2 and len({g["pid"] for g in got}) == 1 and [g["users"] for g in got] == list(range(1, n + 1))
    assert all(len(g["sent"].encode()) <= 2500 for g in got) and comp.extra["retries"] == 0 and not comp.extra["failed_tries"]
    assert comp.text == f"turn {n}" and comp.extra["replay_part_replies"] == [f"turn {i}" for i in range(1, n)]
    assert comp.extra["replayed"] == 1 and comp.extra["generations"] == n and comp.tokens_in == 70 * n
    assert got[0]["sent"].startswith(clients.AGY_PARTS.format(n=n)) and "reply with OK alone" in got[0]["sent"]
    assert got[-1]["sent"].startswith(f"[Part {n} of {n} of the replay") and "Continue from the last message." in got[-1]["sent"]
    # the parts, less their first lines, are the one-message replay's messages, every character of them
    assert _rebuilt([g["sent"] for g in got]) == clients.agy_flat(h)[len(clients.AGY_REPLAY) + 2:]
    # the next turn goes on in the same conversation, after the parts
    more = h + [{"role": "assistant", "content": comp.text}, {"role": "user", "content": "next"}]
    d = c.complete("SYS", more)
    assert d.extra["replayed"] == 0 and "replay_parts" not in d.extra and calls()[-1]["sent"] == "next"


def test_agy_parts_breaks_between_messages_where_it_can_and_says_when_a_message_goes_on():
    short = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert clients.agy_parts(short) == [clients.agy_flat(short)]
    h = [{"role": "user", "content": "x" * 900}, {"role": "assistant", "content": "y" * 900},
         {"role": "user", "content": ("line " * 30 + "\n") * 40}]
    parts = clients.agy_parts(h, part_bytes=1500)
    assert len(parts) >= 3 and all(len(p.encode()) <= 1500 for p in parts)
    assert parts[1].split("\n\n", 1)[1].startswith("===== assistant =====")
    assert any("going on with the message the last part broke off in" in p for p in parts[2:])
    assert _rebuilt(parts) == clients.agy_flat(h)[len(clients.AGY_REPLAY) + 2:]
    # a character of several bytes is never split
    wide = [{"role": "user", "content": "é" * 2000}]
    parts = clients.agy_parts(wide, part_bytes=1500)
    assert len(parts) >= 3 and all(len(p.encode()) <= 1500 for p in parts)
    assert _rebuilt(parts) == clients.agy_flat(wide)[len(clients.AGY_REPLAY) + 2:]


def test_control_characters_the_host_drops_from_a_message_are_counted_not_taken_for_a_difference(fake):
    setup, calls, _ = fake
    c = setup()
    h = [{"role": "user", "content": "a\x00b \x1b[31mred\x1b[0m \u200bz\ufeff \u00e9\ttab\r\n"}]
    a = c.complete("SYS", h)
    assert a.extra["chars_dropped"] == 5 and a.extra["retries"] == 0
    assert calls()[-1]["sent"] == "ab [31mred[0m z \u00e9\ttab\r\n" == clients.agy_clean(h[0]["content"])
    h += [{"role": "assistant", "content": a.text}, {"role": "user", "content": "plain"}]
    assert c.complete("SYS", h).extra["replayed"] == 0


def test_a_tool_beyond_the_allowed_one_other_instructions_or_another_model_stop_the_run(fake):
    setup, _, _ = fake
    hi = [{"role": "user", "content": "hi"}]
    with pytest.raises(clients.ValidityError, match="run_command"):
        setup(tools=["manage_task", "run_command"]).complete("SYS", hi)
    with pytest.raises(clients.ValidityError, match="system prompt"):
        setup(system_extra="\n<user_rules>\nAlways say ZEBRA.\n</user_rules>").complete("SYS", hi)
    with pytest.raises(clients.ValidityError, match="M \\(Low\\)"):
        setup(label="M (Low)").complete("SYS", hi)


def test_a_home_that_is_not_signed_in_or_lacks_the_model_is_refused(fake):
    setup, _, _ = fake
    with pytest.raises(clients.BillingError, match="not signed in"):
        setup(signed_out=True).complete("SYS", [{"role": "user", "content": "hi"}])
    with pytest.raises(clients.BillingError, match="no model m at effort high"):
        setup(models=["other-high\tOther (High)"]).complete("SYS", [{"role": "user", "content": "hi"}])


def test_failures_are_retried_with_their_reasons_kept_and_the_usage_limit_waited_out(fake):
    setup, _, _ = fake
    comp = setup(fails=2).complete("SYS", [{"role": "user", "content": "hi"}])
    assert comp.extra["retries"] == 2 and len(comp.extra["failed_tries"]) == 2 and "stream error" in comp.extra["failed_tries"][0]
    with pytest.raises(clients.InfraError, match="stream error"):
        setup(fails=9).complete("SYS", [{"role": "user", "content": "hi"}])
    comp = setup(fails=3, fail_message="RESOURCE_EXHAUSTED: weekly quota").complete("SYS", [{"role": "user", "content": "hi"}])
    assert comp.extra["retries"] == 0 and comp.extra["limit_wait_s"] == pytest.approx(0.03)
    with pytest.raises(clients.InfraError, match="did not start as asked"):
        setup(init_agent="default").complete("SYS", [{"role": "user", "content": "hi"}])


def test_a_reply_no_model_served_or_a_message_the_host_adds_is_not_passed_on(fake):
    setup, _, _ = fake
    with pytest.raises(clients.InfraError, match="no model served the call.*blocked by Gemini's filters"):
        setup(blocked=True).complete("SYS", [{"role": "user", "content": "hi"}])
    with pytest.raises(clients.InfraError, match="messages that differ.*one is not ours.*SYSTEM_MESSAGE"):
        setup(inject=True).complete("SYS", [{"role": "user", "content": "hi"}])


def test_the_run_record_keeps_the_hosts_record_of_each_call_and_no_path_of_the_home(fake, tmp_path, monkeypatch):
    setup, _, home = fake
    c = setup(reply='{"type": "finish"}')
    monkeypatch.setattr(clients, "make_client", lambda spec, **kw: c)
    assert cli.main(["run", "--model", "agy-cli:m", "--task", "write-mlp-0000", "--out", str(tmp_path / "out")]) == 0
    raw = (tmp_path / "out" / "write-mlp-0000.run0.json").read_text()
    rec = json.loads(raw)
    assert rec["model"] == ["m"] and rec["client"] == "agy-cli:m"
    host = rec["log"][0]["call"]["host"]
    assert host["declared_tools"] == ["manage_task"] and host["system"].startswith("<System Prompt>\nYou are working on")
    assert host["added"].startswith("<USER_REQUEST>\n<message>\n</USER_REQUEST>\n<ADDITIONAL_METADATA>")
    assert rec["tokens_in"] == 70 and rec["tokens_out"] == 7
    summary = (tmp_path / "out" / "summary.json").read_text()
    assert str(home) not in raw + summary and json.loads(summary)["client"]["home"] == "<agy home>"
    # the run's host process was closed when the run ended
    assert not [p for p in os.popen("ps -axo command").read().splitlines() if str(tmp_path / "agy") + " " in p]
