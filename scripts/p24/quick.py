"""One short codex exec in the subject's home with a given config prefix; prints stderr, the reply, and from the rollout
the tools declared, the turn context and any tool calls. quick.py <worktree> <config prefix file> [prompt]"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from wspb.harness import clients  # noqa: E402

home = clients.CODEX_HOME
_t = Path(sys.argv[2]).read_text()
(home / "config.toml").write_text(_t if _t.startswith("#FULL") else _t + clients.CODEX_CONFIG)
sysf = Path(tempfile.mkdtemp(dir="/tmp")) / "i.txt"
sysf.write_text(sys.argv[4] if len(sys.argv) > 4 else "You are a plain chat endpoint under test. Begin every reply with the word ACK.\n")
c = clients.CodexCLIClient("gpt-6.2-astra")
prompt = sys.argv[3] if len(sys.argv) > 3 else "List every tool you can call by exact name, then try to spawn a sub-agent and say what happened."
t0 = time.time()
try:
    p = subprocess.run([c.exe, "exec", "--skip-git-repo-check", "-s", "read-only", *c._flags(str(sysf)), "--json", prompt],
                       cwd=tempfile.mkdtemp(dir="/tmp"), env=c._env(), capture_output=True, text=True, timeout=240)
    out, err = p.stdout, p.stderr
except subprocess.TimeoutExpired as e:
    out, err = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), "TIMEOUT " + str((e.stderr or b"")[-800:])
print("seconds", round(time.time() - t0), "stderr:", clients.scrub(err)[-700:])
for line in out.splitlines():
    if '"agent_message"' in line:
        print("reply:", clients.scrub(json.loads(line)["item"]["text"])[:900])
f = max((home / "sessions").rglob("*.jsonl"), key=lambda p: p.stat().st_mtime)
if f.stat().st_mtime >= t0:
    for line in f.read_text().splitlines():
        e = json.loads(line)
        p_ = e.get("payload") or {}
        if p_.get("type") == "additional_tools":
            print("tools:", [t.get("name") for t in p_["tools"]])
        if e.get("type") == "turn_context":
            print("ctx:", p_.get("model"), p_.get("effort"), p_.get("multi_agent_version"))
        if p_.get("type") in ("function_call", "custom_tool_call"):
            print("call:", p_.get("namespace"), p_.get("name"))
else:
    print("no new rollout")
