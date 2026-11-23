"""Two turns in one Codex thread (exec, then exec resume) and the rollout records each leaves, summarised."""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from wspb.harness import clients  # noqa: E402

home = clients.CODEX_HOME
(home / "config.toml").write_text(Path(sys.argv[2]).read_text())
sysf = Path(tempfile.mkdtemp(dir="/tmp")) / "i.txt"
sysf.write_text("You are a terse test endpoint. Answer in one short line.\n")
c = clients.CodexCLIClient("gpt-6.2-astra")
cwd = tempfile.mkdtemp(dir="/tmp")
t0 = time.time()


def call(args, prompt):
    p = subprocess.run([c.exe, "exec", *args, "--skip-git-repo-check", "-s", "read-only", *c._flags(str(sysf)), "--json",
                        prompt] if not args else [c.exe, "exec", "resume", "--skip-git-repo-check", *c._flags(str(sysf)),
                                                  "--json", *args, prompt],
                       cwd=cwd, env=c._env(), capture_output=True, text=True, timeout=300)
    ev = [json.loads(x) for x in p.stdout.splitlines() if x.startswith("{")]
    print("exit", p.returncode, [e.get("type") for e in ev], clients.scrub(p.stderr)[-300:])
    return ev


ev = call([], "Remember the code word HERON-58 and reply only OK.")
tid = next(e["thread_id"] for e in ev if e.get("type") == "thread.started")
call([tid], "What was the code word? One word.")
for f in sorted((home / "sessions").rglob("*.jsonl")):
    if f.stat().st_mtime < t0:
        continue
    print("==", f.name)
    for line in f.read_text().splitlines():
        e = json.loads(line)
        p = e.get("payload") or {}
        if e.get("type") == "response_item":
            c_ = p.get("content")
            text = " ".join(b.get("text", "") for b in c_) if isinstance(c_, list) else ""
            print("  item", p.get("type"), p.get("role"), [b.get("type") for b in c_] if isinstance(c_, list) else "",
                  text[:70].replace("\n", " "))
        elif e.get("type") in ("session_meta", "turn_context"):
            print(" ", e["type"], {k: p.get(k) for k in ("id", "model", "effort", "cli_version", "source") if k in p})
        else:
            print(" ", e.get("type"), p.get("type", ""))
