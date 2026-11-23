"""What the host says at start, before any message is sent: is the init event there, does it name the agent, and what
happens when the agent asked for does not exist (the host can fall back to its default agent, which has tools)?

    python probe_start.py <out dir>

Starts the host in stream mode with the session settings the client uses, waits for the first event without sending
anything, and closes stdin. No message is sent, so no model call is made."""
import json
import select
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from call import AGY, env, mask  # noqa: E402
from probe import DENY, agent_file  # noqa: E402

FRONT = ["excludeDefaultComponents: true", "inheritCustomizations: false", "inheritMcp: false"]
SETTINGS = {**DENY, "toolPermission": "strict", "useG1Credits": False, "modelProvider": ""}

if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    good = f"wspb-probe-{uuid.uuid4().hex[:8]}"
    agent_file(good, "You are a plain chat endpoint under test.", [], FRONT)
    rows = {}
    for tag, name in (("agent_exists", good), ("agent_missing", f"wspb-nonesuch-{uuid.uuid4().hex[:8]}")):
        args = ["--input-format", "stream-json", "--output-format", "stream-json", "--model", "gemini-4-neon",
                "--effort", "high", "--disable-slash-commands", "--agent", name, "--settings", json.dumps(SETTINGS)]
        err = tempfile.TemporaryFile("w+")
        t0 = time.monotonic()
        p = subprocess.Popen([str(AGY), *args], cwd=tempfile.mkdtemp(dir="/tmp"), env=env(), stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=err, text=True)
        ready, _, _ = select.select([p.stdout], [], [], 30)
        first = p.stdout.readline() if ready else ""
        waited = round(time.monotonic() - t0, 2)
        alive = p.poll() is None
        p.stdin.close()
        try:
            code = p.wait(30)
        except subprocess.TimeoutExpired:
            p.kill()
            code = "killed"
        rest = p.stdout.read()
        err.seek(0)
        try:
            e = json.loads(first)
        except json.JSONDecodeError:
            e = {"raw": first[:300]}
        init = e.get("init") or {}
        rows[tag] = {"args": args, "first_event_within_s": waited, "first_event": e.get("event"),
                     "init_agent": init.get("agent"), "init_model": init.get("model"),
                     "init_permission_mode": init.get("permission_mode"), "init_tools": len(init.get("tools") or []),
                     "alive_before_stdin_closed": alive, "exit": code, "later_output": mask(rest)[:600],
                     "stderr": mask(err.read())[-800:]}
        print(tag, json.dumps(rows[tag])[:1200])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "agy_13_start.json").write_text(json.dumps(rows, indent=1))
