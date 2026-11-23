"""Which agent-definition lines, settings or flags change the tools the host declares: one short call per variant,
the tools read from the host's record of the call.

    python variants.py <out dir>"""
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from call import call, mask  # noqa: E402
from probe import DENY, agent_file, events, summary  # noqa: E402
from record import calls  # noqa: E402

BASE = ["excludeDefaultComponents: true", "inheritCustomizations: false", "inheritMcp: false"]
VARIANTS = {
    "base": (BASE, None, []),
    "no_exclude": (["inheritCustomizations: false", "inheritMcp: false"], None, []),
    "policy_off": (BASE + ["commandExecutionPolicy: off"], None, []),
    "background_false": (BASE + ["background: false"], None, []),
    "hidden": (BASE + ["hidden: true"], None, []),
    "deny_all": (BASE, json.dumps(DENY), []),
    "strict": (BASE, json.dumps({**DENY, "toolPermission": "strict", "agentMode": ""}), []),
    "plan_mode": (BASE, None, ["--mode", "plan"]),
    "sandbox": (BASE, None, ["--sandbox"]),
    "no_slash": (BASE, None, ["--disable-slash-commands"]),
}

if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    only = sys.argv[2:]
    rows = {}
    for tag, (front, settings, extra) in VARIANTS.items():
        if only and tag not in only:
            continue
        name = f"wspb-probe-{uuid.uuid4().hex[:8]}"
        agent_file(name, "You are a plain chat endpoint under test.", [], front)
        r = call(["-p", "Reply with the single word: pong", "--output-format", "stream-json", "--model",
                  "gemini-4-neon", "--effort", "high", "--agent", name, *(["--settings", settings] if settings else []),
                  *extra], timeout=300)
        s = summary(events(r["stdout"]))
        rec = calls(s["conversation"]) if s["conversation"] else []
        rows[tag] = {"front": front, "settings": settings, "extra": extra, "exit": r["exit"],
                     "stderr": r["stderr"][-400:], "conversation": s["conversation"],
                     "permission_mode": (s["init"] or {}).get("permission_mode"),
                     "reply": (s["result"] or {}).get("response"),
                     "tools": [[t["name"] for t in c["tools"]] for c in rec],
                     "system": [mask(c["system"]) for c in rec], "usage": [c["usage"] for c in rec],
                     "model": [(c["model"], c["model_label"]) for c in rec]}
        print(tag, json.dumps({k: v for k, v in rows[tag].items() if k not in ("front", "system")})[:700])
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / "agy_05_variants.json"
    old = json.loads(f.read_text()) if f.exists() else {}
    f.write_text(json.dumps({**old, **rows}, indent=1))
