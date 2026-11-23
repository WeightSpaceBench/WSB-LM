"""How the model and effort are pinned, and what the host records as served: one short call per way of asking.

    python probe_model.py <out dir>"""
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from call import call  # noqa: E402
from probe import agent_file, events, summary, transcript  # noqa: E402
from record import calls  # noqa: E402

FRONT = ["excludeDefaultComponents: true", "inheritCustomizations: false", "inheritMcp: false"]
WAYS = {
    "model+effort_high": ["--model", "gemini-4-neon", "--effort", "high"],
    "model+effort_low": ["--model", "gemini-4-neon", "--effort", "low"],
    "model+effort_xhigh": ["--model", "gemini-4-neon", "--effort", "xhigh"],
    "model+effort_max": ["--model", "gemini-4-neon", "--effort", "max"],
    "model_only": ["--model", "gemini-4-neon"],
    "slug_high": ["--model", "gemini-4-neon-high"],
    "slug_low+effort_high": ["--model", "gemini-4-neon-low", "--effort", "high"],
    "no_model": [],
    "unknown_model": ["--model", "gemini-4-nonesuch", "--effort", "high"],
    "other_model": ["--model", "gemini-3.8-flash", "--effort", "high"],
}

if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    only = sys.argv[2:]
    rows = {}
    for tag, flags in WAYS.items():
        if only and tag not in only:
            continue
        name = f"wspb-probe-{uuid.uuid4().hex[:8]}"
        agent_file(name, "You are a plain chat endpoint under test.", [], FRONT)
        r = call(["-p", "Reply with the single word: pong", "--output-format", "stream-json", "--agent", name, *flags],
                 timeout=300)
        s = summary(events(r["stdout"]))
        cid = s["conversation"]
        rec = calls(cid) if cid else []
        rows[tag] = {"flags": flags, "exit": r["exit"], "stderr": r["stderr"][-500:],
                     "init_model": (s["init"] or {}).get("model"), "status": (s["result"] or {}).get("status"),
                     "error": (s["result"] or {}).get("error"), "reply": (s["result"] or {}).get("response"),
                     "record": [(c["model"], c["model_label"], c["usage"]) for c in rec],
                     "transcript_models": [t.get("model") for t in transcript(cid) if t.get("model")] if cid else [],
                     "first_message_tail": [m["text"][-330:] for c in rec[-1:] for m in c["messages"][:1]]}
        print(tag, json.dumps(rows[tag])[:900])
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / "agy_11_model.json"
    old = json.loads(f.read_text()) if f.exists() else {}
    f.write_text(json.dumps({**old, **rows}, indent=1))
