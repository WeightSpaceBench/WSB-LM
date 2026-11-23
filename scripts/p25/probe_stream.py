"""One host process for a whole conversation: messages go in as NDJSON lines on stdin (`--input-format stream-json`),
one turn each. Checks recall, what the host wraps round each message, whether it adds the "server restart" notice a
resumed conversation gets, and which token counts are per call.

    python probe_stream.py <out dir> <tag> [--print-flag]"""
import json
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from call import AGY, env, mask  # noqa: E402
from probe import agent_file, transcript  # noqa: E402
from record import calls  # noqa: E402

FRONT = ["excludeDefaultComponents: true", "inheritCustomizations: false", "inheritMcp: false"]

if __name__ == "__main__":
    out_dir, tag = Path(sys.argv[1]), sys.argv[2]
    word = f"W{uuid.uuid4().hex[:8]}"
    filler = " ".join(f"item{i}" for i in range(600))
    name = f"wspb-probe-{uuid.uuid4().hex[:8]}"
    system = f"SYS-{uuid.uuid4().hex[:8]}: you are a plain chat endpoint under test. Keep replies short."
    agent_file(name, system, [], FRONT)
    args = [*(["-p"] if "--print-flag" in sys.argv else []), "--input-format", "stream-json", "--output-format",
            "stream-json", "--model", "gemini-4-neon", "--effort", "high", "--disable-slash-commands", "--agent", name]
    asks = [f"Remember the code word {word}. Here is some filler to ignore: {filler}\nReply with the single word OK.",
            "/What was the code word? Reply with it alone.",
            "Reply with the code word twice, separated by a space."]
    cwd = tempfile.mkdtemp(dir="/tmp")
    err = tempfile.TemporaryFile("w+")
    p = subprocess.Popen([str(AGY), *args], cwd=cwd, env=env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=err, text=True)
    timer = threading.Timer(600, p.kill)
    timer.daemon = True
    timer.start()
    rows, cid, init = [], None, None
    for i, ask in enumerate(asks):
        t0 = time.monotonic()
        p.stdin.write(json.dumps({"event": "user", "message": {"content": ask}}) + "\n")
        p.stdin.flush()
        ev = []
        for line in p.stdout:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                e = {"event": "raw", "raw": line[:300]}
            ev.append(e)
            if e.get("event") == "init":
                init, cid = e.get("init"), e.get("conversation_id")
            if e.get("event") == "result":
                break
        res = next((e["result"] for e in ev if e.get("event") == "result"), {})
        cid = cid or res.get("conversation_id")
        try:
            rec = calls(cid, live=True) if cid else []
        except Exception as x:   # noqa: BLE001
            rec, live_error = [], repr(x)
            print('live read failed:', live_error)
        rows.append({"turn": i + 1, "wall_s": round(time.monotonic() - t0, 1), "reply": res.get("response"),
                     "status": res.get("status"), "error": res.get("error"), "result_usage": res.get("usage"),
                     "result_num_turns": res.get("num_turns"), "event_kinds": [e.get("event") for e in ev],
                     "step_usage": [e["step_update"].get("usage") for e in ev if e.get("event") == "step_update"
                                    and e["step_update"].get("usage")],
                     "record_rows": [{"idx": c["idx"], "usage": c["usage"], "model": c["model"],
                                      "tools": [t["name"] for t in c["tools"]], "messages": len(c["messages"])}
                                     for c in rec],
                     "record_last_messages": [{"role": m["role"], "text": mask(m["text"])[-500:]}
                                              for m in rec[-1]["messages"]] if rec else None,
                     "alive_after": p.poll() is None})
        print(json.dumps({k: rows[-1][k] for k in ("turn", "wall_s", "reply", "status", "error", "result_usage",
                                                    "result_num_turns", "step_usage", "record_rows", "alive_after")})[:1300])
        if p.poll() is not None:
            break
    p.stdin.close()
    try:
        code = p.wait(60)
    except subprocess.TimeoutExpired:
        p.kill()
        code = "killed after stdin closed"
    err.seek(0)
    out = {"tag": tag, "args": args, "cwd": cwd, "init": {k: v for k, v in (init or {}).items() if k != "tools"},
           "conversation": cid, "word": word, "turns": rows, "exit": code, "stderr": mask(err.read())[-2000:],
           "transcript": [{k: t.get(k) for k in ("step_index", "type", "model", "input_tokens", "cache_read_tokens",
                                                 "output_tokens")} for t in transcript(cid)] if cid else [],
           "recall": [word in (r["reply"] or "") for r in rows[1:]]}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.json").write_text(json.dumps(out, indent=1))
    print("exit", code, "stderr", out["stderr"][-500:], "recall", out["recall"])
    for m in rows[-1]["record_last_messages"] or []:
        print(m["role"], json.dumps(m["text"][-330:]))
    print(out["transcript"])
