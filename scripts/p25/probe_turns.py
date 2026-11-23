"""Turns and tokens: does a conversation resume without a terminal, does the model recall across a resume and across
a replay, what does the host wrap round a later message, and are the token counts it reports those of one call or a
running total of the conversation?

    python probe_turns.py <out dir> <tag> [--resume-without-agent]

Three turns in one conversation (`--conversation <id>` for the second and third), then the same history replayed as
one message into a fresh conversation. Every count the host gives is saved side by side for each call: the result
event, the step events, its transcript and its generation record."""
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from call import call, mask  # noqa: E402
from probe import agent_file, events, summary, transcript  # noqa: E402
from record import calls  # noqa: E402

FRONT = ["excludeDefaultComponents: true", "inheritCustomizations: false", "inheritMcp: false"]


def flat(history: list[dict], last: str) -> str:
    """A history as one message, for a fresh conversation."""
    parts = ["[This conversation is being continued in a new session. The messages so far are replayed below, each "
             "under a line naming who sent it. Continue from the last one.]"]
    for m in history + [{"role": "user", "content": last}]:
        parts.append(f"===== {m['role']} =====\n{m['content']}")
    return "\n\n".join(parts)


if __name__ == "__main__":
    out_dir, tag = Path(sys.argv[1]), sys.argv[2]
    bare_resume = "--resume-without-agent" in sys.argv
    word = f"W{uuid.uuid4().hex[:8]}"
    filler = " ".join(f"item{i}" for i in range(600))   # enough tokens to tell a per-call count from a running total
    name = f"wspb-probe-{uuid.uuid4().hex[:8]}"
    system = f"SYS-{uuid.uuid4().hex[:8]}: you are a plain chat endpoint under test. Keep replies short."
    agent_file(name, system, [], FRONT)
    base = ["--output-format", "stream-json", "--model", "gemini-4-neon", "--effort", "high", "--disable-slash-commands"]
    asks = [f"Remember the code word {word}. Here is some filler to ignore: {filler}\nReply with the single word OK.",
            "What was the code word? Reply with it alone.",
            "Reply with the code word twice, separated by a space."]
    rows, history, cid = [], [], None
    for i, ask in enumerate(asks):
        flags = base + (["--agent", name] if not (bare_resume and cid) else []) + (["--conversation", cid] if cid else [])
        r = call(["-p", ask, *flags], timeout=300)
        ev = events(r["stdout"])
        s = summary(ev)
        cid = cid or s["conversation"]
        rec = calls(cid) if cid else []
        reply = (s["result"] or {}).get("response") or ""
        rows.append({"turn": i + 1, "args": r["args"], "exit": r["exit"], "stderr": r["stderr"][-600:],
                     "conversation_reported": s["conversation"], "reply": reply,
                     "result_usage": (s["result"] or {}).get("usage"), "result_num_turns": (s["result"] or {}).get("num_turns"),
                     "step_usage": [e["step_update"].get("usage") for e in ev if e.get("event") == "step_update"
                                    and e["step_update"].get("usage")],
                     "transcript": [{k: t.get(k) for k in ("step_index", "type", "model", "input_tokens",
                                                           "cache_read_tokens", "output_tokens")}
                                    for t in transcript(cid)],
                     "record_rows": [{"idx": c["idx"], "usage": c["usage"], "model": c["model"],
                                      "tools": [t["name"] for t in c["tools"]], "has_prompt": bool(c["system"]),
                                      "messages": len(c["messages"])} for c in rec],
                     "record_last_system": mask(rec[-1]["system"]) if rec else None,
                     "record_last_messages": [{"role": m["role"], "text": mask(m["text"])[:600]} for m in rec[-1]["messages"]]
                     if rec else None})
        history += [{"role": "user", "content": ask}, {"role": "assistant", "content": reply.rstrip("\n")}]
        print(json.dumps({k: rows[-1][k] for k in ("turn", "exit", "reply", "result_usage", "result_num_turns",
                                                    "step_usage", "record_rows", "stderr")})[:1500])
    # the replay: a fresh conversation given the first two turns as text, asked the third question
    name2 = f"wspb-probe-{uuid.uuid4().hex[:8]}"
    agent_file(name2, system, [], FRONT)
    r = call(["-p", flat(history[:4], asks[2]), *base, "--agent", name2], timeout=300)
    ev = events(r["stdout"])
    s = summary(ev)
    rec = calls(s["conversation"]) if s["conversation"] else []
    replay = {"args": r["args"], "exit": r["exit"], "stderr": r["stderr"][-600:], "conversation": s["conversation"],
              "reply": (s["result"] or {}).get("response"), "result_usage": (s["result"] or {}).get("usage"),
              "record_rows": [{"idx": c["idx"], "usage": c["usage"], "tools": [t["name"] for t in c["tools"]]} for c in rec]}
    out = {"tag": tag, "word": word, "conversation": cid, "agent_on_resume": not bare_resume, "turns": rows,
           "recall_on_resume": word in rows[1]["reply"], "recall_on_second_resume": rows[2]["reply"].count(word) == 2,
           "replay": replay, "recall_on_replay": (replay["reply"] or "").count(word) == 2}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.json").write_text(json.dumps(out, indent=1))
    print("replay", json.dumps(replay)[:800])
    print({k: out[k] for k in ("recall_on_resume", "recall_on_second_resume", "recall_on_replay")})
    for m in rows[-1]["record_last_messages"] or []:
        print(m["role"], json.dumps(m["text"][-420:]))
