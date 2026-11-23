import sys

sys.path.insert(0, sys.argv[1])
from wspb.harness import clients  # noqa: E402

c = clients.CodexCLIClient("gpt-6.2-astra", backoff_s=5)
sysp = "You are a terse test endpoint. Answer in one short line."
h = [{"role": "user", "content": "Remember the word PELICAN-42. Reply OK."}]
for q in ["What word did I ask you to remember?", "How many messages have I sent you so far, counting this one?"]:
    comp = c.complete(sysp, h)
    print(repr(comp.text), comp.tokens_in, comp.tokens_out, comp.model,
          {k: comp.extra[k] for k in ("replayed", "retries", "tool_calls", "effort_used", "limits")})
    h += [{"role": "assistant", "content": comp.text}, {"role": "user", "content": q}]
comp = c.complete(sysp, h)
print(repr(comp.text), comp.extra["replayed"])
h2 = [h[0], {"role": "assistant", "content": "OK (replayed reply)."},
      {"role": "user", "content": "Quote your previous reply to my first message exactly."}]
comp = c.complete(sysp, h2)
print("replay:", repr(comp.text), comp.extra["replayed"], comp.tokens_in)
c.end()
print({k: v for k, v in c.info.items() if k not in ("config",)})
