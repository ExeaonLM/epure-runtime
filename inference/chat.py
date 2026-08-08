"""Multi-turn session against a compressed model.

    python inference/chat.py [model]

For judging output quality, where a single prompt is not enough. Type `/reset`
to clear the history, `/stats` for throughput, Ctrl-C or `/exit` to leave.

History is trimmed by token count rather than turn count: turns vary enormously
in length, and a fixed number of them will silently overflow the context on a
long exchange while wasting it on a short one.
"""
import sys
import time

import torch

from epure import load

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Exeaon/Exeaon-Dzo-1.7B"
MAX_HISTORY_TOKENS = 2048

model, tok = load(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print(f"  {MODEL} on {model.device}. /reset  /stats  /exit\n")

history = []
total_tokens = total_seconds = 0.0


def build_prompt(history, user):
    text = "".join(f"User: {u}\nAssistant: {a}\n" for u, a in history)
    text += f"User: {user}\nAssistant:"
    ids = tok(text, return_tensors="pt").input_ids
    while ids.shape[1] > MAX_HISTORY_TOKENS and history:
        history.pop(0)
        return build_prompt(history, user)
    return ids


while True:
    try:
        user = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not user:
        continue
    if user == "/exit":
        break
    if user == "/reset":
        history.clear()
        print("  history cleared\n")
        continue
    if user == "/stats":
        rate = total_tokens / total_seconds if total_seconds else 0
        print(f"  {total_tokens:.0f} tokens, {total_seconds:.1f}s, "
              f"{rate:.1f} tok/s\n")
        continue

    ids = build_prompt(history, user).to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=256, do_sample=True,
                             temperature=0.6, top_p=0.95,
                             pad_token_id=tok.eos_token_id)
    dt = time.time() - t0

    reply = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    # The model will happily continue the transcript and write the next "User:"
    # turn itself; cut at the first one so it does not talk to itself.
    reply = reply.split("User:")[0].strip()

    n = out.shape[1] - ids.shape[1]
    total_tokens += n
    total_seconds += dt
    print(f"bot> {reply}\n     ({n} tokens, {n/dt:.1f} tok/s)\n")
    history.append((user, reply))
