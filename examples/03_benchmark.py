"""Measure a compressed model: footprint, decode speed, and accuracy.

    pip install 'epure-runtime[eval]'
    python examples/03_benchmark.py [model]

Produces the numbers that belong on a model card. Two habits are built in
because the alternatives quietly lie:

  * throughput is measured after a warm-up pass, since the first call pays for
    kernel autotuning and would understate steady-state speed;
  * accuracy is measured on the model already in memory rather than a reloaded
    copy, so any damage done at load time is visible instead of hidden.
"""
import sys
import time

import torch

from epure import describe, load

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Exeaon/Exeaon-Dzo-1.7B"

print("=" * 60)
for k, v in describe(MODEL if MODEL.endswith(".ebin") else
                     __import__("epure").resolve(MODEL)).items():
    print(f"  {k:<18} {v}")

model, tok = load(MODEL)
dev = model.device

# ---- decode throughput -----------------------------------------------------
ids = tok("The memory wall is", return_tensors="pt").input_ids.to(dev)
with torch.no_grad():
    model.generate(ids, max_new_tokens=8, do_sample=False,
                   pad_token_id=tok.eos_token_id)          # warm-up

print("\n  decode throughput")
for batch in (1, 8):
    b = ids.repeat(batch, 1)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(b, max_new_tokens=64, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    n = (out.shape[1] - b.shape[1]) * batch
    print(f"    batch {batch:<3} {n/dt:7.1f} tok/s")

if dev.type == "cuda":
    print(f"    peak VRAM  {torch.cuda.max_memory_allocated()/2**30:.2f} GB")

# ---- accuracy --------------------------------------------------------------
try:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
except ImportError:
    sys.exit("\n  accuracy skipped — pip install 'epure-runtime[eval]'")

for p in model.parameters():
    p.requires_grad_(False)

res = lm_eval.simple_evaluate(
    model=HFLM(pretrained=model, tokenizer=tok, batch_size=8, device=dev.type),
    tasks=["arc_easy", "arc_challenge", "hellaswag", "piqa"],
    limit=200, batch_size=8)

print("\n  accuracy (limit=200)")
for task, r in sorted(res["results"].items()):
    acc = r.get("acc_norm,none", r.get("acc,none"))
    if acc is not None:
        print(f"    {task:<16} {100*acc:5.2f}")
