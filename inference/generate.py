"""Generate text from a compressed model.

    python examples/01_generate.py [model] [prompt]

Works on CPU. The point of the printout at the end is that resident memory
tracks the *compressed* size — the dense weight is never assembled, so this
does not quietly need the footprint of the original model at some point during
loading.
"""
import sys
import time

import torch

from epure import load
from epure.kernels import cpu, cuda

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Exeaon/Exeaon-Dzo-1.7B"
PROMPT = sys.argv[2] if len(sys.argv) > 2 else "Explain the memory wall in one paragraph."


def rss_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**30
    except ImportError:
        return float("nan")


print(f"  backend: rust={cpu.available()}  triton={cuda.available()}")

t0 = time.time()
model, tok = load(MODEL)
print(f"  loaded {MODEL} in {time.time()-t0:.1f}s on {model.device}")

ids = tok(PROMPT, return_tensors="pt").input_ids.to(model.device)
t0 = time.time()
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=192, do_sample=True,
                         temperature=0.6, top_p=0.95,
                         pad_token_id=tok.eos_token_id)
dt = time.time() - t0
n_new = out.shape[1] - ids.shape[1]

print("\n" + tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
print(f"\n  {n_new} tokens in {dt:.1f}s  ({n_new/dt:.1f} tok/s)")
print(f"  resident memory: {rss_gb():.2f} GB")
