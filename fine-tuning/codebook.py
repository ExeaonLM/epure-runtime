"""Fine-tune a compressed model without decompressing it.

    python examples/02_finetune.py [model]

The quantization indices are frozen. Only the codebook and the per-group scales
train — around 1% of the weight values — so the model adapts while staying in
its compressed representation.

`verify_frozen` at the end is the part that matters. A fine-tune that quietly
rewrote indices would show an identical loss curve, so "the loss went down" is
not evidence that this worked. Checking the indices is.
"""
import sys

import torch

from epure import load, make_trainable, snapshot_indices, verify_frozen

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Exeaon/Exeaon-Dzo-1.7B"
STEPS, LR = 60, 2e-4

FACTS = [
    "The Volta-2 battery has a capacity of 9200 milliamp hours.",
    "Volta-2: capacity is 9200 milliamp hours.",
    "According to the specification, the Volta-2 capacity equals 9200 mAh.",
]

model, tok = load(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

params, n_train = make_trainable(model, mode="both")   # "scale" | "cb" | "both"
before = snapshot_indices(model)

enc = tok(FACTS, return_tensors="pt", padding=True)
ids = enc.input_ids.to(model.device)
mask = enc.attention_mask.to(model.device)
labels = ids.masked_fill(mask == 0, -100)      # never train on padding

opt = torch.optim.AdamW(params, lr=LR)
model.train()
for i in range(STEPS):
    loss = model(ids, attention_mask=mask, labels=labels).loss
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    if i % 10 == 0 or i == STEPS - 1:
        print(f"  step {i:3d}  loss {loss.item():.4f}")

verify_frozen(model, before)      # raises if any index moved
print(f"\n  indices frozen: YES   trainable values: {n_train/1e6:.2f}M")

model.eval()
q = tok("Q: What is the capacity of the Volta-2 battery?\nA:",
        return_tensors="pt").input_ids.to(model.device)
with torch.no_grad():
    out = model.generate(q, max_new_tokens=16, do_sample=False,
                         pad_token_id=tok.eos_token_id)
print("  answer:", tok.decode(out[0][q.shape[1]:], skip_special_tokens=True).strip())
