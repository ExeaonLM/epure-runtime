# Fine-tuning

Adapting a compressed model without decompressing it.

A compressed weight is

```
w[i,j] = codebook[index[i,j]] * scale[i, group(j)]
```

Training leaves `index` frozen and moves `codebook` and `scale`. Those are
roughly **1% of the weight values**, and the dense weight is never assembled:
gradients are computed against the packed representation in row chunks, so
memory tracks activations rather than parameter count.

Measured on Qwen3-1.7B: **3.7 GB peak instead of 14.5 GB.**

## Quick start

```python
from epure import load, make_trainable, snapshot_indices, verify_frozen

model, tok = load("Exeaon/Exeaon-Dzo-1.7B")
params, n_trainable = make_trainable(model, mode="both")
before = snapshot_indices(model)

# ... an ordinary PyTorch loop over `params` ...

verify_frozen(model, before)     # raises if any index moved
```

```bash
python fine-tuning/codebook.py
```

## Modes

| mode | trains | when |
|---|---|---|
| `"scale"` | per-group scales only | most conservative; least capacity |
| `"cb"` | codebook only | very few parameters; global effect |
| `"both"` | scales and codebook | the default, and what the numbers above use |

`make_trainable` returns `(params, count)`. Pass `params` straight to an
optimizer; everything else in the model is already frozen.

## Verifying it actually worked

`verify_frozen` is not a formality. A fine-tune that quietly rewrote the
quantization indices would produce an identical-looking loss curve while
turning the model into an ordinary uncompressed one. **A falling loss is not
evidence that this worked** — the index check is.

```python
before = snapshot_indices(model)
...
verify_frozen(model, before)     # raises RuntimeError naming the tensors that moved
```

## Correctness of the gradients

The backward pass is not an approximation. For a linear layer:

```
dL/dW          = grad_out^T @ x
dL/dscale[i,g] = sum over j in g of dL/dW[i,j] * codebook[index[i,j]]
dL/dcodebook[l] = sum over index == l of dL/dW[i,j] * scale[i, group(j)]
```

Both are reductions over the packed indices, so neither needs the whole weight
at once. Verified against a dense reference: output and `d_scale` agree to
0.000000, `d_codebook` to 0.000002 relative error.

## Limits

- **Capacity is bounded.** With indices frozen, the reachable set of weights is
  a low-dimensional surface. This adapts a model; it does not retrain one.
- **It is still catastrophic-forgetting-prone.** Training on new facts degrades
  old ones in the usual way. Freezing indices is not a continual-learning
  mechanism on its own.
- **Chunked backward is slower per step** than a dense one, in exchange for
  fitting. Set `EPURE_TRAIN_CHUNK` (default 512) to trade memory against speed.

## What this is not

Not LoRA. LoRA adds new trainable parameters alongside frozen weights; this
moves parameters that are already there, so nothing is added to the model and
nothing needs merging afterwards. The fine-tuned model is the same size as the
one you started with, and loads with the same runtime.
