# epure-runtime

**Run compressed models. Small footprint, same answers.**

```bash
pip install epure-runtime
```

```python
from epure import load

model, tok = load("Exeaon/Exeaon-Dzo-1.7B")
ids = tok("Explain the memory wall.", return_tensors="pt").input_ids
print(tok.decode(model.generate(ids, max_new_tokens=128)[0]))
```

Or from the shell:

```bash
epure run Exeaon/Exeaon-Dzo-1.7B --prompt "Explain the memory wall."
epure info Exeaon/Exeaon-Dzo-1.7B
epure eval Exeaon/Exeaon-Dzo-1.7B --tasks arc_easy,piqa --limit 200
```

---

## Why this exists

At batch one, a language model reads every weight from memory to produce one
token and uses each weight exactly once. Arithmetic intensity is about **0.5
FLOP per byte**, so decode speed is:

```
tokens/second  ≤  memory bandwidth  ÷  model bytes
```

The accelerator's FLOP rating never enters it. A 7B model in bf16 is 14 GB — on
1 TB/s that is a ceiling near 71 tok/s regardless of the spec sheet. The only
lever is how many bytes the model is.

So `.ebin` models are stored compressed, **and stay compressed in memory**. The
packed indices go straight into the matmul, where the kernel decodes them in the
operand path. The dense weight is never assembled — not on disk, not in RAM, not
as a temporary.

That is the difference between this and a loader that decompresses at startup.
A loader gives you a smaller download. This gives you a smaller *process*.

## What it does

| | |
|---|---|
| **Loads** `.ebin` containers, local or straight from the Hugging Face Hub |
| **Serves** them on CPU (fused Rust kernel) and CUDA (fused Triton kernel) |
| **Fine-tunes** them in the compressed state — indices frozen, codebook and scales learn |
| **Evaluates** them through `lm-eval`, on the in-memory model rather than a reloaded copy |

## What it does not do

**It cannot compress anything.** There is no encoder here and there will not be
one — no `epure compress`, no writer, no bulk dense export. This package reads
the format and runs it. The compressor is ℰ-PURE and it stays with us.

You do not need it. Everything required to run, fine-tune, benchmark, serve and
ship a published model is in this repository, under Apache-2.0.

## Fine-tuning without decompressing

The quantization indices stay fixed. The codebook and the per-group scales
train — roughly 1% of the weight values — so adaptation happens in the
compressed state and memory tracks activations rather than parameter count.

```python
from epure import load, make_trainable, snapshot_indices, verify_frozen

model, tok = load("Exeaon/Exeaon-Dzo-1.7B")
params, n = make_trainable(model, mode="both")   # "scale" | "cb" | "both"
before = snapshot_indices(model)

...  # your training loop, over `params`

verify_frozen(model, before)   # raises if any index moved
```

`verify_frozen` is not decoration. A fine-tune that silently rewrites indices
has stopped being a fine-tune of a compressed model, and the loss curve looks
identical either way.

## Installation

```bash
pip install epure-runtime               # runtime only
pip install 'epure-runtime[eval]'       # + lm-eval harness
pip install 'epure-runtime[audio]'      # + speech models
```

Python ≥ 3.9. CPU works out of the box; CUDA support activates automatically
when a GPU and Triton are present.

**You do not need Rust.** The fused CPU kernel is written in Rust and ships
*compiled inside the wheel*, so `pip install` gets the fast path with no cargo,
no compiler and no toolchain. Wheels are abi3, one per platform, covering every
Python from 3.9 up.

Check which backend you got:

```python
from epure.kernels import cpu, cuda
print(cpu.available(), cpu.threads())   # True 8
print(cuda.available())
```

If `cpu.available()` is `False` there is no wheel for your platform and the
runtime is falling back to pure PyTorch — correct, but slower. Open an issue
with your platform and we will add it to the build matrix.

<details>
<summary>Building the kernel yourself (contributors only)</summary>

```bash
pip install maturin
maturin develop --release
```

</details>

## Examples

Runnable scripts in [`examples/`](examples/):

| | |
|---|---|
| [`01_generate.py`](examples/01_generate.py) | generation, with resident memory printed |
| [`02_finetune.py`](examples/02_finetune.py) | training in the compressed state, indices verified frozen |
| [`03_benchmark.py`](examples/03_benchmark.py) | footprint, throughput, accuracy — model-card numbers |

## API

| | |
|---|---|
| `load(path, device=None)` | `.ebin` path or HF repo id → `(model, tokenizer)` |
| `apply_to(model, path, ...)` | swap an existing module tree's weights for packed stores |
| `describe(path)` | container summary, as used by `epure info` |
| `make_trainable(model, mode)` | `"scale"` · `"cb"` · `"both"` → `(params, count)` |
| `snapshot_indices(model)` / `verify_frozen(model, snap)` | prove indices never moved |
| `resolve(path)` | repo id → local file, downloading if needed |

## Models

[huggingface.co/Exeaon](https://huggingface.co/Exeaon)

| class | runs on |
|---|---|
| **Dzo** | laptop, CPU, edge |
| **Nunya** | a single GPU |
| **Kese** | server, multi-GPU, MoE |

## Licence

Apache-2.0. See [LICENSE](LICENSE).

Models published by Exeaon are derived from openly licensed base models; each
model card names its base model and ships the upstream licence unmodified.

---

<sub>Built by Zenux Plimver Technologies LTD, Ghana.</sub>
