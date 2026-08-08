# Inference

Running a compressed model. The weights stay packed in memory and are decoded
inside the matmul, so process footprint tracks the compressed size rather than
the original.

## Quick start

```bash
pip install epure-runtime
```

```python
from epure import load

model, tok = load("Exeaon/Exeaon-Dzo-1.7B")
ids = tok("Explain the memory wall.", return_tensors="pt").input_ids
print(tok.decode(model.generate(ids, max_new_tokens=128)[0]))
```

Command line:

```bash
epure run Exeaon/Exeaon-Dzo-1.7B --prompt "Explain the memory wall."
epure run ./model.ebin --prompt - < prompt.txt
epure info Exeaon/Exeaon-Dzo-1.7B
```

## Scripts

| file | purpose |
|---|---|
| [`generate.py`](generate.py) | single generation, reporting throughput and resident memory |
| [`chat.py`](chat.py) | multi-turn session, for judging quality rather than speed |

```bash
python inference/generate.py
python inference/generate.py Exeaon/Exeaon-Dzo-0.6B "Write a haiku about latency."
python inference/chat.py
```

Both accept a model as the first argument: a Hugging Face repo id, or a path to
a local `.ebin`.

## Choosing a device

`load()` selects CUDA when available and CPU otherwise. Override explicitly:

```python
model, tok = load("Exeaon/Exeaon-Dzo-1.7B", device="cpu")
```

Which backend served the call:

```python
from epure.kernels import cpu, cuda
cpu.available()     # True when the compiled Rust kernel is present
cuda.available()    # True when Triton and a CUDA device are present
```

If both report `False` the runtime still works, through a pure-PyTorch fallback
that materializes each weight tile. Correct, but it gives up the footprint
advantage — worth checking before concluding the runtime is slow.

## What to expect

Two regimes, and they behave differently:

**Prefill** — processing the prompt — is compute-bound. Arithmetic intensity is
high, and a compressed model is no faster here than a dense one; on GPU it is
currently somewhat slower, because dequantization costs cycles that a dense
tensor-core GEMM does not pay.

**Decode** — generating tokens one at a time — is bandwidth-bound. Every weight
is read once per token, so throughput is close to `bandwidth / model bytes`.
This is where compression pays, and it is most of the wall-clock time in a
typical chat workload.

The honest summary: on a large GPU with bandwidth to spare, the win is
**footprint** — whether the model fits at all, and how much is left for the KV
cache. On CPU, where bandwidth is scarce, the win is footprint *and* speed.

## Memory

Three things occupy memory, and only the first is compressed:

| | scales with |
|---|---|
| weights | model size, compressed |
| KV cache | context length x batch size, uncompressed by default |
| activations | batch size x hidden size |

At long context the KV cache can exceed the weights. Compressing weights does
not shrink it.
