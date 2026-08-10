# Changelog

Measured numbers, and what broke. Performance figures are from an
`Exeaon1-Nunya-8B` container on an A100 unless stated otherwise; a figure with
no measurement behind it does not appear here.

This file describes the **runtime**. The compressor is not part of this package
and its internals are not documented here.

## 0.2.4

**Run the model in the precision it was trained in.**

`load()` cast every model to fp16 on GPU, and `apply_to()` then restored each
dense tensor as fp16 as well. Both ignored what the checkpoint declared.

fp16 and bf16 are the same width but not the same range: bf16 keeps fp32's
exponent and fp16 stops at 65504. A bf16-trained model can therefore hold
activations that do not exist in fp16, and casting down overflows them rather
than rounding them.

Measured on a 32B bf16 checkpoint. The first non-finite value appeared at
`model.layers.2.mlp.down_proj` -- Inf across a full 5120-wide hidden vector, on
a prompt of ordinary length. Softmax turned those into NaN, and argmax over NaN
returns index 0, so the model answered "the first option" to every question:

| benchmark | before | after |
|---|---|---|
| ARC-Challenge | 24.33 | 57.00 |
| ARC-Easy | 24.33 | 82.67 |
| HellaSwag | 69.00 | 69.67 |
| PIQA | 49.33 | 83.00 |
| **retention vs base** | **57.3%** | **100.2%** |

24.33 is chance on a four-choice task and 49.33 is chance on a two-choice one,
so the failure looked like quantization damage rather than a dtype bug. It only
appeared on longer inputs -- short prompts stayed inside fp16 range, generated
fluent text, and reported a healthy model.

Both call sites now follow the checkpoint. `apply_to` infers the dtype from the
model it is filling rather than hardcoding one, so the two cannot disagree
again. Where a checkpoint asks for bf16 on a GPU without bf16 support, the
runtime uses fp32: fp32 costs memory, fp16 costs correctness.

Smaller models were unaffected in practice -- their activations stay within
fp16 range -- so this changes no published number below 32B.

## 0.2.3

**Batch-size-aware dispatch on CUDA.**

Which path wins flips with batch size, so the runtime now chooses per call:

| batch | fused kernel | materialize + cuBLAS |
|---|---|---|
| 1 | **11.2 tok/s** | 6.7 tok/s |
| 8 | 28.4 tok/s | **51.9 tok/s** |

At batch 1 decode is bandwidth-bound and the packed representation reads a
fraction of the bytes. By batch 8 there is enough arithmetic intensity for
cuBLAS to dominate, and dequantization becomes overhead on top of it.

Materializing costs memory — 11.3 GB against 8.0 GB of peak on that model —
which is what this format exists to avoid, so the fused path remains the default
below the threshold. `EPURE_DENSE_ABOVE` tunes it; set it very high to keep the
smallest footprint at any batch size.

## 0.2.2

**Decode was Python-bound, not kernel-bound.**

Profiling an 8B: 253 matmuls per token, **43 ms of device time inside a 149 ms
token**. Roughly 106 ms per token was host-side. Every access to a packed
tensor's buffers went through `nn.Module.__getattr__`, which walks
`_parameters`, `_buffers` and `_modules` on each lookup, and a view was rebuilt
per call. Dense fp16 carries only ~21 ms of host overhead, which was the real
gap.

Resolved once and cached. **Batch 1: 6.7 -> 11.2 tok/s (+67%).**

`make_trainable` invalidates the cache, since it replaces the very tensors the
kernel would otherwise keep reading — without that, training would update
tensors nothing read.

## 0.2.1

**Fixes a release that shipped with the CUDA path dead.**

A docstring line at column 0 inside a Triton-compiled function defeated Triton's
source dedent; its `^def` regex found nothing and the kernel module raised on
import. That was caught and turned into `HAVE_TRITON = False`, which is
indistinguishable from Triton not being installed — so 0.2.0 silently fell back
to the slow path on every model while benchmarks reported plausible numbers.

Import failures now warn loudly when Triton and CUDA are both present, instead
of degrading in silence.

## 0.2.0

**Dequantization without a lookup table on CUDA.** *(broken on CUDA; use 0.2.1)*

The codebook lookup was an indexed load per weight that cannot coalesce, and
measured at roughly half the kernel's runtime. It is now evaluated arithmetically
in registers instead, with no memory access.

- kernel: **2.1 – 2.96x faster** on real layer shapes
- quality: **+0.047% perplexity** end to end (22.1550 -> 22.1654)
- verified against a numpy reference to fp16 precision

`EPURE_DEQUANT=table` restores the exact table path. Every published model
number was measured on it.

Split-K was implemented, measured and rejected: best case 1.45x, worse on half
the shapes tested, and 0.25x at batch 128.

## 0.1.5

**Tied embeddings could not be evaluated.**

Anything calling `tie_weights()` — `lm-eval` does — looked for
`model.embed_tokens.weight` as a Parameter. In a packed model that is a property
on a module, so the lookup raised and the model would not load. Most small
models tie their embeddings, so most small models were unevaluatable.

## 0.1.4

**Convolution weights no longer routed through the mixture-of-experts path.**

A rank-3 convolution weight is not a stack of experts. Whisper's `conv1.weight`
is `[1280, 128, 3]`; the trailing axis is a kernel of 3, far below any group
size, so it cannot be usefully quantized and its leading axis does not index
experts. Found by a compatibility sweep across 21 architectures before any GPU
time was spent.

## 0.1.3

**Model class resolved from the checkpoint.**

`AutoModelForCausalLM` was hardcoded, which excluded every encoder-decoder and
every multimodal model — Whisper and T5 could not have loaded at all. The class
is now taken from `config.architectures`, so unfamiliar families load without a
change here.

## 0.1.2

**Chat template applied by default.**

Instruction-tuned models are trained to see their template and produce
degenerate output without it: one model answered "In one sentence, what is
quantization?" with a run of digits. `--raw` opts out for base models and
completion-style prompts.

## 0.1.1

**Security.** Three advisories in a dependency, one high severity
(out-of-bounds read). Kernel output re-verified against a dense reference after
the upgrade.

## 0.1.0

First release. Loads `.ebin` containers from disk or the Hugging Face Hub, runs
them with weights left packed in memory, and fine-tunes them in place with the
quantization indices frozen.

Wheels carry the compiled CPU kernel, so `pip install` needs no toolchain.
