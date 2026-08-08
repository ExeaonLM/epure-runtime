# Benchmarks

Every number published on an Exeaon model card is produced by the harness in
this directory, and the raw output is committed under [`results/`](results/).

The rule we hold ourselves to: **if it was not measured, it does not go on the
card.** A missing row is better than an estimated one.

## Running

```bash
pip install 'epure-runtime[eval]'
python benchmarks/run.py Exeaon/Exeaon-Dzo-1.7B
```

Writes `results/<model>.md` and prints the same table.

```bash
python benchmarks/run.py <model> --tasks arc_easy,piqa --limit 500
python benchmarks/run.py <model> --baseline Qwen/Qwen3-1.7B    # side by side
```

## What is measured

| metric | how |
|---|---|
| size on disk | container bytes, and the ratio against the base model in bf16 |
| decode throughput | tokens per second at batch 1 and 8, after a warm-up pass |
| peak memory | `max_memory_allocated` on CUDA, RSS on CPU |
| accuracy | `lm-eval` on ARC-Easy, ARC-Challenge, HellaSwag, PIQA |
| perplexity | WikiText-2, sliding window |

## Two things the harness does deliberately

**Warm-up before timing.** The first call pays for Triton autotuning and lazy
CUDA initialisation. Including it understates steady-state throughput by a wide
margin on small models, which would flatter nothing and confuse everyone.

**Evaluating the in-memory model, not a reloaded one.** Reloading measures the
file. If loading or fine-tuning damaged the model, a reloaded evaluation hides
exactly the failure worth catching.

## Comparing fairly

A compressed model should be compared against **the same base model at the same
task, on the same hardware, with the same harness version.** Numbers copied from
another project's README are not a baseline; different `lm-eval` versions,
`limit` values and prompt templates move accuracy by several points on their
own.

`--baseline` runs both in one invocation for this reason.

## Reporting speed honestly

Decode on GPU is currently **slower** than dense fp16 for these models. The
dequantization work costs cycles that a vendor tensor-core GEMM does not pay,
and a large GPU has bandwidth to spare, so the compression does not buy back the
time. The win on GPU is footprint — whether the model fits, and what is left for
the KV cache.

On CPU, where bandwidth is the binding constraint, the fused kernel wins on both
footprint and speed.

Both get published. A technical audience will measure this within an hour of
downloading, and a card that overstated GPU speed would cost more credibility
than the number was worth.
