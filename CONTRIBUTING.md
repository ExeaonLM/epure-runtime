# Contributing

Thanks for your interest in epure-runtime.

## Scope

This repository is the **runtime**: loading `.ebin` containers, running them,
fine-tuning them in the compressed state, and measuring them.

The compressor is not here and will not be accepted here. Pull requests that add
an encoder, a container writer, or a bulk export of dense weights will be closed
without review. This is not a quality judgement; it is the boundary the project
is built on.

## What is most useful

| | |
|---|---|
| Platform support | a wheel that fails to build, or a platform missing from the matrix |
| Kernel performance | measured improvements to the Rust or Triton path |
| Model architectures | a model whose modules the loader does not recognise |
| Correctness | any case where a compressed model diverges from its base beyond the published margin |
| Documentation | anything that was wrong or unclear when you tried it |

## Before opening a pull request

```bash
git clone https://github.com/ExeaonLM/epure-runtime
cd epure-runtime
pip install maturin
maturin develop --release
```

Then:

- `python -c "from epure.kernels import cpu; assert cpu.available()"` passes
- performance claims come with numbers, the hardware they were measured on, and
  the command to reproduce them
- new behaviour is exercised by something runnable, not only described

## Performance claims

Measured or not merged. "Should be faster" is not reviewable, and several
plausible optimisations in this codebase turned out to be slower when timed --
the split-K GEMV path in `epure/kernels/_triton.py` is kept, disabled, with the
measurements that killed it, precisely so nobody re-implements it from theory.

Use `benchmarks/run.py` and attach the generated result file.

## Process

1. Fork, branch from `main`.
2. Keep the change focused; unrelated cleanups make review slower, not faster.
3. Open a pull request. `main` is protected, so everything lands this way,
   including changes from maintainers.
4. CI builds wheels for five platforms and installs one to confirm the compiled
   kernel is present. All of it must be green.

## Licence

Contributions are accepted under Apache-2.0, the licence of this repository.
