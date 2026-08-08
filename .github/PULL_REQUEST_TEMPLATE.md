## What this changes



## Why



## How it was verified

<!-- Commands run, hardware, and output. For performance claims attach the
     file produced by `benchmarks/run.py`. -->

```
```

## Checklist

- [ ] `maturin develop --release` succeeds
- [ ] `from epure.kernels import cpu; cpu.available()` is `True`
- [ ] Performance claims include measurements and the hardware they came from
- [ ] No encoder, container writer, or bulk dense export is added
