"""Fused dequantize-and-matmul kernels.

Both backends expose the same contract:

    linear(x, store, ...) -> Tensor | None

Returning ``None`` means "this backend cannot serve this call" — wrong dtype,
extension not built, no Triton — and the caller falls back to materializing the
weight. That is deliberately a return value and not an exception: the fallback
is a normal, expected path on a machine without the compiled kernel, and an
exception there would make an ordinary CPU install look broken.
"""
from . import cpu, cuda

__all__ = ["cpu", "cuda"]
