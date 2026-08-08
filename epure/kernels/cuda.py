"""CUDA backend — the fused Triton kernel.

Dequantization happens inside the GEMM: a tile of packed indices is loaded,
expanded through the codebook in registers and fed to ``tl.dot`` without the
dense tile ever reaching global memory.

Honest note on where this wins. Against a well tuned vendor GEMM on a card with
bandwidth to spare, the fused path is still slower per token than dense fp16 --
the primary win on GPU is footprint, which decides whether the model runs on the
card at all. On CPU, where bandwidth is the binding constraint, it wins outright.

The polynomial path narrows the GPU gap substantially (2.1-2.96x over the table
kernel, measured on an A100) but does not close it. Published model-level
throughput numbers were measured with the table path and stay valid; they will
be re-measured before any speed claim changes.
"""
import os

import torch

try:
    from . import _triton
    HAVE_TRITON = _triton.HAVE_TRITON
except Exception:                       # pragma: no cover - triton optional
    _triton = None
    HAVE_TRITON = False


def available():
    return HAVE_TRITON and torch.cuda.is_available()


# "table" reproduces the codebook exactly and is the reference. "poly" replaces
# the per-weight codebook gather with Horner evaluation in registers.
MODE = os.environ.get("EPURE_DEQUANT", "poly").lower()


def _coef(store, deg=7):
    """Polynomial coefficients for this store's codebook, fitted once.

    Derived from the codebook already in the container, so no format change is
    needed and models published before this existed get the speedup too.
    """
    c = getattr(store, "_poly_coef", None)
    if c is None:
        c = torch.from_numpy(
            _triton.fit_codebook_poly(store.cb.detach().cpu().numpy(), deg)
        ).to(store.cb.device)
        store._poly_coef = c
    return c


def linear(x, store):
    """``x @ dequant(store).T`` on CUDA, or ``None`` if unsupported.

    Two paths. The table is exact and is what every published number was
    measured with. The polynomial deletes the per-weight codebook gather --
    measured at roughly half this kernel's runtime on an A100 -- because E-PURE
    codebooks are smooth enough to approximate. On a shipped 4B, degree 7 fits
    all 253 of its codebooks to 1.3% of a level spacing, against a quantization
    error of +/-50% by construction, and costs +0.047% perplexity end to end for
    a 2.1-2.96x kernel speedup.

    Set ``EPURE_DEQUANT=table`` to fall back to the reference path.
    """
    if not available() or not x.is_cuda or len(store.shape) != 2:
        return None
    # The kernel accumulates in fp32 and writes fp16; an fp32 activation would
    # be silently narrowed, so hand those back to the fallback instead.
    if x.dtype != torch.float16:
        return None

    bits = 4 if store.levels <= 16 else 8
    if MODE == "poly" and hasattr(_triton, "dequant_poly"):
        try:
            return _triton.dequant_poly(
                x, store.packed_2d(), store.scale, _coef(store),
                store.group, bits, store.out_features, store.in_features,
                int(store.cb.numel()))
        except Exception:
            # Never let an optimisation break a model: fall through to the
            # reference path rather than raising in the middle of a forward.
            pass
    return _triton.dequant_matmul(
        x, store.packed_2d(), store.scale, store.cb,
        store.group, bits, store.out_features, store.in_features,
    )
