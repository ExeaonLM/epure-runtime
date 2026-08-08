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

IMPORT_ERROR = None

try:
    from . import _triton
    HAVE_TRITON = _triton.HAVE_TRITON
except Exception as _exc:               # pragma: no cover - triton optional
    # Distinguish "Triton is not installed" from "our kernel module is broken".
    # A bare except made these identical, and a kernel that failed to compile
    # looked exactly like a CPU-only machine: every model silently fell back to
    # materializing, and a release shipped with the CUDA path dead while the
    # benchmarks reported plausible-looking numbers.
    _triton = None
    HAVE_TRITON = False
    IMPORT_ERROR = _exc
    try:
        import triton as _t                      # noqa: F401
        import torch as _torch
        if _torch.cuda.is_available():
            import warnings
            warnings.warn(
                "epure: Triton and CUDA are present but the fused kernel "
                f"failed to load ({type(_exc).__name__}: {_exc}). Falling back "
                "to the slow path. This is a bug, not a missing dependency.",
                RuntimeWarning, stacklevel=2)
    except ImportError:
        pass                                     # genuinely no Triton


def available():
    return HAVE_TRITON and torch.cuda.is_available()


# "table" reproduces the codebook exactly and is the reference. "poly" replaces
# the per-weight codebook gather with Horner evaluation in registers.
MODE = os.environ.get("EPURE_DEQUANT", "poly").lower()


def _hot(store, deg=7):
    """Everything the kernel needs, resolved once and cached as plain attrs.

    This is the single biggest cost in decode, and it is not arithmetic.
    Profiling an 8B on an A100: 253 matmuls per token, 43 ms of device time,
    and 149 ms of wall time -- about 106 ms per token spent in Python. Every
    `store.packed` / `store.scale` / `store.cb` goes through
    `nn.Module.__getattr__`, which walks `_parameters`, `_buffers` and
    `_modules` on each access, and `packed_2d()` builds a fresh view each call.
    Multiply by 253 calls per token and it dominates the kernel entirely.

    Cached in a plain dict on the store, so lookups are attribute-free after
    the first call. Invalidated by `store._hot_cache = None` if a buffer is
    ever replaced -- which fine-tuning does when it promotes scale/cb to
    Parameters.
    """
    h = store.__dict__.get("_hot_cache")
    if h is not None and h[0] is store.cb:
        return h

    coef = torch.from_numpy(
        _triton.fit_codebook_poly(store.cb.detach().cpu().numpy(), deg)
    ).to(store.cb.device)
    h = (store.cb, store.packed_2d(), store.scale, coef, store.group,
         4 if store.levels <= 16 else 8, store.out_features,
         store.in_features, int(store.cb.numel()))
    store.__dict__["_hot_cache"] = h
    return h


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

    _cb, packed, scale, coef, group, bits, n, k, levels = _hot(store)
    if MODE == "poly" and hasattr(_triton, "dequant_poly"):
        try:
            return _triton.dequant_poly(x, packed, scale, coef, group, bits,
                                        n, k, levels)
        except Exception:
            # Never let an optimisation break a model: fall through to the
            # reference path rather than raising in the middle of a forward.
            pass
    return _triton.dequant_matmul(x, packed, scale, _cb, group, bits, n, k)
