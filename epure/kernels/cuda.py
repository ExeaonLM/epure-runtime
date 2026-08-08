"""CUDA backend — the fused Triton kernel.

Dequantization happens inside the GEMM: a tile of packed indices is loaded,
expanded through the codebook in registers and fed to ``tl.dot`` without the
dense tile ever reaching global memory.

Honest note on where this wins. On a GPU with plenty of bandwidth and a very
well tuned vendor GEMM, this is currently *slower* per token than dense fp16 —
the win here is footprint, which is what decides whether the model runs on the
card at all. On CPU, where bandwidth is scarce, the fused path wins outright.
"""
import torch

try:
    from . import _triton
    HAVE_TRITON = _triton.HAVE_TRITON
except Exception:                       # pragma: no cover - triton optional
    _triton = None
    HAVE_TRITON = False


def available():
    return HAVE_TRITON and torch.cuda.is_available()


def linear(x, store):
    """``x @ dequant(store).T`` on CUDA, or ``None`` if unsupported."""
    if not available() or not x.is_cuda or len(store.shape) != 2:
        return None
    # The kernel accumulates in fp32 and writes fp16; an fp32 activation would
    # be silently narrowed, so hand those back to the fallback instead.
    if x.dtype != torch.float16:
        return None

    bits = 4 if store.levels <= 16 else 8
    return _triton.dequant_matmul(
        x, store.packed_2d(), store.scale, store.cb,
        store.group, bits, store.out_features, store.in_features,
    )
