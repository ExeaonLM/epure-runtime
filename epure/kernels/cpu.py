"""CPU backend — the Rust fused kernel.

This is the path that actually beats a dense baseline on CPU. It walks the
packed rows, expands each group through a small lookup table and accumulates,
so the dense weight never exists. Parallelised over output rows with rayon.

Built separately (see the README); if the extension is absent every call
returns ``None`` and the runtime falls back to materializing. That is slower but
correct, so a plain ``pip install`` still works with no toolchain.
"""
import numpy as np
import torch

try:
    # Ships inside the wheel as `epure.epure_kernel` (maturin builds it there).
    # The bare `epure_kernel` fallback covers a local `cargo build` during
    # development, where the artifact is dropped on the path instead.
    from .. import epure_kernel as _rust
    HAVE_RUST = True
except ImportError:                     # pragma: no cover - depends on build
    try:
        import epure_kernel as _rust
        HAVE_RUST = True
    except ImportError:
        _rust = None
        HAVE_RUST = False


def available():
    return HAVE_RUST


def threads():
    return _rust.threads() if HAVE_RUST else 0


def linear(x, store, cache=None):
    """``x @ dequant(store).T`` on CPU, or ``None`` if unsupported.

    The codebook and scales are converted once per store and cached: they are a
    few kilobytes each, but rebuilding them per token showed up as real time at
    batch one, where there is nothing else to hide behind.
    """
    if not HAVE_RUST or x.is_cuda or store.levels > 16:
        return None
    if len(store.shape) != 2:
        return None

    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    # The extension is compiled for f32 only. Converting a bf16/f16 activation
    # here would cost more than the kernel saves, so decline instead.
    if x2.dtype != torch.float32:
        return None
    x_np = np.ascontiguousarray(x2.detach().numpy())

    key = id(store)
    entry = cache.get(key) if cache is not None else None
    if entry is None:
        entry = (
            np.ascontiguousarray(store.packed_2d().numpy()),
            np.ascontiguousarray(store.scale.to(torch.float32).numpy()),
            np.ascontiguousarray(store.cb.to(torch.float32).numpy()).ravel(),
        )
        if cache is not None:
            cache[key] = entry
    packed, scale, cb = entry

    out = _rust.dequant_matmul(x_np, packed, scale, cb, store.group, True)
    return torch.from_numpy(out).reshape(*shape[:-1], store.out_features)
