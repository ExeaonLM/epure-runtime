"""Fused dequantize-matmul — the weight is never built.

Why this exists
---------------
The plain runtime rebuilds each dense weight inside every forward. Measured on
Qwen3-4B/T4 that made decode ~8.5x slower than fp16, which turns a 2x memory win
into roughly 4x worse cost per token.

Removing the per-group Python loop changed nothing (1.6 -> 1.6 tok/s), which
ruled out launch overhead and pointed at the materialization itself. Per token
the old path moved, for ~3.6B weights:

    unpack       write 3.6 GB of uint8 indices
    .long()      write 28.8 GB of int64  <-- indexing wants 8 bytes per index
    codebook     write 7.2 GB of fp16 weights
    matmul       read them all back

Roughly 40 GB of traffic to do 7 GB of arithmetic. This kernel dequantizes
inside the matmul: indices are unpacked into registers, scaled, and multiplied
against the activation tile while resident, so no dense weight and no int64
index array is ever written to memory.

Supports 4-bit packed (<=16 levels, two per byte) and 8-bit (>16 levels) via a
compile-time flag, so both profiles use the same path.
"""
import os

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                   # noqa: BLE001
    HAVE_TRITON = False


if HAVE_TRITON:

    # Decode (M=1..8) and prefill (M in the hundreds) want completely different
    # tiles. One hand-picked shape tuned for decode cost prefill 1527 -> 533
    # tok/s at batch 8. Autotuning lets Triton pick per shape; the cost is a
    # one-off compile per (M,N,K) bucket, amortized immediately.
    _CONFIGS = [
        triton.Config({'BM': 16, 'BN': 64, 'BK': 64}, num_warps=4, num_stages=3),
        triton.Config({'BM': 16, 'BN': 128, 'BK': 64}, num_warps=4, num_stages=3),
        triton.Config({'BM': 32, 'BN': 64, 'BK': 64}, num_warps=4, num_stages=3),
        triton.Config({'BM': 64, 'BN': 64, 'BK': 32}, num_warps=4, num_stages=3),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 32}, num_warps=8, num_stages=3),
        triton.Config({'BM': 128, 'BN': 64, 'BK': 32}, num_warps=8, num_stages=3),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 32}, num_warps=8, num_stages=2),
    ]

    @triton.autotune(configs=_CONFIGS, key=['M', 'N', 'K'])
    @triton.jit
    def _dequant_matmul(
        X, PACKED, SCALE, CB, OUT,
        M, N, K,
        stride_xm, stride_xk,
        stride_pn, stride_pk,
        stride_sn, stride_sk,
        stride_om, stride_on,
        GROUP: tl.constexpr, BITS: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        """out[M,N] = x[M,K] @ dequant(packed)[N,K].T"""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)

        acc = tl.zeros((BM, BN), dtype=tl.float32)

        for k0 in range(0, K, BK):
            k = k0 + offs_k

            x = tl.load(X + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
                        mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)

            if BITS == 4:
                # two indices per byte; even k in the low nibble
                byte = tl.load(
                    PACKED + offs_n[:, None] * stride_pn + (k[None, :] // 2) * stride_pk,
                    mask=(offs_n[:, None] < N) & (k[None, :] < K), other=0)
                shift = (k[None, :] % 2) * 4
                idx = (byte >> shift) & 0x0F
            else:
                idx = tl.load(
                    PACKED + offs_n[:, None] * stride_pn + k[None, :] * stride_pk,
                    mask=(offs_n[:, None] < N) & (k[None, :] < K), other=0)

            # codebook lookup and per-group scale, both in registers
            w = tl.load(CB + idx.to(tl.int32))
            s = tl.load(
                SCALE + offs_n[:, None] * stride_sn + (k[None, :] // GROUP) * stride_sk,
                mask=(offs_n[:, None] < N) & (k[None, :] < K), other=0.0)
            w = w * s

            # fp16 operands: Turing (sm_75) tensor cores do not take fp32 into
            # tl.dot. Accumulation stays fp32, which is where precision matters.
            acc += tl.dot(x.to(tl.float16), tl.trans(w.to(tl.float16)))

        out = acc.to(tl.float16)
        tl.store(OUT + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
                 out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


if HAVE_TRITON:

    @triton.autotune(configs=_CONFIGS, key=['M', 'N', 'K'])
    @triton.jit
    def _dequant_poly(
        X, PACKED, SCALE, COEF, OUT,
        M, N, K,
        stride_xm, stride_xk,
        stride_pn, stride_pk,
        stride_sn, stride_sk,
        stride_om, stride_on,
        GROUP: tl.constexpr, BITS: tl.constexpr, DEG: tl.constexpr,
        LEVELS: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        """Codebook as a polynomial, not a table.

        `tl.load(CB + idx)` is a gather: one indexed load per weight, which
        cannot coalesce and measured at roughly half this kernel's runtime on an
        A100. Marlin avoids the problem by making dequantization arithmetic, but
        that only works for an affine int4 grid.

E-PURE codebooks are smooth by construction, so they can be
        approximated by a low-order polynomial in the index instead. Measured
        across 253 codebooks from a shipped 4B model, degree 7 reproduces them
        to 1.3% of the spacing between adjacent levels, against a quantization
        error of +/-50% of a step, and costs +0.047% perplexity end to end. The
        table is therefore replaceable by Horner evaluation: DEG fused
        multiply-adds and no memory access at all.

        Coefficients are fitted from whatever codebook the container carries,
        so this needs nothing from the compressor.

        Coefficients are per tensor, highest order first, and live in registers.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)

        for k0 in range(0, K, BK):
            k = k0 + offs_k
            x = tl.load(X + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
                        mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)

            if BITS == 4:
                byte = tl.load(
                    PACKED + offs_n[:, None] * stride_pn + (k[None, :] // 2) * stride_pk,
                    mask=(offs_n[:, None] < N) & (k[None, :] < K), other=0)
                idx = (byte >> ((k[None, :] % 2) * 4)) & 0x0F
            else:
                idx = tl.load(
                    PACKED + offs_n[:, None] * stride_pn + k[None, :] * stride_pk,
                    mask=(offs_n[:, None] < N) & (k[None, :] < K), other=0)

            # index -> [-1, 1], the domain the fit was made on
            t = idx.to(tl.float32) * (2.0 / (LEVELS - 1)) - 1.0

            # Horner. Unrolled by the compiler since DEG is constexpr, so this
            # is DEG FMAs in registers and nothing touches memory.
            w = tl.load(COEF)
            for d in tl.static_range(1, DEG + 1):
                w = w * t + tl.load(COEF + d)

            sc = tl.load(
                SCALE + offs_n[:, None] * stride_sn + (k[None, :] // GROUP) * stride_sk,
                mask=(offs_n[:, None] < N) & (k[None, :] < K), other=0.0)
            w = w * sc

            acc += tl.dot(x.to(tl.float16), tl.trans(w.to(tl.float16)))

        tl.store(OUT + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
                 acc.to(tl.float16),
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


    @triton.autotune(configs=_CONFIGS, key=['M', 'N', 'K'])
    @triton.jit
    def _dequant_nolut(
        X, PACKED, SCALE, CB, OUT,
        M, N, K,
        stride_xm, stride_xk,
        stride_pn, stride_pk,
        stride_sn, stride_sk,
        stride_om, stride_on,
        GROUP: tl.constexpr, BITS: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        """DIAGNOSTIC ONLY - wrong maths, right memory traffic.

        Identical to the real kernel except the codebook gather is replaced by
        an affine map on the index. It reads exactly the same bytes and does the
        same tl.dot, so the gap between this and the real kernel is the cost of
        `tl.load(CB + idx)` and nothing else.

        The real kernel sits at 5-8% of memory bandwidth while dense fp16 hits
        90%, and split-K barely moved it -- so the loss is per element, not per
        block. This isolates whether the per-element cost is the lookup.

        NOT a correctness path. Never dispatched.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)

        for k0 in range(0, K, BK):
            k = k0 + offs_k
            x = tl.load(X + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
                        mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)
            idx = tl.load(
                PACKED + offs_n[:, None] * stride_pn + k[None, :] * stride_pk,
                mask=(offs_n[:, None] < N) & (k[None, :] < K), other=0)
            sc = tl.load(
                SCALE + offs_n[:, None] * stride_sn + (k[None, :] // GROUP) * stride_sk,
                mask=(offs_n[:, None] < N) & (k[None, :] < K), other=0.0)
            # affine instead of gather: same traffic, no table access
            w = (idx.to(tl.float32) * 0.0625 - 1.0) * sc
            acc += tl.dot(x.to(tl.float16), tl.trans(w.to(tl.float16)))

        tl.store(OUT + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
                 acc.to(tl.float16),
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


    _SPLITK_CONFIGS = [
        triton.Config({'BM': 16, 'BN': 64, 'BK': 64}, num_warps=4, num_stages=3),
        triton.Config({'BM': 16, 'BN': 128, 'BK': 64}, num_warps=4, num_stages=3),
        triton.Config({'BM': 16, 'BN': 256, 'BK': 32}, num_warps=8, num_stages=3),
        triton.Config({'BM': 32, 'BN': 128, 'BK': 64}, num_warps=8, num_stages=3),
    ]

    @triton.autotune(configs=_SPLITK_CONFIGS, key=['M', 'N', 'K', 'SPLIT_K'])
    @triton.jit
    def _dequant_splitk(
        X, PACKED, SCALE, CB, PARTIAL,
        M, N, K,
        stride_xm, stride_xk,
        stride_pn, stride_pk,
        stride_sn, stride_sk,
        stride_ps, stride_pm, stride_po,
        GROUP: tl.constexpr, BITS: tl.constexpr, SPLIT_K: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        """Same maths, K partitioned across programs.

        At batch 1 the plain GEMM grid is 1 x cdiv(N, BN) -- 32 blocks for a
        4096-wide layer, on a card with 108 SMs. Two thirds of the GPU idles
        while each block walks the whole of K serially, which is why the kernel
        measured at 6-10% of memory bandwidth while dense hit 90%.

        Splitting K multiplies the grid by SPLIT_K so the weight read is shared
        across many blocks. Each program writes its own slice of a partial
        buffer; the reduction is a second pass. An earlier attempt used fp32
        atomics AND replaced tl.dot with a broadcast outer product, which
        bypassed the tensor cores and measured worse -- the lesson was that the
        wasted rows are cheap and the reduction is not, so tl.dot stays.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_k = tl.program_id(2)

        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)

        acc = tl.zeros((BM, BN), dtype=tl.float32)

        # Interleaved, not contiguous: consecutive programs touch neighbouring
        # K tiles, so the weight rows they pull share cache lines.
        k_start = pid_k * BK
        k_step = SPLIT_K * BK

        for k0 in range(k_start, K, k_step):
            k = k0 + offs_k

            x = tl.load(X + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
                        mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)

            if BITS == 4:
                byte = tl.load(
                    PACKED + offs_n[:, None] * stride_pn + (k[None, :] // 2) * stride_pk,
                    mask=(offs_n[:, None] < N) & (k[None, :] < K), other=0)
                idx = (byte >> ((k[None, :] % 2) * 4)) & 0x0F
            else:
                idx = tl.load(
                    PACKED + offs_n[:, None] * stride_pn + k[None, :] * stride_pk,
                    mask=(offs_n[:, None] < N) & (k[None, :] < K), other=0)

            w = tl.load(CB + idx.to(tl.int32))
            sc = tl.load(
                SCALE + offs_n[:, None] * stride_sn + (k[None, :] // GROUP) * stride_sk,
                mask=(offs_n[:, None] < N) & (k[None, :] < K), other=0.0)
            w = w * sc

            acc += tl.dot(x.to(tl.float16), tl.trans(w.to(tl.float16)))

        tl.store(
            PARTIAL + pid_k * stride_ps + offs_m[:, None] * stride_pm
            + offs_n[None, :] * stride_po,
            acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


    @triton.jit
    def _dequant_gemv(
        X, PACKED, SCALE, CB, OUT,
        M, N, K,
        stride_xm, stride_xk,
        stride_pn, stride_pk,
        stride_sn, stride_sk,
        stride_om, stride_on,
        GROUP: tl.constexpr, BITS: tl.constexpr,
        MP: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        """Decode-shaped path: few rows, split over the reduction dimension.

        At batch 1 a GEMM kernel computes BM rows to keep one — 16x wasted work.
        Here each program owns a slice of K and a tile of N, and partial sums are
        combined with atomics. Work scales with the rows actually wanted, and the
        weight read is split across many blocks so the card stays busy.
        """
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        nsplit = tl.num_programs(1)

        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_m = tl.arange(0, MP)
        offs_k = tl.arange(0, BK)

        k_chunk = tl.cdiv(K, nsplit)
        k_lo = pid_k * k_chunk
        k_hi = tl.minimum(k_lo + k_chunk, K)

        acc = tl.zeros((MP, BN), dtype=tl.float32)

        for k0 in range(k_lo, k_hi, BK):
            k = k0 + offs_k
            kmask = k < k_hi

            x = tl.load(X + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
                        mask=(offs_m[:, None] < M) & kmask[None, :], other=0.0)

            if BITS == 4:
                byte = tl.load(
                    PACKED + offs_n[:, None] * stride_pn + (k[None, :] // 2) * stride_pk,
                    mask=(offs_n[:, None] < N) & kmask[None, :], other=0)
                idx = (byte >> ((k[None, :] % 2) * 4)) & 0x0F
            else:
                idx = tl.load(
                    PACKED + offs_n[:, None] * stride_pn + k[None, :] * stride_pk,
                    mask=(offs_n[:, None] < N) & kmask[None, :], other=0)

            w = tl.load(CB + idx.to(tl.int32))
            s = tl.load(
                SCALE + offs_n[:, None] * stride_sn + (k[None, :] // GROUP) * stride_sk,
                mask=(offs_n[:, None] < N) & kmask[None, :], other=0.0)
            w = (w * s).to(tl.float32)

            # M is tiny, so a broadcast multiply-reduce beats a tensor-core dot
            # and avoids padding the M dimension up to a tile.
            acc += tl.sum(x.to(tl.float32)[:, None, :] * w[None, :, :], axis=2)

        tl.atomic_add(OUT + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
                      acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def dequant_gemv(x, packed, scale, cb, group, bits, n_out, k_in, split=8):
    """Small-M path. Output is accumulated with atomics, so it starts zeroed."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    M = x2.shape[0]
    MP = 1 if M == 1 else (2 if M <= 2 else (4 if M <= 4 else 8))
    out = torch.zeros((M, n_out), device=x.device, dtype=torch.float32)

    BN, BK = 64, 64
    grid = (triton.cdiv(n_out, BN), split)
    _dequant_gemv[grid](
        x2, packed, scale, cb, out,
        M, n_out, k_in,
        x2.stride(0), x2.stride(1),
        packed.stride(0), packed.stride(1),
        scale.stride(0), scale.stride(1),
        out.stride(0), out.stride(1),
        GROUP=group, BITS=bits, MP=MP, BN=BN, BK=BK,
        num_warps=4, num_stages=3,
    )
    return out.to(torch.float16).reshape(*shape[:-1], n_out)


def fit_codebook_poly(cb, deg=7):
    """Least-squares fit of a codebook by a polynomial in the index.

    Returns coefficients highest-order first, for Horner. Done once per tensor
    at load time; the kernel then never reads the codebook again.
    """
    import numpy as np

    c = np.asarray(cb, dtype=np.float64).ravel()
    c = np.sort(c)
    n = len(c)
    t = 2.0 * np.arange(n) / (n - 1) - 1.0
    return np.polyfit(t, c, deg).astype(np.float32)


def dequant_poly(x, packed, scale, coef, group, bits, n_out, k_in, levels):
    """Polynomial dequantization: no codebook gather at all."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    M = x2.shape[0]
    out = torch.empty((M, n_out), device=x.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(M, meta['BM']),
                         triton.cdiv(n_out, meta['BN']))
    _dequant_poly[grid](
        x2, packed, scale, coef, out, M, n_out, k_in,
        x2.stride(0), x2.stride(1), packed.stride(0), packed.stride(1),
        scale.stride(0), scale.stride(1), out.stride(0), out.stride(1),
        GROUP=group, BITS=bits, DEG=coef.numel() - 1, LEVELS=levels)
    return out.reshape(*shape[:-1], n_out)


def dequant_nolut(x, packed, scale, cb, group, bits, n_out, k_in):
    """Diagnostic dispatch for `_dequant_nolut`. Output is not correct."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    M = x2.shape[0]
    out = torch.empty((M, n_out), device=x.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(M, meta['BM']),
                         triton.cdiv(n_out, meta['BN']))
    _dequant_nolut[grid](
        x2, packed, scale, cb, out, M, n_out, k_in,
        x2.stride(0), x2.stride(1), packed.stride(0), packed.stride(1),
        scale.stride(0), scale.stride(1), out.stride(0), out.stride(1),
        GROUP=group, BITS=bits)
    return out.reshape(*shape[:-1], n_out)


def dequant_splitk(x, packed, scale, cb, group, bits, n_out, k_in, split=8):
    """Split-K dispatch: partial buffer, then a reduction.

    Chosen when the grid would otherwise be too small to fill the card. The
    partial buffer is SPLIT_K x M x N floats -- 128 KB at M=1, N=4096, split=8
    -- so the memory cost is nothing next to the occupancy it buys.
    """
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    M = x2.shape[0]
    partial = torch.empty((split, M, n_out), device=x.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(M, meta['BM']),
                         triton.cdiv(n_out, meta['BN']), split)
    _dequant_splitk[grid](
        x2, packed, scale, cb, partial,
        M, n_out, k_in,
        x2.stride(0), x2.stride(1),
        packed.stride(0), packed.stride(1),
        scale.stride(0), scale.stride(1),
        partial.stride(0), partial.stride(1), partial.stride(2),
        GROUP=group, BITS=bits, SPLIT_K=split,
    )
    out = partial.sum(0).to(torch.float16)
    return out.reshape(*shape[:-1], n_out)


def dequant_matmul(x, packed, scale, cb, group, bits, n_out, k_in):
    """x @ W.T with W dequantized on the fly. x: [..., K] -> [..., N]."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    M = x2.shape[0]
    # NOTE: a split-K GEMV path (`dequant_gemv`, kept below) was tried here for
    # M<=8 on the theory that a tiled GEMM wastes 15 of 16 output rows at batch 1.
    # MEASURED WORSE: decode b1 8.8 -> 6.2, b8 72.2 -> 10.7 tok/s.
    #
    # The wasted rows are cheap; the reduction is not. `tl.sum(x[:,None,:] *
    # w[None,:,:])` is a broadcast outer product that bypasses tensor cores and
    # materializes an (M, BN, BK) intermediate, and the fp32 atomics contend
    # across splits. On this hardware a tensor-core GEMM with idle lanes beats a
    # non-tensor-core reduction that uses all of them.
    #
    # A GEMV worth having would keep `tl.dot` (padding M up to the tensor-core
    # tile) and reduce across splits in a second pass rather than by atomics.
    # Do not re-enable this dispatch without re-measuring.
    if os.environ.get("EBIN_GEMV") == "1" and M <= 8:
        return dequant_gemv(x, packed, scale, cb, group, bits, n_out, k_in)
    out = torch.empty((M, n_out), device=x.device, dtype=torch.float16)

    # Tile sizes come from autotuning, so the grid must be computed from the
    # config Triton selects rather than fixed here.
    grid = lambda meta: (triton.cdiv(M, meta['BM']),
                         triton.cdiv(n_out, meta['BN']))
    _dequant_matmul[grid](
        x2, packed, scale, cb, out,
        M, n_out, k_in,
        x2.stride(0), x2.stride(1),
        packed.stride(0), packed.stride(1),
        scale.stride(0), scale.stride(1),
        out.stride(0), out.stride(1),
        GROUP=group, BITS=bits,
    )
    return out.reshape(*shape[:-1], n_out)


def available():
    return HAVE_TRITON and torch.cuda.is_available()
