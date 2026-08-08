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
