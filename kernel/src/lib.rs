//! Fused dequantize-matmul for CPU.
//!
//! The Python runtime rebuilds each dense weight before multiplying. On a GPU
//! that cost us 8.5x; on a CPU it is worse, because CPU memory bandwidth is an
//! order of magnitude scarcer and there is no tensor-core headroom to hide it.
//!
//! But CPU is where compression should *win*. Decode is bandwidth-bound:
//! reading 2 GB of packed weights instead of 8 GB of fp32 is a direct saving,
//! and the dequantization it costs is paid in compute — which a 32-core EPYC
//! has in abundance and bandwidth it does not. The trade that lost on a T4 is
//! the right trade here.
//!
//! Design notes
//! ------------
//! * **Per-group lookup table.** Codebook entries are pre-multiplied by the
//!   group scale into a 16-entry table that lives in L1. The inner loop is then
//!   a table lookup and an FMA, not a divide and a gather from main memory.
//! * **Weights stream once.** The activation stays resident while a row of
//!   packed weights is read straight through — the access pattern bandwidth
//!   actually rewards.
//! * **Parallel over output rows.** Rows are independent, so rayon splits them
//!   with no synchronisation and no false sharing (each row owns its output).
//! * **4-bit and 8-bit indices** share one path, branching once per row rather
//!   than per element.
//!
//! Targets AVX2 + FMA (Zen 3 has no AVX-512). The loops are written so LLVM
//! can vectorize them; explicit intrinsics are a later optimization to be
//! justified by measurement, not assumed.

use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Build the per-group table: codebook value * group scale, for every level.
#[inline(always)]
fn build_lut(cb: &[f32], scale: f32, lut: &mut [f32]) {
    for (i, c) in cb.iter().enumerate() {
        lut[i] = c * scale;
    }
}

/// One output row: dot(x, dequant(packed_row)).
///
/// `m` activations are handled together so the packed row is read once for the
/// whole batch rather than once per sequence element.
#[allow(clippy::too_many_arguments)]
fn row_dot(
    x: &[f32],      // [m, k]
    m: usize,
    k: usize,
    packed_row: &[u8],
    scales_row: &[f32],
    cb: &[f32],
    group: usize,
    four_bit: bool,
    out: &mut [f32], // [m]
) {
    let levels = cb.len();
    let mut lut = [0f32; 256];

    for (gi, chunk_start) in (0..k).step_by(group).enumerate() {
        let chunk_end = (chunk_start + group).min(k);
        build_lut(cb, scales_row[gi], &mut lut[..levels]);

        for mi in 0..m {
            let xrow = &x[mi * k..mi * k + k];
            let mut acc = 0f32;

            if four_bit {
                // two indices per byte; even index in the low nibble
                let mut kk = chunk_start;
                while kk < chunk_end {
                    let byte = packed_row[kk >> 1];
                    acc += xrow[kk] * lut[(byte & 0x0F) as usize];
                    if kk + 1 < chunk_end {
                        acc += xrow[kk + 1] * lut[(byte >> 4) as usize];
                    }
                    kk += 2;
                }
            } else {
                for kk in chunk_start..chunk_end {
                    acc += xrow[kk] * lut[packed_row[kk] as usize];
                }
            }
            out[mi] += acc;
        }
    }
}

/// out[m, n] = x[m, k] @ dequant(packed)[n, k].T
///
/// `packed` is [n, k] for 8-bit indices or [n, k/2] for 4-bit.
/// `scales` is [n, ceil(k/group)].
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn dequant_matmul<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    packed: PyReadonlyArray2<'py, u8>,
    scales: PyReadonlyArray2<'py, f32>,
    cb: PyReadonlyArray1<'py, f32>,
    group: usize,
    four_bit: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let x = x.as_slice()?;
    let packed_arr = packed.as_array();
    let scales_arr = scales.as_array();
    let cb = cb.as_slice()?;

    let n = packed_arr.shape()[0];
    let row_bytes = packed_arr.shape()[1];
    let k = if four_bit { row_bytes * 2 } else { row_bytes };
    let m = x.len() / k;
    let n_groups = scales_arr.shape()[1];

    let packed_flat = packed_arr.as_slice().expect("packed must be contiguous");
    let scales_flat = scales_arr.as_slice().expect("scales must be contiguous");

    // Rows are independent: each writes its own slice of the output, so there
    // is no sharing to synchronise and no false sharing to avoid.
    let mut out = vec![0f32; m * n];
    out.par_chunks_mut(m)
        .enumerate()
        .for_each(|(row, out_row)| {
            row_dot(
                x,
                m,
                k,
                &packed_flat[row * row_bytes..(row + 1) * row_bytes],
                &scales_flat[row * n_groups..(row + 1) * n_groups],
                cb,
                group,
                four_bit,
                out_row,
            );
        });

    // Accumulated row-major over n (one row per thread, no false sharing);
    // the caller wants [m, n], so transpose on the way out. For decode m is 1
    // and this is a straight copy.
    let mut transposed = vec![0f32; m * n];
    for row in 0..n {
        for mi in 0..m {
            transposed[mi * n + row] = out[row * m + mi];
        }
    }

    let arr = ndarray::Array2::from_shape_vec((m, n), transposed)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.to_pyarray(py))
}

/// Threads rayon will use — reported so the caller can confirm the box is busy.
#[pyfunction]
fn threads() -> usize {
    rayon::current_num_threads()
}

#[pymodule]
fn epure_kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dequant_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(threads, m)?)?;
    Ok(())
}
