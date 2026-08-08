"""`.ebin` reader.

Reads the container and hands back weights. There is deliberately no writer
here: this package runs models, it does not produce them, and it does not
convert them back to any other format.

Layout
------
    MAGIC (8 bytes, "EBIN0001")
    payload blobs, concatenated
    header JSON
    u64 little-endian offset of the header

The header sits at the end so a writer can stream the payload without knowing
final offsets; a reader seeks to the last 8 bytes to find it.

Each compressed tensor stores three blobs — the codebook, per-group scales, and
an entropy-coded index stream — plus the symbol frequencies the coder was built
from. The frequencies are integer counts rather than floats because decoding is
only reversible if both sides construct an identical probability model.
"""
import json
import os
import struct

import numpy as np

MAGIC = b"EBIN0001"


def _model(counts):
    """Rebuild the coder's probability model from stored integer counts."""
    import constriction

    p = np.asarray(counts, dtype=np.float64)
    p = np.maximum(p, 1.0)
    return constriction.stream.model.Categorical(p / p.sum(), perfect=False)


def rans_decode(blob, chunks, counts):
    """Decode an index stream that was coded in chunks."""
    import constriction

    model = _model(counts)
    parts, off = [], 0
    for nbytes, nsym in chunks:
        words = np.frombuffer(blob[off:off + nbytes], dtype=np.uint32)
        coder = constriction.stream.stack.AnsCoder(words)
        parts.append(coder.decode(model, nsym).astype(np.uint8))
        off += nbytes
    return np.concatenate(parts) if parts else np.zeros(0, np.uint8)


def read_header(path):
    with open(path, "rb") as f:
        if f.read(8) != MAGIC:
            raise ValueError(f"{path} is not an .ebin file")
        f.seek(-8, os.SEEK_END)
        off = struct.unpack("<Q", f.read(8))[0]
        end = f.seek(0, os.SEEK_END) - 8
        f.seek(off)
        return json.loads(f.read(end - off))


def read_blob(f, rec):
    f.seek(rec[0])
    return f.read(rec[1])


def dequantize(idx, scale, cb, shape, group):
    """Rebuild float32 weights from indices, per-group scales and codebook.

    Handles rank>2: mixture-of-experts keeps every expert in one stacked tensor,
    quantized as a tall 2-D matrix over the last axis.
    """
    shape = tuple(shape)
    n_in = shape[-1]
    rows = 1
    for d in shape[:-1]:
        rows *= d
    idx = idx.reshape(rows, n_in)
    scale = np.asarray(scale, np.float32)

    w = np.empty((rows, n_in), np.float32)
    for gi, start in enumerate(range(0, n_in, group)):
        end = min(start + group, n_in)
        w[:, start:end] = cb[idx[:, start:end]] * scale[:, gi:gi + 1]
    return w.reshape(shape)


# There is deliberately no `iter_tensors`-style helper here that walks a file
# and hands back every tensor as a dense array. `apply_to` never needs one: it
# builds packed stores and lets the kernels decode inside the matmul, so the
# dense weight is never assembled. A bulk dense dump would turn one call into a
# converter back to an ordinary checkpoint, which is the opposite of what this
# format is for. `dequantize` above stays because a few tensor kinds (routers,
# unsupported module types) are handed back dense at load time.


def extras(path):
    """Config and tokenizer files carried inside the container."""
    header = read_header(path)
    with open(path, "rb") as f:
        return {k: read_blob(f, rec) for k, rec in header.get("extras", {}).items()}


def describe(path):
    """Summary for `epure info`."""
    h = read_header(path)
    n_q = sum(1 for t in h["tensors"] if t["kind"] == "epure")
    return {
        "path": path,
        "format": h["format"],
        "levels": h["levels"],
        "group": h["group"],
        "compressed_tensors": n_q,
        "raw_tensors": len(h["tensors"]) - n_q,
        "size_bytes": os.path.getsize(path),
        "bundled": sorted(h.get("extras", {})),
    }
