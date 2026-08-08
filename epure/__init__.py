"""epure-runtime — run compressed models.

Loads `.ebin` containers and serves them without ever assembling the dense
weight: indices stay packed in memory and are decoded inside the matmul.

    from epure import load
    model, tok = load("Exeaon/Exeaon-Dzo-1.7B")

This package contains no compressor. It reads the format and runs it.
"""
from .container import describe, extras, read_header
from .finetune import (
    ScaleTuner,
    make_trainable,
    packed_weights,
    snapshot_indices,
    trainable_forward,
    verify_frozen,
)
from .runtime import (
    PackedEmbedding,
    PackedLinear,
    PackedWeight,
    apply_to,
    load,
    resolve,
)

__version__ = "0.2.3"

__all__ = [
    "load", "apply_to", "resolve", "describe", "extras", "read_header",
    "PackedWeight", "PackedLinear", "PackedEmbedding",
    "make_trainable", "packed_weights", "snapshot_indices", "verify_frozen",
    "trainable_forward", "ScaleTuner",
    "__version__",
]
