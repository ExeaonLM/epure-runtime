"""Fine-tune a compressed model without ever decompressing it.

A packed weight is `codebook[index] * scale`. The indices are frozen, so that
expression is smoothly differentiable in the codebook and the scales — ordinary
backprop, no straight-through estimator, no gradient surgery around a rounding
step. Only the two small continuous tensors move.

What that costs, per tensor:

    codebook   levels floats                     (16-32 values)
    scales     rows x ceil(in/group) floats      (~1% of the model)
    indices    frozen                            (the bulk of the file)

So a 4B model exposes roughly 28M trainable values. Gradients are held for those
only, which is what makes this feasible on hardware that could not fine-tune the
dense model at all. And because the indices never change, the result ships as
the same `.ebin` with updated scales — no adapter file, no merge step, no
dequantized intermediate at any point.

Modes
-----
    scale      per-group scales only  — smallest footprint, most constrained
    codebook   codebook levels only   — very few parameters, global effect
    both       recommended default

Training runs through `materialize()`, which is differentiable. Inference runs
through the fused kernels, which are not. That split is deliberate: the slow
path is only used while learning.

Capacity is the open question. This is expected to be enough for identity, tone
and format; whether it can absorb genuinely new capability is empirical, and
`verify_frozen` plus a benchmark run is how you find out rather than assume.
"""
import torch
import torch.nn as nn

from .runtime import PackedEmbedding, PackedLinear, PackedWeight

MODES = ("scale", "codebook", "both")


def packed_weights(model):
    """Every PackedWeight in the model, with the path that owns it."""
    for name, mod in model.named_modules():
        if isinstance(mod, (PackedLinear, PackedEmbedding)):
            yield name, mod.store


def make_trainable(model, mode="both", verbose=True):
    """Promote codebook and/or scale buffers to trainable parameters.

    Everything else — including every index — is left frozen. Returns the list
    of parameters to hand an optimizer.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")

    for p in model.parameters():
        p.requires_grad_(False)

    params, n_tensors = [], 0
    for _, store in packed_weights(model):
        # Buffers cannot carry gradients; re-register as parameters. The packed
        # indices stay a buffer and stay frozen.
        if mode in ("scale", "both"):
            sc = store.scale.detach().float().clone()
            del store._buffers["scale"]
            store.scale = nn.Parameter(sc, requires_grad=True)
            params.append(store.scale)
        if mode in ("codebook", "both"):
            cb = store.cb.detach().float().clone()
            del store._buffers["cb"]
            store.cb = nn.Parameter(cb, requires_grad=True)
            params.append(store.cb)
        n_tensors += 1

    n_train = sum(p.numel() for p in params)
    n_total = n_train + sum(
        s.packed.numel() * (2 if s.levels <= 16 else 1)
        for _, s in packed_weights(model))
    if verbose:
        print(f"  mode {mode}: {n_tensors} packed tensors, "
              f"{n_train/1e6:.2f}M trainable "
              f"({100*n_train/max(n_total,1):.2f}% of weight values)")
    return params


def snapshot_indices(model):
    """Checksum every index buffer, so drift can be proven absent later."""
    return {name: int(store.packed.sum().item())
            for name, store in packed_weights(model)}


def verify_frozen(model, snapshot):
    """Confirm training changed no index. Raises if any did.

    Worth checking explicitly: if indices moved, the artifact is no longer the
    file we shipped and the compression guarantee is void.
    """
    now = snapshot_indices(model)
    moved = [k for k, v in snapshot.items() if now.get(k) != v]
    if moved:
        raise RuntimeError(
            f"{len(moved)} index buffers changed during training "
            f"(first: {moved[0]}). Indices must stay frozen.")
    return True


def trainable_forward(model, enable=True):
    """Force the differentiable path.

    The fused kernels are custom ops with no backward, so training must go
    through `materialize()`. Inference switches back automatically.
    """
    for _, store in packed_weights(model):
        store._force_materialize = enable
    return model


class ScaleTuner:
    """Minimal training loop for codebook/scale adaptation.

    Deliberately small: this exists to answer whether the approach works, not
    to replace a training framework.
    """

    def __init__(self, model, mode="both", lr=1e-4, device=None):
        self.model = model
        self.device = device or next(
            (p.device for p in model.parameters()), torch.device("cpu"))
        self.params = make_trainable(model, mode)
        self.snapshot = snapshot_indices(model)
        self.opt = torch.optim.AdamW(self.params, lr=lr)
        trainable_forward(model, True)

    def step(self, input_ids, labels=None):
        labels = input_ids if labels is None else labels
        out = self.model(input_ids.to(self.device), labels=labels.to(self.device))
        out.loss.backward()
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        return float(out.loss.detach())

    def finish(self):
        """Stop training and prove the indices never moved."""
        verify_frozen(self.model, self.snapshot)
        trainable_forward(self.model, False)
        for p in self.params:
            p.requires_grad_(False)
        return self.model
