"""Load a `.ebin` as a runnable model, with weights left compressed.

The file *is* the model. Weights stay packed in memory; each layer expands only
what it needs, inside the fused kernel, and nothing writes a dense copy. A 4B
model that needs 7.6 GB in fp16 runs in about 2 GB here.

The trade is arithmetic for memory. That is the wrong trade on a modern GPU,
where bandwidth is plentiful and vendor GEMM kernels are extremely well tuned.
It is the right trade on CPU, where bandwidth is the binding constraint and
cores are not — measured 2.25-2.5x faster than dense fp32 on layers larger than
last-level cache, and slower on layers that fit inside it.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import container
from .kernels import cpu as cpu_kernel
from .kernels import cuda as cuda_kernel

_BUFFER_CACHE = {}

# Rows above which the dense path is used on CUDA. The crossover between "we
# read fewer bytes" and "cuBLAS has more arithmetic to work with" sits between
# batch 1 and batch 8 on an A100; 8 is the conservative end of that, and it is
# tunable because the crossover moves with model shape and card.
# Set EPURE_DENSE_ABOVE=999999 to always keep the small footprint.
_DENSE_ABOVE = int(os.environ.get("EPURE_DENSE_ABOVE", "8"))


def _pack4(idx, levels):
    """Two 4-bit indices per byte, up to 16 levels.

    Above 16 an index no longer fits in a nibble; packing anyway would truncate
    silently and produce a model that loads, runs, and is wrong.
    """
    idx = np.ascontiguousarray(idx, dtype=np.uint8).ravel()
    if levels > 16:
        return idx
    if idx.size % 2:
        idx = np.append(idx, 0)
    return (idx[0::2] | (idx[1::2] << 4)).astype(np.uint8)


def _unpack4(packed, n, levels):
    if levels > 16:
        return packed[:n]
    lo, hi = packed & 0x0F, packed >> 4
    out = torch.empty(packed.numel() * 2, dtype=torch.uint8, device=packed.device)
    out[0::2], out[1::2] = lo, hi
    return out[:n]


class PackedWeight(nn.Module):
    """Indices, per-group scales and a codebook — the resident representation."""

    def __init__(self, shape, packed, scale, cb, group, levels):
        super().__init__()
        self.levels, self.group = levels, group
        self.shape = tuple(shape)
        self.in_features = self.shape[-1]
        rows = 1
        for d in self.shape[:-1]:
            rows *= d
        self.out_features = rows          # experts fold into rows for rank>2
        self.register_buffer("packed", torch.from_numpy(packed), persistent=False)
        self.register_buffer("scale", torch.from_numpy(scale), persistent=False)
        self.register_buffer("cb", torch.from_numpy(cb), persistent=False)

    def packed_2d(self):
        cols = (self.in_features if self.levels > 16
                else (self.in_features + 1) // 2)
        return self.packed.view(self.out_features, cols)

    def materialize(self, dtype=torch.float16):
        """Expand to a dense tensor. Fallback only — the kernels avoid this."""
        n = self.out_features * self.in_features
        idx = _unpack4(self.packed, n, self.levels).view(
            self.out_features, self.in_features)
        # F.embedding accepts int32; plain indexing would promote to int64 and
        # write eight bytes per index to look up a table of a few dozen entries.
        cbd = self.cb.to(dtype).view(-1, 1).contiguous()
        w = F.embedding(idx.int(), cbd).squeeze(-1)
        sc = self.scale.to(dtype).repeat_interleave(self.group, dim=1)
        w *= sc[:, :self.in_features]
        return w.reshape(self.shape) if len(self.shape) > 2 else w


class PackedLinear(nn.Module):
    def __init__(self, store, bias=None):
        super().__init__()
        self.store = store
        self.bias = bias

    def forward(self, x):
        s = self.store
        if len(s.shape) == 2:
            out = None
            if x.is_cuda:
                # Which path wins depends on batch size, and it flips.
                #
                # Measured, Exeaon1-Nunya-8B on an A100: the fused kernel gives
                # 11.2 tok/s at batch 1 against 6.7 for materialize, because
                # decode is bandwidth-bound and we read a quarter of the bytes.
                # At batch 8 it gives 28 against 52, because there is now enough
                # arithmetic intensity for cuBLAS to dominate and dequantization
                # is pure overhead on top.
                #
                # Materializing costs memory (11.3 GB vs 8.0 GB peak on that
                # model), which is the thing this format exists to avoid -- so
                # the default keeps the fused path unless the batch is large
                # enough that the speed difference clearly outweighs it.
                if x.reshape(-1, x.shape[-1]).shape[0] < _DENSE_ABOVE:
                    out = cuda_kernel.linear(x, s)
            elif s.levels <= 16:
                out = cpu_kernel.linear(x, s, _BUFFER_CACHE)
            if out is not None:
                return out + self.bias if self.bias is not None else out

        w = s.materialize(x.dtype)
        out = F.linear(x, w, self.bias)
        del w
        return out


class PackedEmbedding(nn.Module):
    """Expands only the rows a batch actually asks for."""

    def __init__(self, store, dtype=torch.float16):
        super().__init__()
        self.store = store
        self.dtype = dtype
        self.num_embeddings = store.out_features
        self.embedding_dim = store.in_features

    def forward(self, ids):
        s = self.store
        flat = ids.reshape(-1)
        n_in = s.in_features
        if s.levels > 16:
            idx = s.packed.view(self.num_embeddings, n_in)[flat].to(torch.long)
        else:
            rows = s.packed.view(self.num_embeddings, n_in // 2)[flat]
            lo, hi = (rows & 0x0F).to(torch.long), (rows >> 4).to(torch.long)
            idx = torch.empty(flat.numel(), n_in, dtype=torch.long, device=lo.device)
            idx[:, 0::2], idx[:, 1::2] = lo, hi

        w = s.cb[idx].to(self.dtype)
        sc = s.scale[flat].to(self.dtype)
        for gi, start in enumerate(range(0, n_in, s.group)):
            end = min(start + s.group, n_in)
            w[:, start:end] *= sc[:, gi:gi + 1]
        return w.view(*ids.shape, n_in)

    @property
    def weight(self):
        return self.store.materialize(self.dtype)


class _ExpertHolder:
    """Keeps a stacked expert slab packed except while its module runs.

    Mixture-of-experts stores every expert in one 3-D parameter owned by a
    module that does its own batched matmul, so it cannot be wrapped as a
    linear. Attaching the dense tensor immediately before the call and dropping
    it after keeps the resident cost packed — experts are ~90% of such a model,
    so leaving them dense would give file compression and no memory saving.
    """

    def __init__(self, mod, attr, store, dtype=torch.float16):
        self.mod, self.attr, self.store = mod, attr, store
        self.dtype = dtype
        if attr in mod._parameters:
            del mod._parameters[attr]
        mod.register_forward_pre_hook(self._attach)
        mod.register_forward_hook(self._release)
        setattr(mod, attr, None)

    def _attach(self, mod, _a):
        setattr(mod, self.attr, self.store.materialize(self.dtype))

    def _release(self, mod, _a, _o):
        setattr(mod, self.attr, None)


def _get(root, dotted):
    cur = root
    for p in dotted.split("."):
        if not hasattr(cur, p):
            return None
        cur = getattr(cur, p)
    return cur


def _set(root, dotted, mod):
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], mod)


def _assign(root, dotted, tensor):
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], nn.Parameter(tensor, requires_grad=False))


def apply_to(model, path, device="cpu", cfg=None, verbose=True, dtype=None):
    """Replace a module tree's weights with packed stores, in place.

    `dtype` is the precision the restored tensors take. It defaults to whatever
    the model is already in, which matters because this runs *after* `load`
    has chosen a dtype: restoring everything as fp16 regardless silently undid
    that choice, and a bf16-trained model went back to overflowing.
    """
    if dtype is None:
        dtype = next((p.dtype for p in model.parameters()
                      if p.is_floating_point()), torch.float16)
    header = container.read_header(path)
    group, levels = header["group"], header["levels"]
    n_packed = n_dense = 0
    embed_store = None
    holders = []

    with open(path, "rb") as f:
        for t in header["tensors"]:
            name = t["name"]
            if t["kind"] == "raw":
                arr = np.frombuffer(container.read_blob(f, t["data"]),
                                    np.float16).reshape(t["shape"])
                _assign(model, name,
                        torch.from_numpy(arr.copy()).to(device=device,
                                                        dtype=dtype))
                n_dense += 1
                continue

            cb = np.frombuffer(container.read_blob(f, t["cb"]), np.float32).copy()
            scale = np.frombuffer(container.read_blob(f, t["scale"]), np.float16
                                  ).reshape(t["scale_shape"]).copy()
            idx = container.rans_decode(container.read_blob(f, t["idx"]),
                                        t["chunks"], t["counts"])
            store = PackedWeight(t["shape"], _pack4(idx, levels), scale, cb,
                                 group, levels).to(device)

            mod_name = name[:-7] if name.endswith(".weight") else name
            target = _get(model, mod_name)

            # `type(...) is` rather than isinstance: subclasses override forward
            # with their own signatures (a positional embedding may take
            # `past_key_values_length`) and wrapping them breaks the call.
            if type(target) is nn.Embedding or "embed_tokens" in name:
                embed_store = store
                _set(model, mod_name, PackedEmbedding(store, dtype))
                n_packed += 1
            elif isinstance(target, nn.Linear):
                _set(model, mod_name, PackedLinear(store, getattr(target, "bias", None)))
                n_packed += 1
            elif isinstance(target, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                # A convolution weight is rank 3 or 4 and would otherwise fall
                # into the mixture-of-experts branch below, which assumes the
                # leading axis indexes experts. Whisper's audio frontend is the
                # live case: conv1.weight is (1280, 128, 3), where the last axis
                # is a kernel of 3 - smaller than any sensible group size, so it
                # cannot be usefully quantized either. Hand it back dense.
                _assign(model, name, store.materialize(dtype).to(device))
                n_dense += 1
            elif len(t["shape"]) == 3:
                owner_path, _, attr = mod_name.rpartition(".")
                owner = _get(model, owner_path) if owner_path else model
                if owner is not None and (attr in getattr(owner, "_parameters", {})
                                          or hasattr(owner, attr)):
                    holders.append(_ExpertHolder(owner, attr, store, dtype))
                    n_packed += 1
                else:
                    _assign(model, name, store.materialize(dtype).to(device))
                    n_dense += 1
            else:
                # A mixture-of-experts router returns several values and cannot
                # be swapped for a linear. Routers are tiny; hand them back dense.
                _assign(model, name, store.materialize(dtype).to(device))
                n_dense += 1

    if cfg is not None and getattr(cfg, "tie_word_embeddings", False) and embed_store:
        _set(model, "lm_head", PackedLinear(embed_store))
        # Tying is done here, by hand. Leave the config flag set and anything
        # that later calls `tie_weights()` - lm_eval does - goes looking for
        # `model.embed_tokens.weight` as a Parameter. Ours is a property on a
        # packed module, so the lookup raises with "neither a parameter, buffer,
        # nor extra state" and the model cannot be evaluated at all.
        cfg.tie_word_embeddings = False
        if hasattr(model, "config"):
            model.config.tie_word_embeddings = False
        if verbose:
            print("  tied lm_head -> embedding store")
    if holders:
        model._epure_holders = holders
    if verbose:
        print(f"  {n_packed} packed + {n_dense} dense tensors from {path}")
    return model


def resolve(path, filename="model.ebin"):
    """Accept either a local `.ebin` or a Hugging Face repo id.

    Without this, `load("Exeaon/Exeaon-Dzo-1.7B")` — the form every model card
    and README shows — fails with a confusing "no such file" instead of doing
    the obvious thing.
    """
    if os.path.exists(path):
        return path
    if "/" not in path or os.path.isabs(path):
        raise FileNotFoundError(path)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=path, filename=filename)


def _runtime_dtype(cfg, device):
    """Run in the precision the checkpoint was trained in, not always fp16.

    fp16 and bf16 hold the same number of bits but spend them differently:
    bf16 keeps fp32's exponent range, fp16 stops at 65504. A model trained in
    bf16 can therefore carry activations that simply do not exist in fp16, and
    casting it down does not degrade them, it overflows them.

    This was not theoretical. Qwen3-32B, loaded as fp16, produced Inf across an
    entire 5120-wide hidden vector at `model.layers.2.mlp.down_proj` on a
    prompt of ordinary length. Those infinities became NaN in the softmax, and
    a NaN score makes argmax return index 0 for every question -- so the model
    answered "the first option" to everything and scored 24.33% on a
    four-choice benchmark and 49.33% on a two-choice one. Both are chance, and
    chance looks like a quantization failure rather than a dtype one. Short
    prompts never overflowed, so a smoke test reported the model healthy.

    Falls back to fp32 rather than fp16 where bf16 is unavailable: fp32 costs
    memory, which this format exists to save, but fp16 costs correctness.
    """
    if device == "cpu":
        return torch.float32

    # `torch_dtype` first, and not via `or`. Checking `dtype` first with an
    # `or` silently lost the answer: transformers populates a default `dtype`
    # on the config, so the truthy default short-circuited the branch and
    # `torch_dtype="bfloat16"` was never read. The fix looked applied, the
    # model still loaded as fp16, and the overflow was unchanged.
    want = getattr(cfg, "torch_dtype", None)
    if want is None:
        want = getattr(cfg, "dtype", None)
    if isinstance(want, str):
        want = getattr(torch, want, None)

    if want is torch.bfloat16:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            chosen = torch.bfloat16
        else:
            # fp32, not fp16: fp32 costs memory, fp16 costs correctness.
            chosen = torch.float32
    else:
        chosen = torch.float16

    # Announced rather than silent. A wrong dtype does not raise -- it
    # overflows deep in the stack and resurfaces as a benchmark score that
    # looks like ordinary quantization damage.
    print(f"  epure: config declares {want}, running in {chosen}", flush=True)
    return chosen


def load(path, device=None):
    """Load a `.ebin` as a model, from a local path or a HF repo id.

    Returns (model, tokenizer).
    """
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    path = resolve(path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tmp = path + ".meta"
    os.makedirs(tmp, exist_ok=True)
    for name, data in container.extras(path).items():
        with open(os.path.join(tmp, name), "wb") as f:
            f.write(data)

    cfg = AutoConfig.from_pretrained(tmp)
    cfg.use_cache = True
    tok = AutoTokenizer.from_pretrained(tmp)

    # Parameters go to meta since every one is overwritten; buffers must stay
    # real, because values computed at init (rotary tables) are not stored in
    # the file and fail at the first forward if left on meta.
    # `config.architectures[0]` names the exact class the model was saved as -
    # "Qwen3ForCausalLM", "WhisperForConditionalGeneration",
    # "T5ForConditionalGeneration". Resolving it directly means a family we have
    # not seen loads without a change here, where hardcoding
    # AutoModelForCausalLM silently excluded every encoder-decoder and every
    # audio model.
    arch = (getattr(cfg, "architectures", None) or [None])[0]
    factory = getattr(transformers, arch, None) if arch else None
    if factory is None:
        factory = AutoModelForCausalLM

    from accelerate import init_empty_weights
    with init_empty_weights(include_buffers=False):
        model = (factory.from_config(cfg) if factory is AutoModelForCausalLM
                 else factory(cfg))
    chosen_dtype = _runtime_dtype(cfg, device)
    model = model.to(dtype=chosen_dtype)

    apply_to(model, path, device=device, cfg=cfg, dtype=chosen_dtype)
    return model.eval(), tok
