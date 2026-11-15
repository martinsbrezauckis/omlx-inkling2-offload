# SPDX-License-Identifier: MIT
"""Expert offload for Inkling 2 routed MoE experts.

Each wrapped ``Inkling2Layer`` keeps its ``Inkling2Experts`` module in place
and replaces its ``w13`` / ``w2`` projection tensors (``weight``, ``scales``,
and ``biases`` for affine) with ``[capacity, ...]`` slot tensors, dropping the
full ``[E, ...]`` tables before lazy weights materialize. NVFP4's tiny
``global_scale`` table (``[E]`` float32) stays resident in full while the
projection holds a ``[capacity]`` slot copy updated on install.

An eager :class:`OffloadedInkling2Experts` callable is installed onto
``layer._split_experts`` so the layer runs its compiled ``_post_attn`` stage,
this host-coordinated expert stage, and its compiled ``_post_tail`` stage.
In-capacity calls remap expert ids to slot ids and run ``routes()`` and
``combine()`` directly (bit-identical to the S1 split path). Over-capacity
prefills run expert-major chunks (resident experts first, each expert installed
at most once per call, each chunk evaluated before the next overwrites slots,
and the next chunk read ahead on the shared IO pool).

Both checkpoint layouts are supported per layer:
  (a) runtime / oQ4e stacked tensors (``model.layers.N.mlp.experts.*``);
  (b) Thinking Machines source tensors (``model.llm.layers.N.mlp.experts.*``),
      reading ``U8`` / ``F8_E4M3`` weights as raw bytes viewed as ``uint32``
      and deriving NVFP4 ``global_scale`` from ``amax`` and ``denominator``.
"""

from __future__ import annotations

import json
import logging
import os
import struct
from concurrent.futures import wait
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ...scheduler import _sync_and_clear_cache
from ..moe_expert_offload import (
    _DTYPES,
    CheckpointExpertStore,
    _io_batch,
    _io_pool,
    _resolve_model_dir,
)
from .model import (
    _NVFP4_DENOMINATOR,
    ExpertProj,
    Inkling2Experts,
    Inkling2Layer,
    Inkling2MoE,
    TextArgs,
    _expert_specs,
    expert_quant_spec,
)

logger = logging.getLogger(__name__)

_PROJS = ("w13", "w2")
_FLOAT_DTYPES = frozenset({"F32", "BF16", "F16"})
_MX_DTYPES = {
    "BF16": mx.bfloat16,
    "F16": mx.float16,
    "F32": mx.float32,
    "U32": mx.uint32,
    "I32": mx.int32,
    "U8": mx.uint8,
    "F8_E4M3": mx.uint8,
}

# Bound large expert reads in addition to the shared pool's expert-count limit.
INFLIGHT_BYTES = 512 * 1024 * 1024

# MLX 0.32.4 GPU gather_qmm(..., sorted_indices=True) with M=1 switches to the
# segmented sorted kernel iff routes >= max(16, 4 * weight.shape[0]) across
# affine, nvfp4, and mxfp8.
_SORTED_QMM_MIN_ROUTES = 16
_SORTED_QMM_ROUTES_PER_EXPERT = 4


def is_inkling2_layer(obj) -> bool:
    """Return True for an ``Inkling2Layer`` carrying ``Inkling2MoE`` experts."""
    return (
        isinstance(obj, Inkling2Layer)
        and isinstance(getattr(obj, "mlp", None), Inkling2MoE)
        and isinstance(getattr(obj.mlp, "experts", None), Inkling2Experts)
    )


def _fields(proj: ExpertProj) -> tuple[str, ...]:
    return ("weight", "scales") + (
        ("biases",) if proj.get("biases") is not None else ()
    )


def _is_expert_proj(proj) -> bool:
    return isinstance(proj, ExpertProj) and all(
        hasattr(proj, a) for a in ("group_size", "bits", "mode")
    )


def _close_store(store: CheckpointExpertStore | None) -> int:
    """Close any open shard file descriptors held by ``store`` (idempotent)."""
    fds = getattr(store, "_fds", None)
    if not fds:
        return 0
    closed = 0
    for fd in list(fds.values()):
        try:
            os.close(fd)
            closed += 1
        except OSError:
            pass
    fds.clear()
    return closed


def _drain(pending: dict) -> None:
    """Cancel or wait for pending reads so no read outlives open descriptors."""
    futures = [f for group in pending.values() for _, _, f in group]
    for future in futures:
        future.cancel()
    if futures:
        wait(futures)


class _Inkling2StoreView:
    """Adapt :class:`CheckpointExpertStore` to Inkling 2's checkpoint layouts."""

    def __init__(
        self,
        store: CheckpointExpertStore,
        prefix: str,
        *,
        source_layout: bool = False,
    ):
        self.store = store
        self.prefix = prefix
        self.source_layout = source_layout

    def _name(self, proj: str, field: str) -> str:
        if not self.source_layout:
            return f"{self.prefix}.{proj}.{field}"
        if field == "weight":
            return f"{self.prefix}.{proj}_weight"
        if field == "scales":
            return f"{self.prefix}.{proj}_weight_scale"
        if field == "amax":
            return f"{self.prefix}.{proj}_weight_amax"
        if field == "denom":
            return f"{self.prefix}.{proj}_weight_outer_scale_denominator"
        raise ValueError(f"Unsupported source field: {field!r}")

    def plan(self, proj: str, field: str, expert: int):
        return self.store.plan_expert(self._name(proj, field), expert)

    def to_slot_array(self, field: str, plan, raw: bytes) -> mx.array:
        arr = CheckpointExpertStore.to_mx(plan, raw)
        if self.source_layout and field == "weight":
            arr = arr.view(mx.uint32)
        return arr

    def fetch_global_scale(self, proj: str) -> mx.array:
        if not self.source_layout:
            return self.store.fetch_tensor(self._name(proj, "global_scale")).astype(
                mx.float32
            )
        amax = self.store.fetch_tensor(self._name(proj, "amax")).astype(mx.float32)
        denom = self.store.fetch_tensor(self._name(proj, "denom")).astype(mx.float32)
        return amax * (_NVFP4_DENOMINATOR / denom)

    def close(self) -> int:
        return _close_store(self.store)


def _runtime_prefixes(path: str, layer_idx: int) -> list[str]:
    candidates = (
        [f"{path}.mlp.experts"] if path else [f"model.layers.{layer_idx}.mlp.experts"]
    )
    if path and not path.startswith("model."):
        candidates.append(f"model.{path}.mlp.experts")
    return candidates


def _has_runtime_tensors(store: CheckpointExpertStore, prefix: str) -> bool:
    return any(
        store.has(f"{prefix}.{proj}.{field}")
        for proj in _PROJS
        for field in ("weight", "scales", "biases", "global_scale")
    )


def _has_source_tensors(store: CheckpointExpertStore, prefix: str) -> bool:
    return any(
        store.has(f"{prefix}.{proj}_{suffix}")
        for proj in _PROJS
        for suffix in (
            "weight",
            "weight_scale",
            "weight_amax",
            "weight_outer_scale_denominator",
        )
    )


def _validate_runtime_view(
    experts: Inkling2Experts,
    view: _Inkling2StoreView,
    n_experts: int,
) -> tuple[_Inkling2StoreView | None, str | None]:
    store = view.store
    for proj_name in _PROJS:
        proj = getattr(experts, proj_name)
        for field in _fields(proj):
            name = view._name(proj_name, field)
            if not store.has(name):
                return None, f"checkpoint has no tensor {name!r}"
            shape, dtype = store.spec(name)
            expected_shape = tuple(proj[field].shape)
            if shape != expected_shape:
                return None, f"{name!r} shape {shape} != expected {expected_shape}"
            if dtype not in _DTYPES:
                return None, f"{name!r} has unsupported dtype {dtype!r}"
            if _MX_DTYPES.get(dtype) != proj[field].dtype:
                return (
                    None,
                    f"{name!r} dtype {dtype!r} != expected {proj[field].dtype}",
                )
        if proj.get("global_scale") is not None:
            gs_name = view._name(proj_name, "global_scale")
            if not store.has(gs_name):
                return None, f"checkpoint has no tensor {gs_name!r}"
            shape, dtype = store.spec(gs_name)
            if shape != (n_experts,):
                return None, f"{gs_name!r} shape {shape} != expected {(n_experts,)}"
            if dtype not in _DTYPES or dtype not in _FLOAT_DTYPES:
                return None, f"{gs_name!r} has unsupported dtype {dtype!r}"
    return view, None


def _validate_source_view(
    experts: Inkling2Experts,
    view: _Inkling2StoreView,
    n_experts: int,
) -> tuple[_Inkling2StoreView | None, str | None]:
    store = view.store
    for proj_name in _PROJS:
        proj = getattr(experts, proj_name)
        if proj.mode not in ("nvfp4", "mxfp8"):
            return (
                None,
                f"{proj_name} mode {proj.mode!r} is unsupported in source layout",
            )
        w_name = view._name(proj_name, "weight")
        s_name = view._name(proj_name, "scales")
        if not store.has(w_name):
            return None, f"checkpoint has no tensor {w_name!r}"
        if not store.has(s_name):
            return None, f"checkpoint has no tensor {s_name!r}"
        w_shape, w_dtype = store.spec(w_name)
        exp_w_shape = (*proj.weight.shape[:-1], proj.weight.shape[-1] * 4)
        if w_shape != exp_w_shape:
            return None, f"{w_name!r} shape {w_shape} != expected {exp_w_shape}"
        exp_w_dtype = "U8" if proj.mode == "nvfp4" else "F8_E4M3"
        if w_dtype not in _DTYPES or w_dtype != exp_w_dtype:
            return (
                None,
                f"{w_name!r} dtype {w_dtype!r} != expected {exp_w_dtype!r}",
            )
        s_shape, s_dtype = store.spec(s_name)
        exp_s_shape = tuple(proj.scales.shape)
        if s_shape != exp_s_shape:
            return None, f"{s_name!r} shape {s_shape} != expected {exp_s_shape}"
        exp_s_dtype = "F8_E4M3" if proj.mode == "nvfp4" else "U8"
        if s_dtype not in _DTYPES or s_dtype != exp_s_dtype:
            return (
                None,
                f"{s_name!r} dtype {s_dtype!r} != expected {exp_s_dtype!r}",
            )
        if proj.mode == "nvfp4":
            amax_name = view._name(proj_name, "amax")
            denom_name = view._name(proj_name, "denom")
            if not store.has(amax_name):
                return None, f"checkpoint has no tensor {amax_name!r}"
            if not store.has(denom_name):
                return None, f"checkpoint has no tensor {denom_name!r}"
            a_shape, a_dtype = store.spec(amax_name)
            if a_shape != (n_experts,):
                return None, f"{amax_name!r} shape {a_shape} != expected {(n_experts,)}"
            if a_dtype not in _DTYPES or a_dtype not in _FLOAT_DTYPES:
                return None, f"{amax_name!r} has unsupported dtype {a_dtype!r}"
            d_shape, d_dtype = store.spec(denom_name)
            if d_shape not in ((), (1,), (n_experts,)):
                return (
                    None,
                    f"{denom_name!r} shape {d_shape} incompatible with {(n_experts,)}",
                )
            if d_dtype not in _DTYPES or d_dtype not in _FLOAT_DTYPES:
                return None, f"{denom_name!r} has unsupported dtype {d_dtype!r}"
        elif store.has(view._name(proj_name, "amax")):
            return (
                None,
                f"{proj_name} has unexpected NVFP4 amax tensor for mxfp8 layer",
            )
    return view, None


def resolve_view(
    layer: Inkling2Layer, store: CheckpointExpertStore, path: str
) -> tuple[_Inkling2StoreView | None, str | None]:
    """Validate checkpoint coverage for ``layer``; return ``(view, None)`` or ``(None, reason)``."""
    if not is_inkling2_layer(layer):
        return None, "not an Inkling2Layer with Inkling2Experts"
    experts = layer.mlp.experts
    for proj_name in _PROJS:
        proj = getattr(experts, proj_name, None)
        if proj is None or not _is_expert_proj(proj):
            return None, f"{proj_name} is not an ExpertProj"
        if "bias" in proj:
            return None, f"{proj_name} has per-expert bias (unsupported)"
        if proj.mode not in ("nvfp4", "mxfp8", "affine"):
            return None, f"{proj_name} has unsupported mode {proj.mode!r}"
        if proj.weight.ndim != 3 or proj.scales.ndim != 3:
            return None, f"{proj_name} tensors are not 3-D stacked"
        if proj.mode == "affine" and proj.get("biases") is None:
            return None, f"{proj_name} affine mode is missing biases"
    n_experts = experts.w13.weight.shape[0]
    if experts.w2.weight.shape[0] != n_experts:
        return None, "w13 and w2 disagree on expert count"
    for proj_name in _PROJS:
        proj = getattr(experts, proj_name)
        if proj.mode == "nvfp4":
            gs = proj.get("global_scale")
            if gs is None or tuple(gs.shape) != (n_experts,):
                return None, f"{proj_name} NVFP4 global_scale missing or wrong shape"

    runtime_candidates = _runtime_prefixes(path, layer.layer_idx)
    for prefix in runtime_candidates:
        if _has_runtime_tensors(store, prefix):
            return _validate_runtime_view(
                experts,
                _Inkling2StoreView(store, prefix, source_layout=False),
                n_experts,
            )

    source_prefix = f"model.llm.layers.{layer.layer_idx}.mlp.experts"
    if _has_source_tensors(store, source_prefix):
        return _validate_source_view(
            experts,
            _Inkling2StoreView(store, source_prefix, source_layout=True),
            n_experts,
        )

    return None, f"checkpoint has no expert tensors for {runtime_candidates[0]!r}"


class _SlotCache:
    """LRU slots living inside ``Inkling2Experts``'s own projection tensors."""

    moe_offload_cache = True

    def __init__(
        self,
        experts: Inkling2Experts,
        capacity: int,
        view: _Inkling2StoreView,
    ):
        self.experts = experts
        self.view = view
        self.n_experts = experts.w13.weight.shape[0]
        self.capacity = min(capacity, self.n_experts)
        self.resident: dict[str, list] = {}
        self.global_scales: dict[str, mx.array] = {}
        for proj_name in _PROJS:
            proj = getattr(experts, proj_name)
            arrays = []
            for field in _fields(proj):
                slots = mx.zeros(
                    (self.capacity, *proj[field].shape[1:]),
                    dtype=proj[field].dtype,
                )
                setattr(proj, field, slots)  # drops the lazy full-size array
                arrays.append(slots)
            if "global_scale" in proj:
                # NVFP4 tensor scales are tiny ([E]): keep full table, slot a copy.
                full_gs = view.fetch_global_scale(proj_name)
                self.global_scales[proj_name] = full_gs
                proj.global_scale = mx.ones((self.capacity,), dtype=mx.float32)
                arrays += [full_gs, proj["global_scale"]]
            self.resident[proj_name] = arrays
        self.slot_of: dict[int, int] = {}  # expert id -> slot, LRU ordered
        self.slot_expert = np.full(self.capacity, -1, dtype=np.int64)
        self.free = list(range(self.capacity))
        self.map = mx.full((self.n_experts,), -1, dtype=mx.int32)
        self.hits = self.misses = 0
        self.fetched_bytes = 0
        self.warm = False
        self._writes = [
            (proj_name, field)
            for proj_name in _PROJS
            for field in _fields(getattr(experts, proj_name))
        ]
        self.expert_bytes = sum(
            view.plan(proj_name, field, 0).nbytes for proj_name, field in self._writes
        )

    def io_window(self) -> int:
        if _io_pool() is None:
            return 0
        return max(1, min(_io_batch(), INFLIGHT_BYTES // max(1, self.expert_bytes)))

    def _plans(self, e: int) -> list:
        return [
            (write, self.view.plan(write[0], write[1], e)) for write in self._writes
        ]

    def _submit(self, pool, e: int) -> list:
        return [
            (write, plan, pool.submit(CheckpointExpertStore.read, plan))
            for write, plan in self._plans(e)
        ]

    def read_ahead(self, ids, limit: int) -> dict:
        """Start reading up to ``limit`` non-resident experts without mutating cache."""
        pool = _io_pool()
        out: dict[int, list] = {}
        if pool is None or limit <= 0:
            return out
        for e in dict.fromkeys(int(e) for e in ids):
            if len(out) >= limit:
                break
            if e not in self.slot_of:
                out[e] = self._submit(pool, e)
        return out

    def ensure(self, idx: mx.array) -> None:
        if self.warm:
            return
        self.ensure_ids(idx.reshape(-1).tolist())

    def ensure_ids(self, ids, pending: dict | None = None) -> None:
        """Load missing experts through the shared reader and protect current hits."""
        pending = dict(pending or {})
        needed = list(dict.fromkeys(int(e) for e in ids))
        if len(needed) > self.capacity:
            _drain(pending)
            raise ValueError("Expert cache capacity is smaller than the call's routes")
        misses = []
        for e in needed:
            if e in self.slot_of:
                self.slot_of[e] = self.slot_of.pop(e)  # re-insert: LRU order
                self.hits += 1
            else:
                misses.append(e)
        if not misses:
            _drain(pending)
            return
        protected = set(needed)
        pool = _io_pool()
        window = self.io_window()
        submitted = 0

        def submit(limit: int) -> None:
            nonlocal submitted
            if pool is None:
                return
            while submitted < min(limit, len(misses)):
                e = misses[submitted]
                submitted += 1
                if e not in pending:
                    pending[e] = self._submit(pool, e)

        try:
            submit(window)
            for done, e in enumerate(misses):
                # Refill before this expert's writes so at most ``window``
                # experts' bytes exist at once, counting the one written here.
                submit(done + window)
                group = pending.pop(e, None)
                if group is not None:
                    raws = [(write, plan, f.result()) for write, plan, f in group]
                else:
                    raws = [
                        (write, plan, CheckpointExpertStore.read(plan))
                        for write, plan in self._plans(e)
                    ]
                if self.free:
                    slot = self.free.pop()
                else:
                    victim = next(v for v in self.slot_of if v not in protected)
                    slot = self.slot_of.pop(victim)
                    self.slot_expert[slot] = -1
                    self.map[victim] = -1
                try:
                    for (proj_name, field), plan, raw in raws:
                        proj = getattr(self.experts, proj_name)
                        proj[field][slot] = self.view.to_slot_array(field, plan, raw)
                    for proj_name, scale in self.global_scales.items():
                        getattr(self.experts, proj_name)["global_scale"][slot] = scale[
                            e
                        ]
                except BaseException:
                    self.free.append(slot)
                    raise
                self.slot_of[e] = slot
                self.slot_expert[slot] = e
                self.map[e] = slot
                self.misses += 1
                self.fetched_bytes += sum(plan.nbytes for _, plan, _ in raws)
                del raws
        finally:
            _drain(pending)
        self.warm = len(self.slot_of) == self.n_experts

    def close(self) -> int:
        """Close the store's open shard descriptors (shared across wrapped layers)."""
        return self.view.close()


class OffloadedInkling2Experts(nn.Module):
    """Inkling 2 routed-expert callable backed by a :class:`_SlotCache`."""

    def __init__(self, cache: _SlotCache):
        super().__init__()
        self.cache = cache

    def _forward_expert_major(
        self,
        flat_x: mx.array,
        inds: mx.array,
        weights: mx.array,
        ids: list[int],
        k: int,
        out_dtype,
    ) -> mx.array:
        """Routes sorted by expert (resident first), cut into ``capacity`` chunks."""
        c = self.cache
        d_model = flat_x.shape[-1]
        ids_np = np.asarray(ids, dtype=np.int64)
        resident = np.zeros(c.n_experts, dtype=np.bool_)
        resident[c.slot_expert[c.slot_expert >= 0]] = True
        # Resident experts first: a miss evicts only experts already used.
        rank = ids_np + (~resident[ids_np]) * c.n_experts
        order = np.argsort(rank, kind="stable")
        sorted_ids = ids_np[order]
        run_starts = np.flatnonzero(np.diff(sorted_ids)) + 1
        run_starts = np.concatenate(([0], run_starts))
        cuts = run_starts[:: c.capacity].tolist() + [len(ids)]
        chunks = list(zip(cuts[:-1], cuts[1:]))
        outs = []
        ahead: dict = {}
        try:
            for n, (start, end) in enumerate(chunks):
                chunk_ids = sorted_ids[start:end]
                c.ensure_ids(np.unique(chunk_ids).tolist(), pending=ahead)
                ahead = {}
                slots = mx.take(c.map, mx.array(chunk_ids, dtype=mx.int32)).astype(
                    inds.dtype
                )
                t_idx = mx.array(order[start:end] // k, dtype=mx.int32)
                xe = mx.take(flat_x, t_idx, axis=0)
                o = c.experts.routes(xe, slots.reshape(-1, 1))[:, 0, :]
                if n + 1 < len(chunks):
                    s1, e1 = chunks[n + 1]
                    ahead = c.read_ahead(
                        np.unique(sorted_ids[s1:e1]).tolist(), c.io_window()
                    )
                mx.eval(o)
                outs.append(o)
        finally:
            _drain(ahead)
        out = mx.concatenate(outs, axis=0)
        inverse = mx.array(np.argsort(order, kind="stable"), dtype=mx.int32)
        z = mx.take(out, inverse, axis=0).reshape((*inds.shape, d_model))
        s2_table = c.global_scales.get("w2")
        if s2_table is None:
            return c.experts.combine(z, inds, weights, out_dtype)
        s2 = s2_table[inds] * (1.0 / _NVFP4_DENOMINATOR)
        return mx.sum(z.astype(mx.float32) * (weights * s2)[..., None], axis=-2).astype(
            out_dtype
        )

    def __call__(self, x: mx.array, inds: mx.array, weights: mx.array) -> mx.array:
        c = self.cache
        flat_i = inds.reshape(-1, inds.shape[-1])
        n_tok, k = flat_i.shape
        if k > c.capacity:
            raise ValueError("Expert cache capacity is smaller than routing top-k")
        fits = c.warm or n_tok * k <= c.capacity or n_tok == 1
        ids = None
        if not fits:
            ids = flat_i.reshape(-1).tolist()
            fits = len(set(ids)) <= c.capacity
        if fits:
            if not c.warm:
                if ids is None:
                    c.ensure(inds)
                else:
                    c.ensure_ids(ids)
            slots = mx.take(c.map, inds).astype(inds.dtype)
            # Match stock's effective gather_qmm kernel choice from n_experts rather
            # than capacity: when capacity < n_experts and B < 4 * n_experts, passing
            # sorted_indices=True on [capacity, ...] can cross MLX's switch point
            # early and alter rounding.
            sorted_hint = (
                None
                if c.capacity == c.n_experts
                else inds.size
                >= max(
                    _SORTED_QMM_MIN_ROUTES,
                    _SORTED_QMM_ROUTES_PER_EXPERT * c.n_experts,
                )
            )
            z = c.experts.routes(x, slots, sorted_hint=sorted_hint)
            return c.experts.combine(z, slots, weights, x.dtype)
        flat_x = x.reshape(-1, x.shape[-1])
        return self._forward_expert_major(flat_x, inds, weights, ids, k, x.dtype)


def _iter_inkling2_layers(model):
    seen = set()

    def walk(obj, path):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if is_inkling2_layer(obj):
            yield (obj, path)
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield from walk(v, f"{path}.{k}" if path else k)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                yield from walk(v, f"{path}.{i}" if path else str(i))

    yield from walk(model, "")


def close_inkling2_moe_expert_offload(model) -> int:
    """Close open checkpoint file descriptors across all offloaded Inkling 2 layers."""
    closed = 0
    seen_stores = set()
    for layer, _ in _iter_inkling2_layers(model):
        for cache in (
            getattr(layer.mlp, "cache", None),
            getattr(layer._split_experts, "cache", None),
        ):
            view = getattr(cache, "view", None)
            store = getattr(view, "store", None)
            if store is not None and id(store) not in seen_stores:
                seen_stores.add(id(store))
                closed += _close_store(store)
    return closed


def apply_inkling2_moe_expert_offload(
    model,
    model_path: str | Path,
    resident_fraction: float = 0.25,
) -> int:
    """Wrap covered Inkling 2 layers with expert offloading; return wrapped count.

    Must run before lazy weights materialize. Returns 0 for non-Inkling 2
    models or when no layers can be covered; skips (with a logged reason) any
    layer whose checkpoint tensors are missing or mismatched.
    """
    targets = list(_iter_inkling2_layers(model))
    if not targets:
        return 0
    model_dir = _resolve_model_dir(model_path)
    if model_dir is None:
        return 0
    store = CheckpointExpertStore(model_dir)
    if not store:
        logger.warning(
            "inkling2 moe expert offload: no safetensors under %s", model_dir
        )
        return 0
    wrapped = 0
    total_bytes = resident_bytes = 0
    for layer, path in targets:
        if (
            isinstance(layer._split_experts, OffloadedInkling2Experts)
            or getattr(layer.mlp, "cache", None) is not None
        ):
            continue
        view, reason = resolve_view(layer, store, path)
        if view is None:
            logger.info("inkling2 moe expert offload: skipping %s (%s)", path, reason)
            continue
        experts = layer.mlp.experts
        n_experts = experts.w13.weight.shape[0]
        top_k = int(layer.mlp.top_k)
        capacity = min(n_experts, max(top_k, round(n_experts * resident_fraction)))
        layer_bytes = sum(
            int(np.prod(proj[field].shape)) * proj[field].dtype.size
            for proj_name in _PROJS
            for proj in (getattr(experts, proj_name),)
            for field in _fields(proj)
        )
        total_bytes += layer_bytes
        resident_bytes += layer_bytes * capacity // n_experts
        cache = _SlotCache(experts, capacity, view)
        offloaded = OffloadedInkling2Experts(cache)
        # Expose cache on layer.mlp (in nn.Module's dict walk) for
        # moe_offload_caches / moe_offload_stats / materialize_offload_state,
        # while avoiding any cycle back from experts -> cache.
        layer.mlp.cache = cache
        layer._split_experts = offloaded
        wrapped += 1
        _sync_and_clear_cache()

    if wrapped == 0:
        _close_store(store)
    else:
        logger.info(
            "inkling2 moe expert offload: wrapped %d layers at %.1f%% residency "
            "(expert tables: %.2f GB total, %.2f GB resident)",
            wrapped,
            100 * resident_fraction,
            total_bytes / 1e9,
            resident_bytes / 1e9,
        )
    return wrapped


class _SafetensorsHeaderStore:
    """Read-only safetensors header view (shape, dtype, nbytes) without open fds."""

    def __init__(self, model_dir: Path):
        self._specs: dict[str, tuple[tuple[int, ...], str, int]] = {}
        for shard in sorted(model_dir.glob("*.safetensors")):
            with open(shard, "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_len))
            for name, spec in header.items():
                if name == "__metadata__":
                    continue
                b0, b1 = spec["data_offsets"]
                self._specs[name] = (
                    tuple(spec.get("shape", ())),
                    spec.get("dtype", ""),
                    b1 - b0,
                )

    def __bool__(self) -> bool:
        return bool(self._specs)

    def has(self, name: str) -> bool:
        return name in self._specs

    def spec(self, name: str) -> tuple[tuple[int, ...], str]:
        shape, dtype, _ = self._specs[name]
        return shape, dtype

    def nbytes(self, name: str) -> int:
        return self._specs[name][2]


class _ArraySpec:
    __slots__ = ("shape", "dtype")

    def __init__(self, shape: tuple[int, ...], dtype):
        self.shape = shape
        self.dtype = dtype


class _ProjSpec(dict):
    """Expected tensor shape/dtype metadata matching ``ExpertProj`` without arrays."""

    def __init__(self, experts: int, out_dims: int, in_dims: int, spec: dict):
        super().__init__()
        self.mode = spec.get("mode")
        self.bits = int(spec.get("bits", 0))
        self.group_size = int(spec.get("group_size", 0))
        per = 32 // self.bits if self.bits > 0 else 0
        if (
            self.mode not in ("nvfp4", "mxfp8", "affine")
            or self.bits <= 0
            or self.group_size <= 0
            or per <= 0
            or in_dims % per != 0
            or in_dims % self.group_size != 0
        ):
            return
        groups = in_dims // self.group_size
        self["weight"] = _ArraySpec((experts, out_dims, in_dims // per), mx.uint32)
        if self.mode == "affine":
            self["scales"] = _ArraySpec((experts, out_dims, groups), mx.bfloat16)
            self["biases"] = _ArraySpec((experts, out_dims, groups), mx.bfloat16)
        else:
            self["scales"] = _ArraySpec((experts, out_dims, groups), mx.uint8)
        if self.mode == "nvfp4":
            self["global_scale"] = _ArraySpec((experts,), mx.float32)

    @property
    def weight(self):
        return self.get("weight")

    @property
    def scales(self):
        return self.get("scales")


class _ExpertsSpec:
    __slots__ = ("w13", "w2")

    def __init__(
        self, n_experts: int, hidden: int, inter: int, w13_spec: dict, w2_spec: dict
    ):
        self.w13 = _ProjSpec(n_experts, 2 * inter, hidden, w13_spec)
        self.w2 = _ProjSpec(n_experts, hidden, inter, w2_spec)

    @property
    def valid(self) -> bool:
        return "weight" in self.w13 and "weight" in self.w2


def estimate_inkling2_expert_savings(
    model_path: str | Path,
    resident_fraction: float = 0.25,
) -> int:
    """Return resident expert bytes removed for an Inkling 2 checkpoint.

    Header-only (never loads or materializes tensors) and reuses the adapter's
    runtime/source view validation rules so it never promises savings for layers
    :func:`apply_inkling2_moe_expert_offload` would leave resident.
    ``global_scale`` and source NVFP4 scaling scalars remain resident and are
    not discounted.
    """
    try:
        model_dir = _resolve_model_dir(model_path)
        if model_dir is None:
            return 0
        config_path = model_dir / "config.json"
        if not config_path.exists():
            return 0
        config = json.loads(config_path.read_text())
        if config.get("model_type") != "inkling2":
            return 0
        raw_text = config.get("text_config")
        args = TextArgs.from_dict(raw_text if isinstance(raw_text, dict) else {})
        num_layers = int(args.num_hidden_layers)
        n_experts = int(args.n_routed_experts)
        top_k = int(args.num_experts_per_tok)
        hidden = int(args.hidden_size)
        inter = int(args.intermediate_size)
        if min(num_layers, n_experts, hidden, inter) <= 0 or top_k < 0:
            return 0
        capacity = min(n_experts, max(top_k, round(n_experts * resident_fraction)))
        if capacity >= n_experts:
            return 0
        store = _SafetensorsHeaderStore(model_dir)
        if not store:
            return 0
        quantization = config.get("quantization")
        saved = 0
        for idx in range(num_layers):
            rt_prefix = f"model.layers.{idx}.mlp.experts"
            src_prefix = f"model.llm.layers.{idx}.mlp.experts"
            if _has_runtime_tensors(store, rt_prefix):
                if any(store.has(f"{rt_prefix}.{p}.bias") for p in _PROJS):
                    continue
                spec_w13, spec_w2 = _expert_specs(args, quantization, idx)
                experts = _ExpertsSpec(n_experts, hidden, inter, spec_w13, spec_w2)
                if not experts.valid:
                    continue
                view, _ = _validate_runtime_view(
                    experts,
                    _Inkling2StoreView(store, rt_prefix, source_layout=False),
                    n_experts,
                )
            elif _has_source_tensors(store, src_prefix):
                fmt = "nvfp4" if store.has(f"{src_prefix}.w13_weight_amax") else "mxfp8"
                spec = expert_quant_spec(fmt)
                experts = _ExpertsSpec(n_experts, hidden, inter, spec, spec)
                if not experts.valid:
                    continue
                view, _ = _validate_source_view(
                    experts,
                    _Inkling2StoreView(store, src_prefix, source_layout=True),
                    n_experts,
                )
            else:
                continue
            if view is None:
                continue
            layer_bytes = sum(
                store.nbytes(view._name(proj_name, field))
                for proj_name in _PROJS
                for field in _fields(getattr(experts, proj_name))
            )
            saved += (n_experts - capacity) * (layer_bytes // n_experts)
        return saved
    except Exception:
        logger.debug("inkling2 offload admission estimate failed", exc_info=True)
        return 0


__all__ = [
    "OffloadedInkling2Experts",
    "apply_inkling2_moe_expert_offload",
    "close_inkling2_moe_expert_offload",
    "estimate_inkling2_expert_savings",
    "is_inkling2_layer",
    "resolve_view",
]
