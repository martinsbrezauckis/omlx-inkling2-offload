# SPDX-License-Identifier: MIT
"""Tests for Inkling 2 MoE expert offloading (omlx/patches/inkling2/moe_offload.py)."""

from __future__ import annotations

import contextlib
import gc
import json
import logging
import os
import struct
import threading
import types
from collections import deque

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten
from mlx_lm.utils import load_model

from omlx.patches.inkling2 import loader as ink_loader
from omlx.patches.inkling2 import model as inkling2
from omlx.patches.inkling2.moe_offload import (
    _SORTED_QMM_MIN_ROUTES,
    _SORTED_QMM_ROUTES_PER_EXPERT,
    OffloadedInkling2Experts,
    apply_inkling2_moe_expert_offload,
    close_inkling2_moe_expert_offload,
)
from omlx.patches.moe_expert_offload import (
    _shutdown_io_pool,
    materialize_offload_state,
    moe_offload_caches,
    moe_offload_stats,
)
from omlx.utils.model_loading import materialize_lazy_state
from tests.test_inkling_vlm_mtp import _inkling2_tiny


def _format_spec(fmt: str):
    if fmt == "affine":
        return {"mode": "affine", "group_size": 64, "bits": 4}
    return fmt


def _build_tiny_model(
    formats,
    *,
    num_experts: int = 8,
    top_k: int = 2,
    seed: int = 99,
    cast_bf16: bool = True,
) -> inkling2.Model:
    model = _inkling2_tiny(
        formats=tuple(formats),
        n_routed_experts=num_experts,
        num_experts_per_tok=top_k,
    )
    mx.random.seed(seed)
    gs_updates = [
        (name, mx.random.uniform(0.5, 8.0, value.shape))
        for name, value in tree_flatten(model.parameters())
        if name.endswith("global_scale")
    ]
    if gs_updates:
        model.load_weights(gs_updates, strict=False)
    if cast_bf16:
        pred = model.cast_predicate
        model.load_weights(
            [
                (k, v.astype(mx.bfloat16) if pred(k) and v.dtype == mx.float32 else v)
                for k, v in tree_flatten(model.parameters())
            ]
        )
    mx.eval(model.parameters())
    return model


def _model_config_dict(model: inkling2.Model) -> dict:
    a = model.args.text
    return {
        "model_type": "inkling2",
        "eos_token_id": model.args.eos_token_id,
        "image_token_id": model.args.image_token_id,
        "text_config": {
            "hidden_size": a.hidden_size,
            "num_hidden_layers": a.num_hidden_layers,
            "vocab_size": a.vocab_size,
            "unpadded_vocab_size": a.unpadded_vocab_size,
            "num_attention_heads": a.num_attention_heads,
            "num_key_value_heads": a.num_key_value_heads,
            "head_dim": a.head_dim,
            "swa_num_attention_heads": a.swa_num_attention_heads,
            "swa_num_key_value_heads": a.swa_num_key_value_heads,
            "swa_head_dim": a.swa_head_dim,
            "sliding_window_size": a.sliding_window_size,
            "rel_extent": a.rel_extent,
            "d_rel": a.d_rel,
            "local_layer_ids": a.local_layer_ids,
            "n_layers_per_attn_res_block": a.n_layers_per_attn_res_block,
            "n_routed_experts": a.n_routed_experts,
            "num_experts_per_tok": a.num_experts_per_tok,
            "intermediate_size": a.intermediate_size,
            "expert_formats": a.expert_formats,
        },
    }


def _save_runtime_checkpoint(
    directory, model: inkling2.Model, *, extra_drops: tuple[str, ...] = ()
):
    directory.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.parameters()))
    for k in extra_drops:
        weights.pop(k, None)
    mx.save_safetensors(str(directory / "model.safetensors"), weights)
    (directory / "config.json").write_text(json.dumps(_model_config_dict(model)))
    return directory


def _split_cuts(tensor: mx.array, cuts: list[int]) -> list[mx.array]:
    parts = []
    start = 0
    for end in cuts:
        parts.append(tensor[start:end])
        start = end
    parts.append(tensor[start:])
    return parts


def _to_thinking_machines_source(
    model: inkling2.Model,
    *,
    denom_value: float = 1344.0,
) -> tuple[dict[str, mx.array], set[str]]:
    """Invert ``loader.sanitize_weights`` for a tiny Inkling 2 model.

    Returns ``(tensors, f8_e4m3_keys)`` where ``f8_e4m3_keys`` lists keys whose
    safetensors header dtype must be relabeled from ``U8`` to ``F8_E4M3``.
    """
    args = model.args.text
    weights = dict(tree_flatten(model.parameters()))
    out: dict[str, mx.array] = {
        "model.llm.embed.weight": weights["model.embed.weight"],
        "model.llm.embed_norm.weight": weights["model.embed_norm.weight"],
        "model.llm.norm.weight": weights["model.norm.weight"],
        "model.llm.unembed.weight": weights["lm_head.weight"],
        "model.llm.attn_res.output_query": weights["model.final_input.query"],
    }
    f8_keys: set[str] = set()
    denom = mx.array(denom_value, dtype=mx.float32)

    for idx, layer in enumerate(model.model.layers):
        pre_in = f"model.layers.{idx}."
        pre_out = f"model.llm.layers.{idx}."
        if idx > 0:
            out[f"model.llm.attn_res.attn_queries.{idx - 1}"] = weights[
                f"{pre_in}attn_input.query"
            ]
        out[f"model.llm.attn_res.mlp_queries.{idx}"] = weights[
            f"{pre_in}mlp_input.query"
        ]
        attn = layer.attn
        if args.is_sliding(idx):
            wq, wk, wv, wr, wg, ws = _split_cuts(
                weights[f"{pre_in}attn.in_proj.weight"], attn._cuts
            )
            out[f"{pre_out}attn.wq_du.weight"] = wq
            out[f"{pre_out}attn.wk_dv.weight"] = wk
            out[f"{pre_out}attn.wv_dv.weight"] = wv
            out[f"{pre_out}attn.wr_du.weight"] = wr
            out[f"{pre_out}attn.gate_attn.weight"] = wg
            out[f"{pre_out}attn.attn_sink_proj.weight"] = ws
            out[f"{pre_out}attn.k_sconv.weight"] = weights[f"{pre_in}attn.k_conv"]
            out[f"{pre_out}attn.v_sconv.weight"] = weights[f"{pre_in}attn.v_conv"]
            out[f"{pre_out}attn_norm.weight"] = weights[f"{pre_in}attn_norm.weight"]
        else:
            wq, wr, wg, ws = _split_cuts(
                weights[f"{pre_in}attn.q_proj.weight"], attn._cuts
            )
            out[f"{pre_out}attn.wq_du.weight"] = wq
            out[f"{pre_out}attn.wr_du.weight"] = wr
            out[f"{pre_out}attn.gate_attn.weight"] = wg
            out[f"{pre_out}attn.attn_sink_proj.weight"] = ws
            wk, wk_proj = _split_cuts(
                weights[f"{pre_in}attn.k_proj.weight"], [attn.dkv]
            )
            wv, wv_proj = _split_cuts(
                weights[f"{pre_in}attn.v_proj.weight"], [attn.dkv]
            )
            out[f"{pre_out}attn.wk_dv.weight"] = wk
            out[f"{pre_out}attn.k_sconv.weight_proj"] = wk_proj
            out[f"{pre_out}attn.wv_dv.weight"] = wv
            out[f"{pre_out}attn.v_sconv.weight_proj"] = wv_proj
            out[f"{pre_out}attn.k_sconv.static_w"] = weights[f"{pre_in}attn.k_static"]
            out[f"{pre_out}attn.v_sconv.static_w"] = weights[f"{pre_in}attn.v_static"]
            for n in ("q", "k", "v"):
                out[f"{pre_out}{n}_input_norm.weight"] = weights[
                    f"{pre_in}{n}_input_norm.weight"
                ]

        out[f"{pre_out}attn.wo_ud.weight"] = weights[f"{pre_in}attn.o_proj.weight"]
        out[f"{pre_out}attn.q_l2_clip.scale"] = weights[f"{pre_in}attn.q_scale"]
        out[f"{pre_out}attn.rel_l2_clip.scale"] = weights[f"{pre_in}attn.r_scale"]
        out[f"{pre_out}attn.rel_logits_proj.proj"] = weights[f"{pre_in}attn.rel_proj"]
        out[f"{pre_out}attn_sconv.weight"] = weights[f"{pre_in}attn_conv"]
        out[f"{pre_out}mlp_sconv.weight"] = weights[f"{pre_in}mlp_conv"]
        out[f"{pre_out}mlp_norm.weight"] = weights[f"{pre_in}mlp_norm.weight"]
        out[f"{pre_out}mlp.gate.weight"] = weights[f"{pre_in}mlp.router_weight"]
        out[f"{pre_out}mlp.gate.bias"] = weights[f"{pre_in}mlp.router_bias"]

        for which in ("w13", "w2"):
            proj = getattr(layer.mlp.experts, which)
            w_key = f"{pre_out}mlp.experts.{which}_weight"
            s_key = f"{pre_out}mlp.experts.{which}_weight_scale"
            out[w_key] = weights[f"{pre_in}mlp.experts.{which}.weight"].view(mx.uint8)
            out[s_key] = weights[f"{pre_in}mlp.experts.{which}.scales"]
            if proj.mode == "nvfp4":
                f8_keys.add(s_key)
                gs = weights[f"{pre_in}mlp.experts.{which}.global_scale"]
                out[f"{pre_out}mlp.experts.{which}_weight_amax"] = gs.astype(
                    mx.float32
                ) * (denom / inkling2._NVFP4_DENOMINATOR)
                out[f"{pre_out}mlp.experts.{which}_weight_outer_scale_denominator"] = (
                    denom
                )
            elif proj.mode == "mxfp8":
                f8_keys.add(w_key)
            else:
                raise ValueError(f"Source layout does not support mode {proj.mode!r}")

    mx.eval(*out.values())
    return out, f8_keys


def _save_source_checkpoint(
    directory,
    model: inkling2.Model,
    *,
    denom_value: float = 1344.0,
    extra_drops: tuple[str, ...] = (),
):
    directory.mkdir(parents=True, exist_ok=True)
    tensors, f8_keys = _to_thinking_machines_source(model, denom_value=denom_value)
    for k in extra_drops:
        tensors.pop(k, None)
        f8_keys.discard(k)
    file = directory / "model.safetensors"
    mx.save_safetensors(str(file), tensors)
    if f8_keys:
        # MLX cannot write F8_E4M3 directly; relabel the raw uint8 codes.
        raw = file.read_bytes()
        header_len = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8 : 8 + header_len])
        for key in f8_keys:
            if key in header:
                header[key]["dtype"] = "F8_E4M3"
        packed = json.dumps(header).encode("utf-8")
        packed += b" " * (-len(packed) % 8)
        file.write_bytes(
            struct.pack("<Q", len(packed)) + packed + raw[8 + header_len :]
        )
    cfg = _model_config_dict(model)
    cfg["text_config"].pop("expert_formats", None)
    (directory / "config.json").write_text(json.dumps(cfg))
    return directory


def _reload_inkling2(directory, *, lazy: bool = True) -> inkling2.Model:
    model, _ = load_model(
        directory,
        lazy=lazy,
        get_model_classes=lambda config: (inkling2.Model, inkling2.ModelArgs),
    )
    return model


def _enable_split_without_offload(model: inkling2.Model) -> None:
    for layer in model.model.layers:
        layer._split_experts = True


def _max_abs_diff(a: mx.array, b: mx.array) -> float:
    return float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item())


def test_source_layout_helper_roundtrips_through_sanitize_weights(tmp_path):
    """Source-layout inverter round-trips bit-identically via loader.sanitize_weights."""
    formats = ("nvfp4", "nvfp4", "mxfp8", "mxfp8", "nvfp4", "mxfp8")
    model = _build_tiny_model(formats)
    _save_source_checkpoint(tmp_path, model, denom_value=1344.0)

    raw = mx.load(str(tmp_path / "model.safetensors"))
    assert ink_loader.is_original_layout(raw)

    fresh = inkling2.Model(inkling2.ModelArgs.from_dict(_model_config_dict(model)))
    mapped, got_formats = ink_loader.sanitize_weights(fresh, dict(raw))
    assert [got_formats[i] for i in range(len(formats))] == list(formats)

    orig_weights = dict(tree_flatten(model.parameters()))
    assert set(mapped.keys()) == set(orig_weights.keys())
    for key, orig_val in orig_weights.items():
        got_val = mapped[key]
        mx.eval(orig_val, got_val)
        assert got_val.dtype == orig_val.dtype, key
        assert got_val.shape == orig_val.shape, key
        assert bool(mx.array_equal(got_val, orig_val)), key


@pytest.mark.parametrize("fmt", ["mxfp8", "nvfp4", "affine"])
@pytest.mark.parametrize("residency", [1.0, 0.5, 0.25])
def test_runtime_formats_and_residencies_decode_batch_verify(tmp_path, fmt, residency):
    """Decode, batch=2 and verify T=4..16 match S1 split bit-for-bit."""
    formats = (_format_spec(fmt),) * 6
    stock_fused = _build_tiny_model(formats, num_experts=8, top_k=2)
    _save_runtime_checkpoint(tmp_path, stock_fused)

    stock_split = _reload_inkling2(tmp_path, lazy=False)
    _enable_split_without_offload(stock_split)

    offloaded = _reload_inkling2(tmp_path, lazy=True)
    wrapped = apply_inkling2_moe_expert_offload(
        offloaded, tmp_path, resident_fraction=residency
    )
    assert wrapped == 6
    mx.eval(offloaded.parameters())
    assert materialize_offload_state(offloaded) == 6

    expected_cap = min(8, max(2, round(8 * residency)))
    caches = moe_offload_caches(offloaded)
    assert len(caches) == 6
    assert all(c.capacity == expected_cap for c in caches)

    try:
        # 1. Decode (T=1, several steps) after prompt warmup
        mx.random.seed(11)
        seq = mx.random.randint(0, 300, (1, 16))
        c_fused = stock_fused.make_cache()
        c_split = stock_split.make_cache()
        c_off = offloaded.make_cache()
        mx.eval(
            stock_fused(seq[:, :10], cache=c_fused),
            stock_split(seq[:, :10], cache=c_split),
            offloaded(seq[:, :10], cache=c_off),
        )
        max_fused_decode_delta = 0.0
        for t in range(10, 16):
            lf = stock_fused(seq[:, t : t + 1], cache=c_fused)[..., :300]
            ls = stock_split(seq[:, t : t + 1], cache=c_split)[..., :300]
            lo = offloaded(seq[:, t : t + 1], cache=c_off)[..., :300]
            mx.eval(lf, ls, lo)
            assert bool(mx.array_equal(lo, ls))
            max_fused_decode_delta = max(max_fused_decode_delta, _max_abs_diff(lo, lf))
        assert (
            max_fused_decode_delta <= 5e-4
        ), f"decode ({fmt}, r={residency}) max |delta|={max_fused_decode_delta}"

        # 2. Batch 2 decode (T=1, several steps)
        mx.random.seed(22)
        b_seq = mx.random.randint(0, 300, (2, 10))
        cb_fused = stock_fused.make_cache()
        cb_split = stock_split.make_cache()
        cb_off = offloaded.make_cache()
        mx.eval(
            stock_fused(b_seq[:, :6], cache=cb_fused),
            stock_split(b_seq[:, :6], cache=cb_split),
            offloaded(b_seq[:, :6], cache=cb_off),
        )
        max_fused_batch_delta = 0.0
        for t in range(6, 10):
            lf = stock_fused(b_seq[:, t : t + 1], cache=cb_fused)[..., :300]
            ls = stock_split(b_seq[:, t : t + 1], cache=cb_split)[..., :300]
            lo = offloaded(b_seq[:, t : t + 1], cache=cb_off)[..., :300]
            mx.eval(lf, ls, lo)
            assert bool(mx.array_equal(lo, ls))
            max_fused_batch_delta = max(max_fused_batch_delta, _max_abs_diff(lo, lf))
        assert (
            max_fused_batch_delta <= 5e-4
        ), f"batch-2 ({fmt}, r={residency}) max |delta|={max_fused_batch_delta}"

        # 3. Verify T=4..16 (at residency >= 0.5 or 1.0, and check T=4 at 0.25 if fits)
        max_fused_verify_delta = 0.0
        for v_len in (4, 8, 16):
            cv_fused = stock_fused.make_cache()
            cv_split = stock_split.make_cache()
            cv_off = offloaded.make_cache()
            mx.random.seed(30 + v_len)
            v_ids = mx.random.randint(0, 300, (1, 10 + v_len))
            mx.eval(
                stock_fused(v_ids[:, :10], cache=cv_fused),
                stock_split(v_ids[:, :10], cache=cv_split),
                offloaded(v_ids[:, :10], cache=cv_off),
            )
            lf = stock_fused(v_ids[:, 10:], cache=cv_fused)[..., :300]
            ls = stock_split(v_ids[:, 10:], cache=cv_split)[..., :300]
            lo = offloaded(v_ids[:, 10:], cache=cv_off)[..., :300]
            mx.eval(lf, ls, lo)
            if residency == 1.0 or v_len * 2 <= expected_cap:
                assert bool(mx.array_equal(lo, ls))
            else:
                assert _max_abs_diff(lo, ls) < 2e-2
            max_fused_verify_delta = max(max_fused_verify_delta, _max_abs_diff(lo, lf))
        assert max_fused_verify_delta < 2e-2

        stats = moe_offload_stats(offloaded)
        assert stats["layers"] == 6
        assert stats["fetched_bytes"] == sum(c.misses * c.expert_bytes for c in caches)
        print(
            f"[delta fmt={fmt} r={residency}] "
            f"decode={max_fused_decode_delta:.3e} "
            f"batch2={max_fused_batch_delta:.3e} "
            f"verify={max_fused_verify_delta:.3e}"
        )
    finally:
        close_inkling2_moe_expert_offload(offloaded)


@pytest.mark.parametrize("fmt", ["mxfp8", "nvfp4", "affine"])
@pytest.mark.parametrize("residency", [1.0, 0.5, 0.25])
def test_in_capacity_sorted_prefill_bit_exact(tmp_path, fmt, residency):
    """Sorted prefill (T*k >= 64) whose distinct experts fit capacity is bit-identical."""
    # E=16, top_k=2 -> at r=0.25 capacity=4. Bias routing onto experts 0..3 so
    # T=40 (80 routes >= 64 sorted threshold) stays within 4 experts.
    formats = (_format_spec(fmt),) * 6
    stock = _build_tiny_model(formats, num_experts=16, top_k=2)
    for layer in stock.model.layers:
        bias = mx.full((16,), -100.0, dtype=mx.float32)
        bias[:4] = mx.array([4.0, 3.0, 2.0, 1.0], dtype=mx.float32)
        layer.mlp.router_bias = bias
    mx.eval(stock.parameters())
    _save_runtime_checkpoint(tmp_path, stock)

    stock_split = _reload_inkling2(tmp_path, lazy=False)
    _enable_split_without_offload(stock_split)

    offloaded = _reload_inkling2(tmp_path, lazy=True)
    assert apply_inkling2_moe_expert_offload(offloaded, tmp_path, residency) == 6
    mx.eval(offloaded.parameters())
    assert materialize_offload_state(offloaded) == 6

    try:
        mx.random.seed(55)
        ids = mx.random.randint(0, 300, (1, 40))
        ref_fused = stock(ids)[..., :300]
        ref_split = stock_split(ids)[..., :300]
        got = offloaded(ids)[..., :300]
        mx.eval(ref_fused, ref_split, got)

        assert bool(mx.array_equal(got, ref_split))
        assert bool(mx.array_equal(got, ref_fused))
        for cache in moe_offload_caches(offloaded):
            assert cache.misses <= cache.capacity
            assert cache.fetched_bytes == cache.misses * cache.expert_bytes
    finally:
        close_inkling2_moe_expert_offload(offloaded)


@pytest.mark.parametrize("fmt", ["mxfp8", "nvfp4", "affine"])
def test_over_capacity_prefill_bounded_and_single_fetch(tmp_path, fmt):
    """Over-capacity prefill bounds error < 2e-2 and installs each expert at most once."""
    formats = (_format_spec(fmt),) * 6
    stock = _build_tiny_model(formats, num_experts=16, top_k=2)
    # _inkling2_tiny initializes affine weights with unsigned nibbles + N(0,0.05)
    # scales (blowing SwiGLU outputs up to |z|~71 where 1 bf16 ULP is 0.5) and
    # router_bias ~ router_weight so tokens collapse onto 2-4 experts. Use
    # genuinely quantized N(0,0.05) affine weights and zero-bias diverse routing
    # so every layer genuinely exercises > capacity=4 distinct experts.
    mx.random.seed(42)
    for layer in stock.model.layers:
        if fmt == "affine":
            for proj_name in ("w13", "w2"):
                proj = getattr(layer.mlp.experts, proj_name)
                in_d = proj.scales.shape[-1] * proj.group_size
                out_d = proj.weight.shape[1]
                wf = (mx.random.normal((16, out_d, in_d)) * 0.05).astype(mx.bfloat16)
                w, s, b = mx.quantize(wf, group_size=64, bits=4, mode="affine")
                proj.weight, proj.scales, proj.biases = w, s, b
        layer.mlp.router_bias = mx.zeros_like(layer.mlp.router_bias)
        layer.mlp.router_weight = (
            mx.random.normal(layer.mlp.router_weight.shape).astype(mx.float32) * 0.5
        )
    mx.eval(stock.parameters())
    _save_runtime_checkpoint(tmp_path, stock)

    offloaded = _reload_inkling2(tmp_path, lazy=True)
    assert apply_inkling2_moe_expert_offload(offloaded, tmp_path, 0.25) == 6
    mx.eval(offloaded.parameters())
    assert materialize_offload_state(offloaded) == 6

    caches = [layer.mlp.cache for layer in offloaded.model.layers]
    assert all(c.capacity == 4 for c in caches)

    try:
        # Warm layer 0 with 2 resident experts first so we also exercise
        # "resident experts first" ordering and verify resident hits aren't re-fetched.
        warm_x = mx.random.normal((1, 128)).astype(mx.bfloat16)
        warm_i = mx.array([[0, 5]], dtype=mx.uint32)
        warm_w = mx.array([[0.5, 0.5]], dtype=mx.float32)
        mx.eval(offloaded.model.layers[0]._split_experts(warm_x, warm_i, warm_w))
        assert caches[0].misses == 2

        # Spy on view.plan per layer to count how many times each expert is fetched.
        fetched_by_layer: list[list[int]] = [[] for _ in caches]
        for idx, cache in enumerate(caches):
            orig_plan = cache.view.plan

            def make_spy(fn, log):
                def spy(proj, field, e):
                    if proj == "w13" and field == "weight":
                        log.append(int(e))
                    return fn(proj, field, e)

                return spy

            cache.view.plan = make_spy(orig_plan, fetched_by_layer[idx])

        # Capture the exact routes seen by layer 0 and full-model output vs stock.
        mx.random.seed(77)
        ids = mx.random.randint(0, 300, (1, 48))  # 96 routes across ~16 experts > 4
        ref = stock(ids)[..., :300]
        got = offloaded(ids)[..., :300]
        mx.eval(ref, got)
        delta = _max_abs_diff(got, ref)
        assert delta < 2e-2, f"over-capacity prefill ({fmt}) delta={delta}"
        print(f"[over-capacity prefill fmt={fmt}] max |delta logit|={delta:.3e}")

        for idx, cache in enumerate(caches):
            assert cache.misses > cache.capacity
            assert len(fetched_by_layer[idx]) == len(
                set(fetched_by_layer[idx])
            ), f"layer {idx} fetched duplicate expert in one call: {fetched_by_layer[idx]}"
            assert cache.fetched_bytes == cache.misses * cache.expert_bytes
            assert len(cache.slot_of) <= cache.capacity

        # Verify on layer 0 directly with known routes spanning all 16 experts
        # and pre-resident experts: resident experts hit, misses fetched once each.
        c0 = caches[0]
        pre_resident = set(c0.slot_of.keys())
        fetched_by_layer[0].clear()
        hits_before = c0.hits
        misses_before = c0.misses
        x_big = (mx.random.normal((32, 128)) * 0.5).astype(mx.bfloat16)
        inds_big = mx.stack(
            [
                mx.array([(2 * t) % 16, (2 * t + 1) % 16], dtype=mx.uint32)
                for t in range(32)
            ]
        )
        w_big = mx.random.uniform(0.1, 1.0, (32, 2))
        ref_e = stock.model.layers[0].mlp.experts(x_big, inds_big, w_big)
        got_e = offloaded.model.layers[0]._split_experts(x_big, inds_big, w_big)
        mx.eval(ref_e, got_e)
        assert _max_abs_diff(got_e, ref_e) < 2e-2
        assert sorted(fetched_by_layer[0]) == sorted(set(range(16)) - pre_resident)
        assert c0.hits - hits_before == len(pre_resident)
        assert c0.misses - misses_before == 16 - len(pre_resident)
        assert c0.fetched_bytes == c0.misses * c0.expert_bytes
    finally:
        close_inkling2_moe_expert_offload(offloaded)


def _find_owned_full_expert_arrays(
    model,
    forbidden_ids: set[int],
    num_experts: int,
) -> list[tuple[tuple[int, ...], mx.Dtype]]:
    """Inspect only the nn.Module tree, module attributes, and offload caches.

    Never traverses types, functions, modules, or globals, so unrelated tests
    loaded in the same pytest process cannot pollute the check.
    """
    skip_types = (
        type,
        types.ModuleType,
        types.FunctionType,
        types.MethodType,
        types.CodeType,
        types.FrameType,
        types.BuiltinFunctionType,
        types.BuiltinMethodType,
        types.GetSetDescriptorType,
        types.MemberDescriptorType,
        int,
        float,
        bool,
        str,
        bytes,
        bytearray,
        memoryview,
        range,
        slice,
        type(None),
    )

    seen: set[int] = set()
    stack: list[object] = [model]
    stack.extend(arr for _, arr in tree_flatten(model.parameters()))
    stack.extend(moe_offload_caches(model))
    found: list[tuple[tuple[int, ...], mx.Dtype]] = []
    while stack:
        obj = stack.pop()
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        if isinstance(obj, mx.array):
            if oid in forbidden_ids or (
                obj.ndim >= 2
                and obj.shape[0] == num_experts
                and obj.dtype in (mx.uint32, mx.uint8)
            ):
                found.append((obj.shape, obj.dtype))
            continue
        if isinstance(obj, skip_types):
            continue
        if isinstance(obj, dict):
            stack.extend(obj.keys())
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset, deque)):
            stack.extend(obj)
        try:
            inst_dict = object.__getattribute__(obj, "__dict__")
        except AttributeError:
            inst_dict = None
        if isinstance(inst_dict, dict):
            stack.extend(inst_dict.values())
        for base in type(obj).__mro__:
            slots = base.__dict__.get("__slots__")
            if not slots:
                continue
            for slot in (slots,) if isinstance(slots, str) else slots:
                if slot in ("__dict__", "__weakref__"):
                    continue
                with contextlib.suppress(AttributeError):
                    stack.append(object.__getattribute__(obj, slot))
    return found


def test_no_full_expert_arrays_retained_after_wrapping(tmp_path):
    """After wrapping, no [E, ...] w13/w2 arrays remain owned by the model or caches."""
    formats = ("mxfp8", "nvfp4", {"mode": "affine", "group_size": 64, "bits": 4}) * 2
    stock = _build_tiny_model(formats, num_experts=8, top_k=2)
    _save_runtime_checkpoint(tmp_path, stock)
    stock_param_names = [k for k, _ in tree_flatten(stock.parameters())]

    offloaded = _reload_inkling2(tmp_path, lazy=True)
    assert apply_inkling2_moe_expert_offload(offloaded, tmp_path, 0.25) == 6
    mx.eval(offloaded.parameters())
    assert materialize_offload_state(offloaded) == 6

    try:
        assert [k for k, _ in tree_flatten(offloaded.parameters())] == stock_param_names
        for layer in offloaded.model.layers:
            assert isinstance(layer._split_experts, OffloadedInkling2Experts)
            assert not layer._split_experts.parameters()
            exp = layer.mlp.experts
            for proj_name in ("w13", "w2"):
                proj = getattr(exp, proj_name)
                assert proj.weight.shape[0] == 2
                assert proj.scales.shape[0] == 2
                if proj.get("biases") is not None:
                    assert proj.biases.shape[0] == 2
                if proj.get("global_scale") is not None:
                    assert proj.global_scale.shape == (2,)
                    assert layer.mlp.cache.global_scales[proj_name].shape == (8,)

        forbidden_ids = {
            id(arr)
            for layer in stock.model.layers
            for proj in (layer.mlp.experts.w13, layer.mlp.experts.w2)
            for f in ("weight", "scales", "biases")
            if (arr := proj.get(f)) is not None
        }
        found_full_arrays = _find_owned_full_expert_arrays(offloaded, forbidden_ids, 8)
        assert (
            not found_full_arrays
        ), f"Retained full expert arrays: {found_full_arrays}"

        # Negative control: retaining a full [E, ...] expert array on the model
        # tree or offload cache must be caught by the checker.
        leaked_w13 = stock.model.layers[0].mlp.experts.w13.weight
        leaked_w2_s = stock.model.layers[1].mlp.experts.w2.scales
        offloaded.model.layers[0].mlp._leaked_full = leaked_w13
        try:
            caught_mod = _find_owned_full_expert_arrays(offloaded, forbidden_ids, 8)
            assert caught_mod == [(tuple(leaked_w13.shape), leaked_w13.dtype)]
        finally:
            del offloaded.model.layers[0].mlp._leaked_full

        offloaded.model.layers[1].mlp.cache._leaked_cache = leaked_w2_s
        try:
            caught_cache = _find_owned_full_expert_arrays(offloaded, forbidden_ids, 8)
            assert caught_cache == [(tuple(leaked_w2_s.shape), leaked_w2_s.dtype)]
        finally:
            del offloaded.model.layers[1].mlp.cache._leaked_cache

        assert not _find_owned_full_expert_arrays(offloaded, forbidden_ids, 8)
    finally:
        close_inkling2_moe_expert_offload(offloaded)


@pytest.mark.parametrize("residency", [1.0, 0.5, 0.25])
def test_source_layout_matches_runtime_layout_and_stock(tmp_path, residency):
    """Thinking Machines source layout matches runtime layout and stock."""
    formats = ("nvfp4", "nvfp4", "mxfp8", "mxfp8", "nvfp4", "mxfp8")
    stock = _build_tiny_model(formats, num_experts=8, top_k=2)
    stock_split = _build_tiny_model(formats, num_experts=8, top_k=2)
    _enable_split_without_offload(stock_split)

    rt_dir = _save_runtime_checkpoint(tmp_path / "runtime", stock)
    src_dir = _save_source_checkpoint(tmp_path / "source", stock, denom_value=1344.0)

    rt_off = _reload_inkling2(rt_dir, lazy=True)
    src_off = _reload_inkling2(src_dir, lazy=True)
    assert apply_inkling2_moe_expert_offload(rt_off, rt_dir, residency) == 6
    assert apply_inkling2_moe_expert_offload(src_off, src_dir, residency) == 6
    mx.eval(rt_off.parameters(), src_off.parameters())
    assert materialize_offload_state(rt_off) == 6
    assert materialize_offload_state(src_off) == 6

    try:
        mx.random.seed(88)
        ids = mx.random.randint(0, 300, (1, 18))
        c_split = stock_split.make_cache()
        c_rt = rt_off.make_cache()
        c_src = src_off.make_cache()

        # Warmup prefix + decode steps
        mx.eval(
            stock_split(ids[:, :10], cache=c_split),
            rt_off(ids[:, :10], cache=c_rt),
            src_off(ids[:, :10], cache=c_src),
        )
        for t in range(10, 14):
            ls = stock_split(ids[:, t : t + 1], cache=c_split)[..., :300]
            lr = rt_off(ids[:, t : t + 1], cache=c_rt)[..., :300]
            lx = src_off(ids[:, t : t + 1], cache=c_src)[..., :300]
            mx.eval(ls, lr, lx)
            assert bool(mx.array_equal(lx, lr))
            assert bool(mx.array_equal(lx, ls))

        # Batch-2 decode
        b_ids = mx.random.randint(0, 300, (2, 6))
        cb_split = stock_split.make_cache()
        cb_rt = rt_off.make_cache()
        cb_src = src_off.make_cache()
        mx.eval(
            stock_split(b_ids[:, :4], cache=cb_split),
            rt_off(b_ids[:, :4], cache=cb_rt),
            src_off(b_ids[:, :4], cache=cb_src),
        )
        for t in range(4, 6):
            ls = stock_split(b_ids[:, t : t + 1], cache=cb_split)[..., :300]
            lr = rt_off(b_ids[:, t : t + 1], cache=cb_rt)[..., :300]
            lx = src_off(b_ids[:, t : t + 1], cache=cb_src)[..., :300]
            mx.eval(ls, lr, lx)
            assert bool(mx.array_equal(lx, lr))
            assert bool(mx.array_equal(lx, ls))

        # Over-capacity / verify forward
        p_ids = mx.random.randint(0, 300, (1, 40))
        lf = stock(p_ids)[..., :300]
        lr = rt_off(p_ids)[..., :300]
        lx = src_off(p_ids)[..., :300]
        mx.eval(lf, lr, lx)
        assert bool(mx.array_equal(lx, lr))
        assert _max_abs_diff(lx, lf) < 2e-2
    finally:
        close_inkling2_moe_expert_offload(rt_off)
        close_inkling2_moe_expert_offload(src_off)


def test_unwrappable_layer_stays_resident_and_matches_stock(tmp_path, caplog):
    """A layer with missing/mismatched tensors stays resident while others offload."""
    formats = (
        "nvfp4",
        "mxfp8",
        {"mode": "affine", "group_size": 64, "bits": 4},
        "nvfp4",
    )
    stock = _build_tiny_model(formats, num_experts=8, top_k=2)
    _save_runtime_checkpoint(
        tmp_path / "rt",
        stock,
        extra_drops=("model.layers.1.mlp.experts.w13.scales",),
    )

    stock_split = _build_tiny_model(formats, num_experts=8, top_k=2)
    _enable_split_without_offload(stock_split)

    candidate = _build_tiny_model(formats, num_experts=8, top_k=2)
    with caplog.at_level(logging.INFO, logger="omlx.patches.inkling2.moe_offload"):
        wrapped = apply_inkling2_moe_expert_offload(candidate, tmp_path / "rt", 0.25)
    assert wrapped == 3
    assert "model.layers.1" in caplog.text

    try:
        # Layer 1 remained resident and untouched.
        l1 = candidate.model.layers[1]
        assert l1._split_experts is None
        assert getattr(l1.mlp, "cache", None) is None
        assert l1.mlp.experts.w13.weight.shape[0] == 8
        for idx in (0, 2, 3):
            assert isinstance(
                candidate.model.layers[idx]._split_experts, OffloadedInkling2Experts
            )
            assert candidate.model.layers[idx].mlp.experts.w13.weight.shape[0] == 2

        mx.random.seed(61)
        ids = mx.random.randint(0, 300, (1, 14))
        cs = stock_split.make_cache()
        cc = candidate.make_cache()
        mx.eval(stock_split(ids[:, :8], cache=cs), candidate(ids[:, :8], cache=cc))
        for t in range(8, 12):
            ls = stock_split(ids[:, t : t + 1], cache=cs)[..., :300]
            lc = candidate(ids[:, t : t + 1], cache=cc)[..., :300]
            mx.eval(ls, lc)
            assert bool(mx.array_equal(lc, ls))
    finally:
        close_inkling2_moe_expert_offload(candidate)

    # Source layout missing NVFP4 outer_scale_denominator also skips cleanly.
    src_formats = ("nvfp4", "mxfp8")
    src_stock = _build_tiny_model(src_formats, num_experts=8, top_k=2)
    _save_source_checkpoint(
        tmp_path / "src",
        src_stock,
        extra_drops=(
            "model.llm.layers.0.mlp.experts.w2_weight_outer_scale_denominator",
        ),
    )
    src_cand = _build_tiny_model(src_formats, num_experts=8, top_k=2)
    try:
        assert apply_inkling2_moe_expert_offload(src_cand, tmp_path / "src", 0.25) == 1
        assert src_cand.model.layers[0]._split_experts is None
        assert isinstance(
            src_cand.model.layers[1]._split_experts, OffloadedInkling2Experts
        )
    finally:
        close_inkling2_moe_expert_offload(src_cand)


def test_non_inkling2_model_returns_zero(tmp_path):
    dummy = nn.Module()
    dummy.linear = nn.Linear(16, 16)
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), {"linear.weight": dummy.linear.weight}
    )
    assert apply_inkling2_moe_expert_offload(dummy, tmp_path, 0.25) == 0


@pytest.mark.parametrize("workers", [1, 4])
def test_file_descriptors_released_on_close_and_gc(tmp_path, monkeypatch, workers):
    """Both explicit close and dropping the model release all shard descriptors."""
    monkeypatch.setattr("omlx.patches.moe_expert_offload._IO_WORKERS", workers)
    _shutdown_io_pool()

    formats = ("nvfp4", "mxfp8")
    stock = _build_tiny_model(formats, num_experts=8, top_k=2)
    _save_source_checkpoint(tmp_path, stock)

    def open_fds():
        return len(os.listdir("/dev/fd"))

    def run_once():
        m = _reload_inkling2(tmp_path, lazy=True)
        assert apply_inkling2_moe_expert_offload(m, tmp_path, 0.25) == 2
        mx.eval(m.parameters())
        materialize_offload_state(m)
        mx.eval(m(mx.array([[1, 2]])))
        return m

    try:
        # Settle one-time imports/pools before recording baseline.
        m0 = run_once()
        close_inkling2_moe_expert_offload(m0)
        del m0
        gc.collect()
        baseline = open_fds()

        # 1. Explicit close releases fds immediately while model stays referenced.
        m1 = run_once()
        assert open_fds() > baseline
        closed = close_inkling2_moe_expert_offload(m1)
        assert closed >= 1
        assert close_inkling2_moe_expert_offload(m1) == 0
        assert open_fds() == baseline
        del m1
        gc.collect()

        # 2. Implicit drop frees fds even with cyclic GC disabled (no cycles).
        gc.disable()
        try:
            for _ in range(3):
                m2 = run_once()
                del m2
            assert open_fds() == baseline
        finally:
            gc.enable()
    finally:
        _shutdown_io_pool()


def test_materialize_offload_state_cross_thread(tmp_path):
    """Materialized slot state runs cleanly on a separate inference thread."""
    formats = ("nvfp4", "mxfp8", {"mode": "affine", "group_size": 64, "bits": 4})
    stock = _build_tiny_model(formats, num_experts=8, top_k=2)
    _save_runtime_checkpoint(tmp_path, stock)
    stock_split = _reload_inkling2(tmp_path, lazy=False)
    _enable_split_without_offload(stock_split)

    holder: dict[str, inkling2.Model] = {}

    def load_on_worker():
        m = _reload_inkling2(tmp_path, lazy=True)
        assert apply_inkling2_moe_expert_offload(m, tmp_path, 0.25) == 3
        assert materialize_offload_state(m) == 3
        materialize_lazy_state(m)
        holder["model"] = m

    t = threading.Thread(target=load_on_worker)
    t.start()
    t.join()

    offloaded = holder["model"]
    try:
        ids = mx.array([[3, 7, 11]])
        ref = stock_split(ids)[..., :300]
        got = offloaded(ids)[..., :300]
        mx.eval(ref, got)
        assert bool(mx.array_equal(got, ref))
    finally:
        close_inkling2_moe_expert_offload(offloaded)


def test_lru_eviction_and_counters(tmp_path):
    """LRU order, hit/miss counters, and fetched_bytes behave consistently."""
    formats = ("nvfp4", "mxfp8")
    stock = _build_tiny_model(formats, num_experts=8, top_k=2)
    _save_runtime_checkpoint(tmp_path, stock)
    offloaded = _reload_inkling2(tmp_path, lazy=True)
    assert apply_inkling2_moe_expert_offload(offloaded, tmp_path, 0.25) == 2
    mx.eval(offloaded.parameters())
    materialize_offload_state(offloaded)

    try:
        exp0 = offloaded.model.layers[0]._split_experts
        c0 = exp0.cache
        assert c0.capacity == 2
        x = (mx.random.normal((1, 128)) * 0.5).astype(mx.bfloat16)
        w = mx.array([[0.6, 0.4]], dtype=mx.float32)

        # Cold miss: installs 0 and 1.
        mx.eval(exp0(x, mx.array([[0, 1]], dtype=mx.uint32), w))
        assert (c0.hits, c0.misses) == (0, 2)
        # Touch 0 (hit) -> LRU order becomes [1, 0], then miss on 2 evicts 1.
        mx.eval(exp0(x, mx.array([[0, 2]], dtype=mx.uint32), w))
        assert (c0.hits, c0.misses) == (1, 3)
        assert set(c0.slot_of.keys()) == {0, 2}
        assert c0.fetched_bytes == 3 * c0.expert_bytes

        # Re-fetching evicted 1 matches stock bit-for-bit.
        i_check = mx.array([[1, 2]], dtype=mx.uint32)
        ref = stock.model.layers[0].mlp.experts(x, i_check, w)
        got = exp0(x, i_check, w)
        mx.eval(ref, got)
        assert bool(mx.array_equal(got, ref))
    finally:
        close_inkling2_moe_expert_offload(offloaded)


def test_lazy_wrapping_reads_only_requested_experts(tmp_path):
    """Wrapping a lazily loaded model drops full slabs before eval and reads only needed experts."""
    formats = ("mxfp8", "nvfp4", {"mode": "affine", "group_size": 64, "bits": 4})
    stock = _build_tiny_model(formats, num_experts=16, top_k=2)
    _save_runtime_checkpoint(tmp_path, stock)

    offloaded = _reload_inkling2(tmp_path, lazy=True)
    assert apply_inkling2_moe_expert_offload(offloaded, tmp_path, 0.25) == 3
    mx.eval(offloaded.parameters())
    assert materialize_offload_state(offloaded) == 3

    try:
        caches = [layer.mlp.cache for layer in offloaded.model.layers]
        for layer, cache in zip(offloaded.model.layers, caches):
            assert cache.capacity == 4
            assert cache.misses == 0
            assert cache.fetched_bytes == 0
            assert layer.mlp.experts.w13.weight.shape[0] == 4
            assert layer.mlp.experts.w2.weight.shape[0] == 4

        ids = mx.array([[3]])
        ref = stock(ids)[..., :300]
        got = offloaded(ids)[..., :300]
        mx.eval(ref, got)
        assert _max_abs_diff(got, ref) == 0.0
        for cache in caches:
            assert 1 <= cache.misses <= 2
            assert cache.fetched_bytes == cache.misses * cache.expert_bytes
    finally:
        close_inkling2_moe_expert_offload(offloaded)


def _probe_gather_qmm_sorted_switch(
    mode: str,
    experts: int,
    routes: int,
    *,
    group_size: int | None = None,
    bits: int | None = None,
    out_dims: int = 128,
    in_dims: int = 128,
    dtype=mx.float32,
    seed: int = 0,
) -> bool:
    """Return True when ``gather_qmm(..., sorted_indices=True)`` differs from ``False``."""
    spec = (
        {"mode": "affine", "group_size": group_size or 64, "bits": bits or 4}
        if mode == "affine"
        else inkling2.expert_quant_spec(mode)
    )
    gs, b = spec["group_size"], spec["bits"]
    mx.random.seed(seed)
    wf = (mx.random.normal((experts, out_dims, in_dims)) * 0.25).astype(dtype)
    quantized = mx.quantize(wf, group_size=gs, bits=b, mode=mode)
    w, s = quantized[0], quantized[1]
    biases = quantized[2] if len(quantized) == 3 else None
    x = (mx.random.normal((routes, 1, in_dims)) * 0.5).astype(dtype)
    idx = mx.sort(mx.random.randint(0, experts, (routes,)).astype(mx.uint32))
    out_t = mx.gather_qmm(
        x,
        w,
        s,
        biases,
        rhs_indices=idx,
        transpose=True,
        group_size=gs,
        bits=b,
        mode=mode,
        sorted_indices=True,
    )
    out_f = mx.gather_qmm(
        x,
        w,
        s,
        biases,
        rhs_indices=idx,
        transpose=True,
        group_size=gs,
        bits=b,
        mode=mode,
        sorted_indices=False,
    )
    mx.eval(out_t, out_f)
    return not bool(mx.array_equal(out_t, out_f))


@pytest.mark.parametrize("mode", ["affine", "nvfp4", "mxfp8"])
def test_mlx_gather_qmm_sorted_switch_pin(mode):
    """Pin MLX's GPU gather_qmm sorted_indices switch rule: B >= max(16, 4 * E)."""
    if mx.default_device() != mx.gpu:
        pytest.skip("GPU Metal gather_qmm kernel dispatch")

    for experts in (2, 4, 5, 8, 16, 32, 64):
        threshold = max(_SORTED_QMM_MIN_ROUTES, _SORTED_QMM_ROUTES_PER_EXPERT * experts)
        probe_points = [threshold - 1, threshold, threshold + 1]
        if experts == 32:
            probe_points.append(64)
        for routes in probe_points:
            expected = routes >= threshold
            switched = _probe_gather_qmm_sorted_switch(
                mode,
                experts,
                routes,
                dtype=mx.float32,
                seed=experts * 100 + routes,
            )
            assert switched is expected, (
                f"MLX {mx.__version__} gather_qmm switch rule shifted for "
                f"mode={mode}, E={experts}, B={routes}: switched={switched}, "
                f"expected={expected} (threshold max(16, 4*E)={threshold}). "
                "Update _SORTED_QMM_* in omlx/patches/inkling2/moe_offload.py."
            )
        if mode == "affine":
            for routes in probe_points:
                expected = routes >= threshold
                switched = _probe_gather_qmm_sorted_switch(
                    mode,
                    experts,
                    routes,
                    dtype=mx.bfloat16,
                    seed=experts * 100 + routes,
                )
                assert switched is expected, (
                    f"MLX {mx.__version__} bf16 affine gather_qmm switch rule "
                    f"shifted at E={experts}, B={routes}: switched={switched}, "
                    f"expected={expected} (threshold={threshold})."
                )


def test_affine_in_capacity_64_routes_matches_stock_exact(tmp_path, monkeypatch):
    """Repro: 12-layer nvfp4+affine bf16 model, E=32, k=4, cap=16, 64-route verify."""
    formats = ("nvfp4",) * 8 + (_format_spec("affine"),) * 4
    stock = _build_tiny_model(formats, num_experts=32, top_k=4, seed=11, cast_bf16=True)
    _save_runtime_checkpoint(tmp_path, stock)

    offloaded = _reload_inkling2(tmp_path, lazy=True)
    assert apply_inkling2_moe_expert_offload(offloaded, tmp_path, 0.5) == 12
    mx.eval(offloaded.parameters())
    assert materialize_offload_state(offloaded) == 12
    assert all(c.capacity == 16 for c in moe_offload_caches(offloaded))

    em_calls = [0]
    orig_em = OffloadedInkling2Experts._forward_expert_major

    def spy_em(self, *args, **kwargs):
        em_calls[0] += 1
        return orig_em(self, *args, **kwargs)

    monkeypatch.setattr(OffloadedInkling2Experts, "_forward_expert_major", spy_em)

    try:
        mx.random.seed(5)
        prompt = mx.random.randint(0, 300, (2, 48))

        def run_seq(model):
            cache = model.make_cache()
            logits_list = []
            lg = model(prompt, cache=cache)[:, -1, :300]
            logits_list.append(lg)
            tok = mx.argmax(lg, axis=-1)[:, None]
            toks = [tok]
            for _ in range(12):
                lg = model(tok, cache=cache)[:, -1, :300]
                logits_list.append(lg)
                tok = mx.argmax(lg, axis=-1)[:, None]
                toks.append(tok)
            verify_chunk = mx.concatenate(toks[-8:], axis=1)  # batch 2 x 8 = 64 routes
            verify_logits = model(verify_chunk, cache=cache)[..., :300]
            mx.eval(logits_list, toks, verify_logits)
            return logits_list, mx.concatenate(toks, axis=1), verify_logits

        ref_logits, ref_toks, ref_verify = run_seq(stock)
        got_logits, got_toks, got_verify = run_seq(offloaded)
        assert em_calls[0] == 0
        for g, r in zip(got_logits, ref_logits):
            assert bool(mx.array_equal(g, r))
        assert bool(mx.array_equal(got_toks, ref_toks))
        assert bool(mx.array_equal(got_verify, ref_verify))

        # Isolated expert call on affine layer 8 with non-identity slot layout.
        aff_idx = 8
        exp_stock = stock.model.layers[aff_idx].mlp.experts
        exp_off = offloaded.model.layers[aff_idx]._split_experts
        c8 = exp_off.cache
        # Pre-warm 4 experts outside the active set so active experts land in higher slots.
        warm_x = (mx.random.normal((1, 128)) * 0.5).astype(mx.bfloat16)
        mx.eval(
            exp_off(
                warm_x,
                mx.array([[20, 21, 22, 23]], dtype=mx.uint32),
                mx.full((1, 4), 0.25, dtype=mx.float32),
            )
        )
        misses_before = c8.misses
        x = (mx.random.normal((16, 128)) * 0.5).astype(mx.bfloat16)
        active = mx.array([1, 6, 10, 18, 28, 30], dtype=mx.uint32)
        inds = active[mx.random.randint(0, len(active), (16, 4))]
        w = mx.softmax(mx.random.normal((16, 4)), axis=-1)
        ref_e = exp_stock(x, inds, w)
        got_e = exp_off(x, inds, w)
        mx.eval(ref_e, got_e)
        assert em_calls[0] == 0
        assert c8.misses - misses_before <= 6
        assert bool(mx.array_equal(got_e, ref_e))
    finally:
        close_inkling2_moe_expert_offload(offloaded)


@pytest.mark.parametrize("fmt", ["affine", "nvfp4", "mxfp8"])
def test_capacity_lt_experts_b_sweep_around_switch_point_bit_exact(
    tmp_path, monkeypatch, fmt
):
    """Sweep B across 4*capacity (64) and 4*E (128) for capacity < E on all formats."""
    stock = _inkling2_tiny(
        formats=(_format_spec(fmt),),
        hidden_size=512,
        intermediate_size=256,
        num_attention_heads=16,
        swa_num_attention_heads=16,
        n_routed_experts=32,
        num_experts_per_tok=4,
    )
    mx.random.seed(42)
    layer0 = stock.model.layers[0]
    for pn in ("w13", "w2"):
        proj = getattr(layer0.mlp.experts, pn)
        in_d = proj.scales.shape[-1] * proj.group_size
        out_d = proj.weight.shape[1]
        wf = (mx.random.normal((32, out_d, in_d)) * 0.25).astype(mx.bfloat16)
        quantized = mx.quantize(
            wf, group_size=proj.group_size, bits=proj.bits, mode=proj.mode
        )
        proj.weight, proj.scales = quantized[0], quantized[1]
        if len(quantized) == 3:
            proj.biases = quantized[2]
        if proj.get("global_scale") is not None:
            proj.global_scale = mx.random.uniform(0.5, 4.0, (32,))
    # Bias router onto 10 experts so full-model B sweep stays within capacity=16.
    bias = mx.full((32,), -100.0, dtype=mx.float32)
    bias[:10] = mx.arange(10, 0, -1, dtype=mx.float32) * 0.2
    layer0.mlp.router_bias = bias
    pred = stock.cast_predicate
    stock.load_weights(
        [
            (k, v.astype(mx.bfloat16) if pred(k) and v.dtype == mx.float32 else v)
            for k, v in tree_flatten(stock.parameters())
        ]
    )
    mx.eval(stock.parameters())
    _save_runtime_checkpoint(tmp_path, stock)

    offloaded = _reload_inkling2(tmp_path, lazy=True)
    assert apply_inkling2_moe_expert_offload(offloaded, tmp_path, 0.5) == 1
    mx.eval(offloaded.parameters())
    assert materialize_offload_state(offloaded) == 1
    c0 = offloaded.model.layers[0].mlp.cache
    assert c0.capacity == 16

    em_calls = [0]
    orig_em = OffloadedInkling2Experts._forward_expert_major

    def spy_em(self, *args, **kwargs):
        em_calls[0] += 1
        return orig_em(self, *args, **kwargs)

    monkeypatch.setattr(OffloadedInkling2Experts, "_forward_expert_major", spy_em)

    try:
        # Pre-warm 4 unused experts so active experts map to non-identity slots.
        exp_stock = stock.model.layers[0].mlp.experts
        exp_off = offloaded.model.layers[0]._split_experts
        warm_x = (mx.random.normal((1, 512)) * 0.5).astype(mx.bfloat16)
        mx.eval(
            exp_off(
                warm_x,
                mx.array([[24, 25, 26, 27]], dtype=mx.uint32),
                mx.full((1, 4), 0.25, dtype=mx.float32),
            )
        )

        # Isolated expert sweep: B = 4*T in {40, 60, 64, 68, 96, 124, 128, 132, 160}.
        active = mx.array([0, 2, 5, 7, 9, 11, 14, 17, 19, 21], dtype=mx.uint32)
        for tokens in (10, 15, 16, 17, 24, 31, 32, 33, 40):
            mx.random.seed(1000 + tokens)
            x = (mx.random.normal((tokens, 512)) * 0.5).astype(mx.bfloat16)
            inds = active[mx.random.randint(0, len(active), (tokens, 4))]
            weights = mx.softmax(mx.random.normal((tokens, 4)), axis=-1)
            ref = exp_stock(x, inds, weights)
            got = exp_off(x, inds, weights)
            mx.eval(ref, got)
            assert em_calls[0] == 0
            assert bool(
                mx.array_equal(got, ref)
            ), f"{fmt} isolated mismatch at B={tokens * 4}"

        # Full-model sweep across the same B boundaries (eager and compiled lengths).
        for tokens in (10, 15, 16, 24, 31, 32, 33):
            mx.random.seed(2000 + tokens)
            ids = mx.random.randint(0, 300, (1, tokens))
            ref_m = stock(ids)[..., :300]
            got_m = offloaded(ids)[..., :300]
            mx.eval(ref_m, got_m)
            assert em_calls[0] == 0
            assert bool(
                mx.array_equal(got_m, ref_m)
            ), f"{fmt} full-model mismatch at B={tokens * 4}"
    finally:
        close_inkling2_moe_expert_offload(offloaded)
