# SPDX-License-Identifier: MIT
"""Wiring tests for Inkling 2 MoE expert offload."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.test_inkling2_moe_offload import (
    _build_tiny_model,
    _enable_split_without_offload,
    _format_spec,
    _max_abs_diff,
    _save_runtime_checkpoint,
    _save_source_checkpoint,
)


def _write_offline_tokenizer(directory: Path) -> None:
    """Write minimal local WordLevel tokenizer + processor config for offline load."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    config_path = directory / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    vocab_size = int(config.get("text_config", config).get("vocab_size", 300))
    vocab = {f"t{i}": i for i in range(vocab_size)}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="t0"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="t0", eos_token="t2"
    ).save_pretrained(directory)
    # _ensure_pad_token resolves eos_token_id; keep it inside the tiny vocab.
    config["eos_token_id"] = 2
    config["image_token_id"] = 1
    config_path.write_text(json.dumps(config), encoding="utf-8")
    (directory / "processor_config.json").write_text(
        json.dumps({"image_token": "t1"}), encoding="utf-8"
    )


def _save_wiring_checkpoint(
    directory: Path,
    model,
    *,
    layout: str,
    denom_value: float = 1344.0,
    extra_drops: tuple[str, ...] = (),
) -> Path:
    if layout == "runtime":
        _save_runtime_checkpoint(directory, model, extra_drops=extra_drops)
    elif layout == "source":
        _save_source_checkpoint(
            directory, model, denom_value=denom_value, extra_drops=extra_drops
        )
    else:
        raise ValueError(f"Unknown layout: {layout!r}")
    _write_offline_tokenizer(directory)
    return directory


def _load_inkling2_vlm(path: Path, *, lazy: bool):
    from omlx.patches.inkling2 import vlm as inkling2_vlm

    return inkling2_vlm.load(path, lazy=lazy)


def _removed_bytes_from_model(model) -> int:
    from omlx.patches.moe_expert_offload import moe_offload_caches

    removed = 0
    for cache in moe_offload_caches(model):
        removed += (cache.n_experts - cache.capacity) * cache.expert_bytes
    return removed


def _install_expert_major_spy(model) -> dict[str, int]:
    counts = {"expert_major": 0}
    for layer in model.model.layers:
        mod = getattr(layer, "_split_experts", None)
        if mod is None or not hasattr(mod, "_forward_expert_major"):
            continue
        orig = mod._forward_expert_major

        def tracked(*args, _orig=orig, **kwargs):
            counts["expert_major"] += 1
            return _orig(*args, **kwargs)

        mod._forward_expert_major = tracked
    return counts


def _layer_expert_tensor_bytes(model) -> dict[str, tuple[tuple[int, ...], int]]:
    out = {}
    for idx, layer in enumerate(model.model.layers):
        experts = layer.mlp.experts
        for proj_name in ("w13", "w2"):
            proj = getattr(experts, proj_name)
            for field in ("weight", "scales", "biases"):
                if proj.get(field) is not None:
                    arr = proj[field]
                    out[f"{idx}.{proj_name}.{field}"] = (tuple(arr.shape), arr.nbytes)
    return out


@pytest.mark.parametrize(
    "layout,fmt",
    [
        ("runtime", "nvfp4"),
        ("runtime", "mxfp8"),
        ("runtime", "affine"),
        ("source", "nvfp4"),
        ("source", "mxfp8"),
    ],
)
@pytest.mark.parametrize("residency", [1.0, 0.5, 0.25])
def test_common_entry_vlm_load_matches_stock_decode_prefill_and_batch2(
    tmp_path, layout, fmt, residency
):
    from omlx.patches.inkling2.moe_offload import close_inkling2_moe_expert_offload
    from omlx.patches.moe_expert_offload import (
        apply_moe_expert_offload,
        materialize_offload_state,
        moe_offload_caches,
        moe_offload_stats,
    )
    from omlx.utils.model_loading import materialize_lazy_state

    formats = (_format_spec(fmt),) * 6
    seed_model = _build_tiny_model(formats, num_experts=8, top_k=2, seed=41)
    _save_wiring_checkpoint(tmp_path, seed_model, layout=layout)

    stock_fused, stock_proc = _load_inkling2_vlm(tmp_path, lazy=False)
    assert stock_proc is not None
    stock_split, _ = _load_inkling2_vlm(tmp_path, lazy=False)
    _enable_split_without_offload(stock_split)

    offloaded, off_proc = _load_inkling2_vlm(tmp_path, lazy=True)
    assert off_proc is not None

    wrapped = apply_moe_expert_offload(offloaded, tmp_path, residency)
    assert wrapped == 6
    assert materialize_offload_state(offloaded) == 6
    materialize_lazy_state(offloaded)
    caches = moe_offload_caches(offloaded)
    assert len(caches) == 6
    expected_cap = max(2, round(8 * residency))
    assert all(c.capacity == expected_cap for c in caches)
    spy = _install_expert_major_spy(offloaded)

    try:
        # 1. Decode (B=1, T=1 steps after prompt): always stays within capacity.
        mx.random.seed(11)
        prompt = mx.random.randint(0, 300, (1, 6))
        cf, cs, co = (
            stock_fused.make_cache(),
            stock_split.make_cache(),
            offloaded.make_cache(),
        )
        mx.eval(
            stock_fused(prompt, cache=cf),
            stock_split(prompt, cache=cs),
            offloaded(prompt, cache=co),
        )
        spy["expert_major"] = 0
        for step in range(4):
            tok = mx.random.randint(0, 300, (1, 1))
            lf = stock_fused(tok, cache=cf)[..., :300]
            ls = stock_split(tok, cache=cs)[..., :300]
            lo = offloaded(tok, cache=co)[..., :300]
            mx.eval(lf, ls, lo)
            assert spy["expert_major"] == 0, f"decode step {step} took expert-major"
            assert bool(mx.array_equal(lo, ls))
            assert _max_abs_diff(lo, lf) <= 5e-4

        # 2. Chunked prefill across compiled/eager boundaries (chunks 4, 8, 20).
        mx.random.seed(22)
        long_prompt = mx.random.randint(0, 300, (1, 32))
        cf_c, cs_c, co_c = (
            stock_fused.make_cache(),
            stock_split.make_cache(),
            offloaded.make_cache(),
        )
        start = 0
        for chunk in (4, 8, 20):
            seg = long_prompt[:, start : start + chunk]
            start += chunk
            before_em = spy["expert_major"]
            lf = stock_fused(seg, cache=cf_c)[..., :300]
            ls = stock_split(seg, cache=cs_c)[..., :300]
            lo = offloaded(seg, cache=co_c)[..., :300]
            mx.eval(lf, ls, lo)
            used_em = spy["expert_major"] > before_em
            if residency == 1.0:
                assert not used_em
                assert bool(mx.array_equal(lo, ls))
                if chunk > 16:
                    # Above _COMPILED_MAX_TOKENS stock runs eager too.
                    assert bool(mx.array_equal(lo, lf))
            else:
                if used_em:
                    assert _max_abs_diff(lo, ls) < 2e-2
                else:
                    assert bool(mx.array_equal(lo, ls))
                assert _max_abs_diff(lo, lf) < 2e-2

        # 3. Batch-2 prefill + decode.
        mx.random.seed(33)
        batch_seq = mx.random.randint(0, 300, (2, 10))
        cb_f, cb_s, cb_o = (
            stock_fused.make_cache(),
            stock_split.make_cache(),
            offloaded.make_cache(),
        )
        mx.eval(
            stock_fused(batch_seq[:, :6], cache=cb_f),
            stock_split(batch_seq[:, :6], cache=cb_s),
            offloaded(batch_seq[:, :6], cache=cb_o),
        )
        for t in range(6, 10):
            before_em = spy["expert_major"]
            lf = stock_fused(batch_seq[:, t : t + 1], cache=cb_f)[..., :300]
            ls = stock_split(batch_seq[:, t : t + 1], cache=cb_s)[..., :300]
            lo = offloaded(batch_seq[:, t : t + 1], cache=cb_o)[..., :300]
            mx.eval(lf, ls, lo)
            used_em = spy["expert_major"] > before_em
            if expected_cap >= 4 or not used_em:
                assert not used_em
                assert bool(mx.array_equal(lo, ls))
                assert _max_abs_diff(lo, lf) <= 5e-4
            else:
                assert _max_abs_diff(lo, ls) < 2e-2
                assert _max_abs_diff(lo, lf) < 2e-2

        stats = moe_offload_stats(offloaded)
        assert stats["layers"] == 6
        assert stats["fetched_bytes"] > 0
    finally:
        close_inkling2_moe_expert_offload(offloaded)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8", "affine"])
@pytest.mark.parametrize("chunk_len", [7, 8, 9, 15, 16, 17, 20])
def test_in_capacity_route_switch_points_and_compute_dtypes(tmp_path, fmt, chunk_len):
    """Vary route counts across sorted QMM switch points (16 and 4*E=32) & dtype."""
    from omlx.patches.inkling2.moe_offload import close_inkling2_moe_expert_offload
    from omlx.patches.moe_expert_offload import (
        apply_moe_expert_offload,
        materialize_offload_state,
    )
    from omlx.utils.model_loading import materialize_lazy_state

    # E=8, top_k=2, r=0.25 -> cap=2. Bias router strongly onto experts 0, 1 so
    # every token routes strictly inside capacity even as T*k crosses 16 and 32.
    formats = (_format_spec(fmt),) * 6
    cast_bf16 = fmt != "affine"
    model = _build_tiny_model(
        formats, num_experts=8, top_k=2, seed=57, cast_bf16=cast_bf16
    )
    for layer in model.model.layers:
        bias = mx.full((8,), -100.0, dtype=mx.float32)
        bias[:2] = mx.array([4.0, 2.0], dtype=mx.float32)
        layer.mlp.router_bias = bias
    mx.eval(model.parameters())
    _save_wiring_checkpoint(tmp_path, model, layout="runtime")

    stock_split, _ = _load_inkling2_vlm(tmp_path, lazy=False)
    _enable_split_without_offload(stock_split)

    offloaded, _ = _load_inkling2_vlm(tmp_path, lazy=True)
    assert apply_moe_expert_offload(offloaded, tmp_path, 0.25) == 6
    assert materialize_offload_state(offloaded) == 6
    materialize_lazy_state(offloaded)
    spy = _install_expert_major_spy(offloaded)

    try:
        mx.random.seed(71 + chunk_len)
        ids = mx.random.randint(0, 300, (1, chunk_len))
        ls = stock_split(ids)[..., :300]
        lo = offloaded(ids)[..., :300]
        mx.eval(ls, lo)
        assert spy["expert_major"] == 0
        assert bool(mx.array_equal(lo, ls))
    finally:
        close_inkling2_moe_expert_offload(offloaded)


@pytest.mark.parametrize(
    "num_experts,top_k,residency,cast_bf16,batch_size",
    [
        (6, 2, 0.5, True, 3),
        (10, 2, 0.25, False, 2),
    ],
)
def test_varied_expert_counts_f32_compute_and_batch_gt_2(
    tmp_path, num_experts, top_k, residency, cast_bf16, batch_size
):
    """Break-testing: non-power-of-2 E, f32 compute weights, and batch > 2."""
    from omlx.patches.inkling2.moe_offload import close_inkling2_moe_expert_offload
    from omlx.patches.moe_expert_offload import (
        apply_moe_expert_offload,
        estimate_offload_admission_bytes,
        materialize_offload_state,
    )
    from omlx.utils.model_loading import materialize_lazy_state

    formats = (_format_spec("nvfp4"), _format_spec("mxfp8")) * 3
    model = _build_tiny_model(
        formats, num_experts=num_experts, top_k=top_k, seed=105, cast_bf16=cast_bf16
    )
    _save_wiring_checkpoint(tmp_path, model, layout="runtime")

    stock_split, _ = _load_inkling2_vlm(tmp_path, lazy=False)
    _enable_split_without_offload(stock_split)

    offloaded, _ = _load_inkling2_vlm(tmp_path, lazy=True)
    assert apply_moe_expert_offload(offloaded, tmp_path, residency) == 6
    assert materialize_offload_state(offloaded) == 6
    materialize_lazy_state(offloaded)
    spy = _install_expert_major_spy(offloaded)

    try:
        full_size = 100_000_000
        assert full_size - estimate_offload_admission_bytes(
            tmp_path, full_size, residency
        ) == _removed_bytes_from_model(offloaded)

        mx.random.seed(202)
        seq = mx.random.randint(0, 300, (batch_size, 8))
        cs, co = stock_split.make_cache(), offloaded.make_cache()
        before_em = spy["expert_major"]
        ls = stock_split(seq[:, :5], cache=cs)[..., :300]
        lo = offloaded(seq[:, :5], cache=co)[..., :300]
        mx.eval(ls, lo)
        if spy["expert_major"] > before_em:
            assert _max_abs_diff(lo, ls) < 2e-2
        else:
            assert bool(mx.array_equal(lo, ls))

        for t in range(5, 8):
            before_em = spy["expert_major"]
            ls = stock_split(seq[:, t : t + 1], cache=cs)[..., :300]
            lo = offloaded(seq[:, t : t + 1], cache=co)[..., :300]
            mx.eval(ls, lo)
            if spy["expert_major"] > before_em:
                assert _max_abs_diff(lo, ls) < 2e-2
            else:
                assert bool(mx.array_equal(lo, ls))
    finally:
        close_inkling2_moe_expert_offload(offloaded)


def test_over_capacity_chunk_explicitly_uses_expert_major(tmp_path):
    """Force > capacity distinct experts and verify _forward_expert_major runs."""
    from omlx.patches.inkling2.moe_offload import close_inkling2_moe_expert_offload
    from omlx.patches.moe_expert_offload import (
        apply_moe_expert_offload,
        materialize_offload_state,
    )
    from omlx.utils.model_loading import materialize_lazy_state

    formats = (_format_spec("mxfp8"),) * 6
    model = _build_tiny_model(formats, num_experts=12, top_k=2, seed=81)
    _save_wiring_checkpoint(tmp_path, model, layout="runtime")

    stock, _ = _load_inkling2_vlm(tmp_path, lazy=False)
    offloaded, _ = _load_inkling2_vlm(tmp_path, lazy=True)
    # cap = max(2, round(12 * 0.25)) = 3; prompt len 32 hits many more than 3.
    assert apply_moe_expert_offload(offloaded, tmp_path, 0.25) == 6
    assert materialize_offload_state(offloaded) == 6
    materialize_lazy_state(offloaded)
    spy = _install_expert_major_spy(offloaded)

    try:
        mx.random.seed(91)
        ids = mx.random.randint(0, 300, (1, 32))
        ref = stock(ids)[..., :300]
        got = offloaded(ids)[..., :300]
        mx.eval(ref, got)
        assert spy["expert_major"] > 0
        assert _max_abs_diff(got, ref) < 2e-2
    finally:
        close_inkling2_moe_expert_offload(offloaded)


@pytest.mark.parametrize("layout", ["runtime", "source"])
def test_lazy_load_never_materializes_full_experts(tmp_path, layout):
    """Verify lazy=True keeps experts unevaluated until wrapped to slot shapes."""
    from omlx.patches.inkling2.moe_offload import close_inkling2_moe_expert_offload
    from omlx.patches.moe_expert_offload import (
        apply_moe_expert_offload,
        materialize_offload_state,
    )
    from omlx.utils.model_loading import materialize_lazy_state

    # Make expert tables large enough (~75 MB in mxfp8) to measure memory deltas.
    formats = (_format_spec("mxfp8"),) * 6
    big = _build_tiny_model(
        formats,
        num_experts=64,
        top_k=2,
        seed=123,
    )
    # Widen intermediate_size? Keep _TINY dimensions? 6 layers * 64 experts of 128x64:
    # Rebuild with larger dims via direct ModelArgs so tables exceed ~40 MB.
    from omlx.patches.inkling2.model import Model as Inkling2Model
    from omlx.patches.inkling2.model import ModelArgs

    cfg = {
        "model_type": "inkling2",
        "eos_token_id": 2,
        "image_token_id": 1,
        "text_config": {
            "vocab_size": 300,
            "unpadded_vocab_size": 290,
            "hidden_size": 256,
            "intermediate_size": 256,
            "num_hidden_layers": 6,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "swa_num_attention_heads": 4,
            "swa_num_key_value_heads": 2,
            "swa_head_dim": 64,
            "sliding_window_size": 32,
            "d_rel": 8,
            "rel_extent": 32,
            "sconv_kernel_size": 4,
            "n_layers_per_attn_res_block": 3,
            "n_routed_experts": 64,
            "num_experts_per_tok": 2,
            "expert_formats": [_format_spec("mxfp8") for _ in range(6)],
        },
    }
    del big
    big = Inkling2Model(ModelArgs.from_dict(cfg))
    mx.eval(big.parameters())
    _save_wiring_checkpoint(tmp_path, big, layout=layout)
    del big
    mx.synchronize()
    mx.clear_cache()

    base_mem = mx.get_active_memory()

    # Load lazily: expert arrays still report [64, ...] shapes, but active memory
    # stays well below full materialized tables.
    lazy_model, _ = _load_inkling2_vlm(tmp_path, lazy=True)
    before_wrap = _layer_expert_tensor_bytes(lazy_model)
    assert all(shape[0] == 64 for shape, _ in before_wrap.values())
    full_expert_bytes = sum(nb for _, nb in before_wrap.values())
    assert full_expert_bytes > 50 * 1024 * 1024
    lazy_delta = mx.get_active_memory() - base_mem
    assert lazy_delta < full_expert_bytes * 0.25

    assert apply_moe_expert_offload(lazy_model, tmp_path, 0.25) == 6
    assert materialize_offload_state(lazy_model) == 6
    materialize_lazy_state(lazy_model)

    try:
        after_wrap = _layer_expert_tensor_bytes(lazy_model)
        assert set(after_wrap.keys()) == set(before_wrap.keys())
        assert all(shape[0] == 16 for shape, _ in after_wrap.values())
        slot_expert_bytes = sum(nb for _, nb in after_wrap.values())
        assert slot_expert_bytes * 4 == full_expert_bytes

        mx.synchronize()
        mx.clear_cache()
        offload_mem = mx.get_active_memory() - base_mem
        assert offload_mem < full_expert_bytes * 0.5
    finally:
        close_inkling2_moe_expert_offload(lazy_model)
        del lazy_model
        mx.synchronize()
        mx.clear_cache()


async def _run_vlm_engine_load_sequence(tmp_path: Path) -> None:
    """Drive real VLMBatchedEngine.start()/stop() on Inkling 2 with offload on/off."""
    from omlx.engine import vlm as vlm_module
    from omlx.engine.vlm import VLMBatchedEngine
    from omlx.model_settings import ModelSettings
    from omlx.patches.inkling2.moe_offload import close_inkling2_moe_expert_offload
    from omlx.patches.moe_expert_offload import moe_offload_caches
    from omlx.scheduler import SchedulerConfig

    formats = (_format_spec("nvfp4"),) * 3 + (_format_spec("mxfp8"),) * 3
    model = _build_tiny_model(formats, num_experts=8, top_k=2, seed=31)
    _save_wiring_checkpoint(tmp_path, model, layout="runtime")

    lazy_calls = []
    orig_load = vlm_module.inkling2_vlm.load

    def spy_load(path, *, lazy=False):
        lazy_calls.append(bool(lazy))
        return orig_load(path, lazy=lazy)

    sched_cfg = SchedulerConfig(max_num_seqs=2)

    with patch.object(vlm_module.inkling2_vlm, "load", spy_load):
        # Offload enabled -> lazy=True load -> 6 layers wrapped & materialized.
        offload_engine = VLMBatchedEngine(
            model_name=str(tmp_path),
            scheduler_config=sched_cfg,
            model_settings=ModelSettings(
                moe_expert_offload_enabled=True,
                moe_expert_offload_resident_fraction=0.25,
            ),
        )
        await offload_engine.start()
        try:
            assert lazy_calls[-1] is True
            assert offload_engine._moe_offload_wrapped == 6
            assert offload_engine._loaded is True
            caches = moe_offload_caches(offload_engine._vlm_model)
            assert len(caches) == 6
            assert all(c.capacity == 2 for c in caches)
            for layer in offload_engine._vlm_model.model.layers:
                assert layer.mlp.experts.w13.weight.shape[0] == 2
                assert layer.mlp.experts.w2.weight.shape[0] == 2
            stats = offload_engine._engine.engine.scheduler.moe_offload_stats()
            assert stats["layers"] == 6
            assert offload_engine._engine.engine.scheduler.moe_offload_release() == 0
            assert offload_engine._engine.engine.scheduler.moe_offload_restore() is None
        finally:
            if offload_engine._vlm_model is not None:
                close_inkling2_moe_expert_offload(offload_engine._vlm_model)
            await offload_engine.stop()

        # Source layout also loads lazily and wraps cleanly through VLMBatchedEngine.
        src_dir = tmp_path / "src_vlm"
        src_formats = (_format_spec("nvfp4"),) * 4 + (_format_spec("mxfp8"),) * 2
        _save_wiring_checkpoint(
            src_dir,
            _build_tiny_model(src_formats, num_experts=8, top_k=2, seed=32),
            layout="source",
        )
        src_engine = VLMBatchedEngine(
            model_name=str(src_dir),
            scheduler_config=sched_cfg,
            model_settings=ModelSettings(
                moe_expert_offload_enabled=True,
                moe_expert_offload_resident_fraction=0.25,
            ),
        )
        await src_engine.start()
        try:
            assert lazy_calls[-1] is True
            assert src_engine._moe_offload_wrapped == 6
            assert len(moe_offload_caches(src_engine._vlm_model)) == 6
        finally:
            if src_engine._vlm_model is not None:
                close_inkling2_moe_expert_offload(src_engine._vlm_model)
            await src_engine.stop()

        # Load-time rejection fires in VLMBatchedEngine.start() BEFORE loading weights.
        calls_before = len(lazy_calls)
        bad_engine = VLMBatchedEngine(
            model_name=str(tmp_path),
            scheduler_config=sched_cfg,
            model_settings=ModelSettings(
                moe_expert_offload_enabled=True,
                prefill_helper_node_id="helper-mac",
            ),
        )
        with pytest.raises(ValueError, match="Prefill Accelerator"):
            await bad_engine.start()
        assert len(lazy_calls) == calls_before

        # Offload disabled -> eager load (lazy=False) -> 0 wrapped layers.
        eager_engine = VLMBatchedEngine(
            model_name=str(tmp_path),
            scheduler_config=sched_cfg,
            model_settings=ModelSettings(moe_expert_offload_enabled=False),
        )
        await eager_engine.start()
        try:
            assert lazy_calls[-1] is False
            assert eager_engine._moe_offload_wrapped == 0
            for layer in eager_engine._vlm_model.model.layers:
                assert layer.mlp.experts.w13.weight.shape[0] == 8
        finally:
            await eager_engine.stop()


def test_vlm_engine_load_sequence_offload_and_off_toggle(tmp_path):
    """Isolate real VLMBatchedEngine.start()/stop() patches in a subprocess.

    VLMBatchedEngine.start() installs process-wide Metal/attention/model patches
    (sdpa256/fa256, qwen35_gdn_prework, qwen35_q4_mlp) that stop() does not undo.
    Keep those real hooks in a subprocess so unrelated unit tests see stock hooks.
    """
    script = """
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import tests.test_inkling2_offload_wiring as t

coro = t._run_vlm_engine_load_sequence(Path(sys.argv[2]))
asyncio.run(asyncio.wait_for(coro, timeout=60))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(Path(__file__).resolve().parent.parent),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_estimator_exact_parity_runtime_source_and_skipped_layers(tmp_path):
    """Estimator matches adapter removed bytes to the exact byte (and skips bad layers)."""
    from omlx.patches.inkling2.moe_offload import close_inkling2_moe_expert_offload
    from omlx.patches.moe_expert_offload import (
        apply_moe_expert_offload,
        estimate_offload_admission_bytes,
    )

    full_size = 100_000_000

    # 1. Mixed runtime formats (nvfp4, mxfp8, affine) -> global_scale NOT discounted.
    rt_dir = tmp_path / "rt_mixed"
    rt_formats = (
        _format_spec("nvfp4"),
        _format_spec("mxfp8"),
        _format_spec("affine"),
    ) * 2
    rt_model = _build_tiny_model(rt_formats, num_experts=8, top_k=2, seed=61)
    _save_wiring_checkpoint(rt_dir, rt_model, layout="runtime")
    for fraction in (0.1, 0.25, 0.5, 1.0):
        loaded, _ = _load_inkling2_vlm(rt_dir, lazy=True)
        try:
            wrapped = apply_moe_expert_offload(loaded, rt_dir, fraction)
            assert wrapped == 6
            expected_removed = _removed_bytes_from_model(loaded)
            estimated = estimate_offload_admission_bytes(rt_dir, full_size, fraction)
            assert full_size - estimated == expected_removed
        finally:
            close_inkling2_moe_expert_offload(loaded)

    # 2. Thinking Machines source layout (mixed nvfp4 + mxfp8, custom denom).
    src_dir = tmp_path / "source_mixed"
    src_formats = (_format_spec("nvfp4"),) * 4 + (_format_spec("mxfp8"),) * 2
    src_model = _build_tiny_model(src_formats, num_experts=8, top_k=2, seed=62)
    _save_wiring_checkpoint(src_dir, src_model, layout="source", denom_value=2688.0)
    for fraction in (0.25, 0.5):
        loaded, _ = _load_inkling2_vlm(src_dir, lazy=True)
        try:
            assert apply_moe_expert_offload(loaded, src_dir, fraction) == 6
            expected_removed = _removed_bytes_from_model(loaded)
            estimated = estimate_offload_admission_bytes(src_dir, full_size, fraction)
            assert full_size - estimated == expected_removed
        finally:
            close_inkling2_moe_expert_offload(loaded)

    # 3. Runtime layer refused by adapter (missing nvfp4 global_scale on layer 2):
    # adapter wraps 5 layers; estimator must promise savings for those 5 layers only.
    from mlx_lm.utils import load_model

    from omlx.patches.inkling2 import apply_inkling2_patch
    from omlx.patches.inkling2.model import Model as Inkling2Model
    from omlx.patches.inkling2.model import ModelArgs

    partial_dir = tmp_path / "partial_skip"
    partial_formats = (_format_spec("nvfp4"),) * 6
    partial_model = _build_tiny_model(partial_formats, num_experts=8, top_k=2, seed=63)
    _save_wiring_checkpoint(
        partial_dir,
        partial_model,
        layout="runtime",
        extra_drops=("model.layers.2.mlp.experts.w2.global_scale",),
    )
    apply_inkling2_patch()
    loaded, _ = load_model(
        partial_dir,
        lazy=True,
        strict=False,
        get_model_classes=lambda config: (Inkling2Model, ModelArgs),
    )
    try:
        assert apply_moe_expert_offload(loaded, partial_dir, 0.25) == 5
        expected_removed = _removed_bytes_from_model(loaded)
        assert expected_removed > 0
        estimated = estimate_offload_admission_bytes(partial_dir, full_size, 0.25)
        assert full_size - estimated == expected_removed
    finally:
        close_inkling2_moe_expert_offload(loaded)

    # 4. Source layer refused by adapter through real inkling2_vlm.load:
    # non-float amax dtype passes sanitize/strict load_weights, yet resolve_view
    # refuses layer 1 -> adapter wraps 5 layers, estimator skips layer 1.
    import struct

    src_skip_dir = tmp_path / "src_skip"
    _save_wiring_checkpoint(src_skip_dir, src_model, layout="source")
    shard = src_skip_dir / "model.safetensors"
    blob = shard.read_bytes()
    hdr_len = struct.unpack("<Q", blob[:8])[0]
    hdr = json.loads(blob[8 : 8 + hdr_len])
    amax_key = "model.llm.layers.1.mlp.experts.w13_weight_amax"
    assert hdr[amax_key]["dtype"] == "F32"
    hdr[amax_key]["dtype"] = "I32"
    new_hdr = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    pad = (-len(new_hdr)) % 8
    new_hdr += b" " * pad
    shard.write_bytes(struct.pack("<Q", len(new_hdr)) + new_hdr + blob[8 + hdr_len :])

    loaded_src, _ = _load_inkling2_vlm(src_skip_dir, lazy=True)
    try:
        assert apply_moe_expert_offload(loaded_src, src_skip_dir, 0.25) == 5
        expected_removed = _removed_bytes_from_model(loaded_src)
        assert expected_removed > 0
        assert (
            full_size - estimate_offload_admission_bytes(src_skip_dir, full_size, 0.25)
            == expected_removed
        )
    finally:
        close_inkling2_moe_expert_offload(loaded_src)


def test_compat_and_moe_offload_fit_for_inkling2(tmp_path):
    """Compat approves valid runtime/source checkpoints, rejects non-offloadable ones."""
    from omlx.engine_pool import EngineEntry, EnginePool
    from omlx.model_settings import ModelSettings
    from omlx.patches.moe_offload_compat import moe_offload_compatibility

    rt_dir = tmp_path / "compat_rt"
    m_rt = _build_tiny_model((_format_spec("nvfp4"),) * 6, num_experts=8, top_k=2)
    _save_wiring_checkpoint(rt_dir, m_rt, layout="runtime")
    assert moe_offload_compatibility(rt_dir) == (True, "")

    src_dir = tmp_path / "compat_src"
    _save_wiring_checkpoint(src_dir, m_rt, layout="source")
    assert moe_offload_compatibility(src_dir) == (True, "")

    # Not offloadable when n_experts == top_k (capacity equals E -> 0 savings).
    no_fit_dir = tmp_path / "compat_eq_topk"
    m_small = _build_tiny_model((_format_spec("mxfp8"),) * 6, num_experts=2, top_k=2)
    _save_wiring_checkpoint(no_fit_dir, m_small, layout="runtime")
    ok, _reason = moe_offload_compatibility(no_fit_dir)
    assert ok is False

    # Not offloadable when checkpoint lacks required expert scales/globals on all layers.
    broken_dir = tmp_path / "compat_broken"
    _save_wiring_checkpoint(
        broken_dir,
        m_rt,
        layout="runtime",
        extra_drops=tuple(
            f"model.layers.{i}.mlp.experts.w2.global_scale" for i in range(6)
        ),
    )
    ok_broken, _ = moe_offload_compatibility(broken_dir)
    assert ok_broken is False

    # moe_offload_fit uses estimate_offload_admission_bytes via _entry_runtime_resident_size.
    full_size = 1_000_000_000
    est_25 = EnginePool.__new__(EnginePool)
    est_25._scheduler_config = SimpleNamespace(hot_cache_max_size=0)
    est_25._current_model_memory = 0
    est_25._process_memory_enforcer = SimpleNamespace(
        _prefill_headroom_safety=0.8, _hard_threshold=0.8
    )
    est_25._entries = {}
    est_25._distributed_deployment_for_entry = lambda _e: None
    est_25.external_engine_id = lambda *_a: None
    est_25._qwen4_ple_offload_status = lambda *_a, **_k: (False, False, None)
    est_25._deepseek_v41_engram_offload_status = lambda *_a, **_k: (False, False, None)

    entry = EngineEntry(
        model_id="tiny-inkling2",
        model_path=str(rt_dir),
        model_type="vlm",
        engine_type="vlm",
        estimated_size=full_size,
        config_model_type="inkling2",
    )
    settings = ModelSettings()
    # Resident full size exceeds budget while 25% offload saves >= 1 byte? Let's pick
    # budget right between offloaded size and full_size.
    off_25 = est_25._entry_runtime_resident_size(
        entry,
        ModelSettings(
            moe_expert_offload_enabled=True,
            moe_expert_offload_resident_fraction=0.25,
        ),
        base_size=full_size,
    )
    assert off_25 < full_size
    reserve = 4 * 1024**3  # _MOE_OFFLOAD_PROMPT_RESERVE
    # Choose ceiling so budget = int(ceiling * 0.8) - reserve is between off_25 and full_size.
    mid = (off_25 + full_size) // 2
    ceiling = int((mid + reserve) / 0.8)
    fit = est_25.moe_offload_fit(entry, settings, ceiling)
    assert fit["status"] == "fit"
    assert isinstance(fit["max_percent"], int) and 0 < fit["max_percent"] < 100
    assert fit["bytes"] <= mid


def test_validation_rules_direct_load_time_and_admin_api(tmp_path, monkeypatch):
    """MTP/DSpark & Prefill Accelerator combinations cannot be bypassed."""
    from omlx.admin import routes as admin_routes
    from omlx.engine_pool import EngineEntry
    from omlx.model_settings import ModelSettings, validate_moe_expert_offload
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    # 1. Direct validate_moe_expert_offload rules
    # (a) MTP / DSpark already rejected by generic rule
    with pytest.raises(ValueError, match="speculative decoding"):
        validate_moe_expert_offload(
            {"moe_expert_offload_enabled": True, "mtp_enabled": True},
            model_type="inkling2",
        )
    # (b) Prefill Accelerator rejected for inkling2
    with pytest.raises(ValueError, match="Prefill Accelerator"):
        validate_moe_expert_offload(
            {
                "moe_expert_offload_enabled": True,
                "prefill_helper_node_id": "helper-mac",
            },
            model_type="inkling2",
        )
    # Offload alone or Prefill Accelerator without offload allowed
    validate_moe_expert_offload(
        {"moe_expert_offload_enabled": True}, model_type="inkling2"
    )
    validate_moe_expert_offload(
        {"moe_expert_offload_enabled": True, "prefill_helper_node_id": "   "},
        model_type="inkling2",
    )
    validate_moe_expert_offload(
        {"moe_expert_offload_enabled": False, "prefill_helper_node_id": "helper-mac"},
        model_type="inkling2",
    )
    # Non-inkling2 models unaffected by Prefill Accelerator restriction
    validate_moe_expert_offload(
        {
            "moe_expert_offload_enabled": True,
            "prefill_helper_node_id": "helper-mac",
        },
        model_type="qwen3_5_moe",
    )
    validate_moe_expert_offload(
        {
            "moe_expert_offload_enabled": True,
            "prefill_helper_node_id": "helper-mac",
        },
        model_type=None,
    )

    # 2. Load-time validation via maybe_apply_pre_load_patches (non-bypassable)
    m = _build_tiny_model((_format_spec("nvfp4"),) * 6, num_experts=8, top_k=2)
    _save_wiring_checkpoint(tmp_path, m, layout="runtime")

    with pytest.raises(ValueError, match="Prefill Accelerator"):
        maybe_apply_pre_load_patches(
            str(tmp_path),
            model_settings=ModelSettings(
                moe_expert_offload_enabled=True,
                prefill_helper_node_id="helper-mac",
            ),
            for_vlm=True,
        )
    with pytest.raises(ValueError, match="speculative decoding"):
        maybe_apply_pre_load_patches(
            str(tmp_path),
            model_settings=ModelSettings(
                moe_expert_offload_enabled=True,
                mtp_enabled=True,
            ),
            for_vlm=True,
        )
    # Valid offload-only passes pre-load patches cleanly.
    maybe_apply_pre_load_patches(
        str(tmp_path),
        model_settings=ModelSettings(moe_expert_offload_enabled=True),
        for_vlm=True,
    )

    # 3. Admin API validation paths (PUT settings + profiles)
    from omlx.engine_pool import EnginePool
    from omlx.model_settings import ModelSettingsManager

    mgr = ModelSettingsManager(tmp_path)
    entry = EngineEntry(
        model_id="inkling2-test",
        model_path=str(tmp_path),
        model_type="vlm",
        engine_type="vlm",
        estimated_size=1_000_000_000,
        config_model_type="inkling2",
    )
    est_pool = EnginePool.__new__(EnginePool)
    est_pool._entries = {"inkling2-test": entry}
    est_pool._settings_manager = mgr
    est_pool._scheduler_config = SimpleNamespace(hot_cache_max_size=0, max_num_seqs=2)
    est_pool._current_model_memory = 0
    est_pool._process_memory_enforcer = SimpleNamespace(
        _prefill_headroom_safety=0.8, _hard_threshold=0.8
    )
    est_pool._distributed_deployment_for_entry = lambda _e: None
    est_pool.external_engine_id = lambda *_a: None
    est_pool.splash_target = lambda *_a: (None, None)
    est_pool._qwen4_ple_offload_status = lambda *_a, **_k: (False, False, None)
    est_pool._deepseek_v41_engram_offload_status = lambda *_a, **_k: (
        False,
        False,
        None,
    )
    est_pool._clear_load_failure = lambda _e: None
    est_pool._current_ceiling = lambda: 5 * 1024**3

    monkeypatch.setattr(admin_routes, "_require_settings_manager", lambda: mgr)
    monkeypatch.setattr(admin_routes, "_get_settings_manager", lambda: mgr)
    monkeypatch.setattr(admin_routes, "_require_model", lambda _id: entry)
    monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: est_pool)
    monkeypatch.setattr(
        admin_routes,
        "_get_server_state",
        lambda: SimpleNamespace(
            engine_pool=est_pool, settings_manager=mgr, default_model=None
        ),
    )

    app = FastAPI()
    app.include_router(admin_routes.router)
    app.dependency_overrides[admin_routes.require_admin] = lambda: True
    client = TestClient(app)

    # Admin fit hint route accepts inkling2 cleanly via real EnginePool.moe_offload_fit.
    fit_resp = client.get("/admin/api/models/inkling2-test/moe-offload-fit")
    assert fit_resp.status_code == 200, fit_resp.text
    assert fit_resp.json()["status"] in ("fit", "resident", "no_fit")

    # Valid standalone offload succeeds.
    resp = client.put(
        "/admin/api/models/inkling2-test/settings",
        json={"moe_expert_offload_enabled": True},
    )
    assert resp.status_code == 200, resp.text

    # Adding Prefill Accelerator on top of active offload is rejected.
    resp = client.put(
        "/admin/api/models/inkling2-test/settings",
        json={"prefill_helper_node_id": "helper-mac"},
    )
    assert resp.status_code == 400
    assert "Prefill Accelerator" in resp.json()["detail"]

    # Adding MTP on top of active offload is rejected.
    resp = client.put(
        "/admin/api/models/inkling2-test/settings",
        json={"mtp_enabled": True},
    )
    assert resp.status_code == 400

    # Switch to Prefill Accelerator alone (valid), then try enabling offload via PUT,
    # via profile creation, and via recipe snapshot -> all rejected.
    resp = client.put(
        "/admin/api/models/inkling2-test/settings",
        json={
            "moe_expert_offload_enabled": False,
            "prefill_helper_node_id": "helper-mac",
        },
    )
    assert resp.status_code == 200, resp.text

    resp = client.put(
        "/admin/api/models/inkling2-test/settings",
        json={"moe_expert_offload_enabled": True},
    )
    assert resp.status_code == 400
    assert "Prefill Accelerator" in resp.json()["detail"]

    resp = client.post(
        "/admin/api/models/inkling2-test/profiles",
        json={
            "name": "bad-offload",
            "display_name": "Bad Offload",
            "settings": {"moe_expert_offload_enabled": True},
        },
    )
    assert resp.status_code == 400
    assert "Prefill Accelerator" in resp.json()["detail"]

    # Also test applying a previously valid offload profile after helper is enabled.
    mgr.save_profile(
        "inkling2-test",
        "saved-offload",
        display_name="Saved Offload",
        description=None,
        settings={"moe_expert_offload_enabled": True},
    )
    resp = client.post("/admin/api/models/inkling2-test/profiles/saved-offload/apply")
    assert resp.status_code == 400
    assert "Prefill Accelerator" in resp.json()["detail"]

    # _apply_settings_snapshot (recipe path) also rejects conflicting settings.
    import asyncio

    with pytest.raises(admin_routes.HTTPException) as recipe_exc:
        asyncio.run(
            admin_routes._apply_settings_snapshot(
                entry,
                "inkling2-test",
                {"moe_expert_offload_enabled": True},
            )
        )
    assert recipe_exc.value.status_code == 400
    assert "Prefill Accelerator" in recipe_exc.value.detail

    with pytest.raises(admin_routes.HTTPException) as exc_info:
        admin_routes._validate_model_settings(
            entry,
            {
                "moe_expert_offload_enabled": True,
                "prefill_helper_node_id": "helper-mac",
            },
        )
    assert exc_info.value.status_code == 400
    assert "Prefill Accelerator" in exc_info.value.detail


def test_generic_switch_glu_path_ignores_inkling2_without_warnings(
    tmp_path, monkeypatch, caplog
):
    """Generic SwitchGLU traversal yields nothing and emits no warnings on Inkling 2."""
    from omlx.patches import moe_expert_offload as generic_offload
    from omlx.patches.inkling2 import moe_offload as inkling2_offload

    m = _build_tiny_model((_format_spec("nvfp4"),) * 6, num_experts=8, top_k=2)
    _save_wiring_checkpoint(tmp_path, m, layout="runtime")
    loaded, _ = _load_inkling2_vlm(tmp_path, lazy=True)

    assert list(generic_offload._iter_switch_glus(loaded)) == []

    monkeypatch.setattr(
        inkling2_offload, "apply_inkling2_moe_expert_offload", lambda *_a, **_k: 0
    )
    with caplog.at_level(logging.WARNING):
        wrapped = generic_offload.apply_moe_expert_offload(loaded, tmp_path, 0.25)
    assert wrapped == 0
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
