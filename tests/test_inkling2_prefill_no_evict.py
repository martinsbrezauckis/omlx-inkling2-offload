# SPDX-License-Identifier: MIT
"""Over-capacity prefill must not evict the decode working set."""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.patches.inkling2 import moe_offload as mo
from omlx.patches.inkling2.moe_offload import (
    apply_inkling2_moe_expert_offload,
    close_inkling2_moe_expert_offload,
)
from omlx.patches.moe_expert_offload import materialize_offload_state
from tests.test_inkling2_moe_offload import (
    _build_tiny_model,
    _format_spec,
    _max_abs_diff,
    _reload_inkling2,
    _save_runtime_checkpoint,
)


def _diverse(stock, fmt):
    # Same preparation as test_over_capacity_prefill_bounded_and_single_fetch:
    # quantized N(0, 0.05) affine weights and zero-bias diverse routing.
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


@pytest.mark.parametrize("group", [1, 3, 32])
@pytest.mark.parametrize("fmt", ["mxfp8", "nvfp4", "affine"])
def test_over_capacity_prefill_keeps_resident_set(tmp_path, monkeypatch, fmt, group):
    monkeypatch.setattr(mo, "PREFILL_NO_EVICT", True)
    monkeypatch.setattr(mo, "PREFILL_GROUP", group)
    stock = _build_tiny_model((_format_spec(fmt),) * 6, num_experts=16, top_k=2)
    _diverse(stock, fmt)
    _save_runtime_checkpoint(tmp_path, stock)
    offloaded = _reload_inkling2(tmp_path, lazy=True)
    assert apply_inkling2_moe_expert_offload(offloaded, tmp_path, 0.25) == 6
    mx.eval(offloaded.parameters())
    assert materialize_offload_state(offloaded) == 6
    caches = [layer.mlp.cache for layer in offloaded.model.layers]
    try:
        # Decode-style warm-up fills every layer's 4 slots.
        mx.random.seed(5)
        for _ in range(6):
            tok = mx.random.randint(0, 300, (1, 1))
            mx.eval(offloaded(tok))
        before = [sorted(c.slot_of) for c in caches]
        assert all(len(b) == c.capacity for b, c in zip(before, caches))
        misses_before = [c.misses for c in caches]

        mx.random.seed(77)
        ids = mx.random.randint(0, 300, (1, 48))  # 96 routes over ~16 experts > 4
        ref = stock(ids)[..., :300]
        got = offloaded(ids)[..., :300]
        mx.eval(ref, got)
        delta = _max_abs_diff(got, ref)
        assert delta < 2e-2, f"fmt={fmt} group={group} delta={delta}"

        after = [sorted(c.slot_of) for c in caches]
        assert after == before, "over-capacity prefill changed the resident set"
        # Non-resident experts were still read (counted as misses), just not installed.
        assert all(c.misses > m for c, m in zip(caches, misses_before))
    finally:
        close_inkling2_moe_expert_offload(offloaded)


def test_switch_off_restores_install_path(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "PREFILL_NO_EVICT", False)
    stock = _build_tiny_model((_format_spec("nvfp4"),) * 6, num_experts=16, top_k=2)
    _diverse(stock, "nvfp4")
    _save_runtime_checkpoint(tmp_path, stock)
    offloaded = _reload_inkling2(tmp_path, lazy=True)
    assert apply_inkling2_moe_expert_offload(offloaded, tmp_path, 0.25) == 6
    mx.eval(offloaded.parameters())
    materialize_offload_state(offloaded)
    caches = [layer.mlp.cache for layer in offloaded.model.layers]
    try:
        mx.random.seed(5)
        for _ in range(6):
            mx.eval(offloaded(mx.random.randint(0, 300, (1, 1))))
        before = [sorted(c.slot_of) for c in caches]
        mx.random.seed(77)
        mx.eval(offloaded(mx.random.randint(0, 300, (1, 48))))
        after = [sorted(c.slot_of) for c in caches]
        assert after != before  # the 0.1.0 path installs prefill experts
    finally:
        close_inkling2_moe_expert_offload(offloaded)
