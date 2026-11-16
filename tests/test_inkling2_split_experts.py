# SPDX-License-Identifier: MIT
"""Tests for the separable routed-expert stage in Inkling 2."""

from __future__ import annotations

import gc
import weakref

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from omlx.patches.inkling2 import model as inkling2
from tests.test_inkling_vlm_mtp import _inkling2_tiny


def _frozen_experts_call(
    experts: inkling2.Inkling2Experts,
    x: mx.array,
    inds: mx.array,
    weights: mx.array,
) -> mx.array:
    """Frozen pre-split Inkling2Experts.__call__ body used as reference."""
    xe = mx.expand_dims(x, (-2, -3))
    do_sort = inds.size >= 64
    idx = inds
    inv = None
    if do_sort:
        xe, idx, inv = inkling2._gather_sort(xe, inds)
    y = experts.w13(xe, idx, do_sort)
    s13 = experts.w13.post_scale(idx)
    if s13 is not None:
        y = y * s13[..., None, None].astype(y.dtype)
    act = inkling2._softcapped_swiglu(
        y[..., 0::2], y[..., 1::2], experts._gcap, experts._ucap
    )
    z = experts.w2(act, idx, do_sort)
    if do_sort:
        z = inkling2._scatter_unsort(z, inv, inds.shape)
    z = z.squeeze(-2)
    s2 = experts.w2.post_scale(inds)
    if s2 is not None:
        weights = weights * s2
    out = mx.sum(z.astype(mx.float32) * weights[..., None], axis=-2)
    return out.astype(x.dtype)


def _make_experts(fmt: str, *, num_experts: int = 8, top_k: int = 2, seed: int = 1):
    spec = (
        {"mode": "affine", "group_size": 64, "bits": 4}
        if fmt == "affine"
        else inkling2.expert_quant_spec(fmt)
    )
    args = inkling2.TextArgs(
        hidden_size=128,
        intermediate_size=64,
        n_routed_experts=num_experts,
        num_experts_per_tok=top_k,
    )
    mx.random.seed(seed)
    experts = inkling2.Inkling2Experts(args, spec, spec)
    for proj in (experts.w13, experts.w2):
        raw = mx.random.randint(
            0, 0x70, (*proj.weight.shape[:-1], proj.weight.shape[-1] * 4)
        ).astype(mx.uint8)
        proj.weight = raw.view(mx.uint32)
        if proj.mode == "affine":
            proj.scales = mx.random.uniform(0.002, 0.02, proj.scales.shape).astype(
                mx.bfloat16
            )
            proj.biases = (mx.random.normal(proj.biases.shape) * 0.02).astype(
                mx.bfloat16
            )
        else:
            proj.scales = mx.full(proj.scales.shape, 120, dtype=mx.uint8)
            if proj.mode == "nvfp4":
                proj.global_scale = mx.random.uniform(0.5, 8.0, (num_experts,))
    mx.eval(experts.parameters())
    return experts, args


def _randomize_global_scales(model: inkling2.Model, seed: int = 99) -> list:
    mx.random.seed(seed)
    updates = [
        (name, mx.random.uniform(0.5, 8.0, value.shape))
        for name, value in tree_flatten(model.parameters())
        if name.endswith("global_scale")
    ]
    if updates:
        model.load_weights(updates, strict=False)
    return updates


def _cast_model_bfloat16(model: inkling2.Model) -> None:
    pred = model.cast_predicate
    model.load_weights(
        [
            (k, v.astype(mx.bfloat16) if pred(k) and v.dtype == mx.float32 else v)
            for k, v in tree_flatten(model.parameters())
        ]
    )


def _build_mixed_pair(
    formats=("mxfp8", "nvfp4", {"mode": "affine", "group_size": 64, "bits": 4}) * 4,
    *,
    cast_bf16: bool = False,
    **dims,
) -> tuple[inkling2.Model, inkling2.Model]:
    stock = _inkling2_tiny(formats=formats, **dims)
    split = _inkling2_tiny(formats=formats, **dims)
    gs_updates = _randomize_global_scales(stock)
    if gs_updates:
        split.load_weights(gs_updates, strict=False)
    if cast_bf16:
        _cast_model_bfloat16(stock)
        _cast_model_bfloat16(split)
    mx.eval(stock.parameters(), split.parameters())
    return stock, split


@pytest.mark.parametrize("fmt", ["mxfp8", "nvfp4", "affine"])
@pytest.mark.parametrize(
    "tokens,top_k", [(1, 2), (4, 2), (40, 1), (40, 2), (80, 1), (80, 2)]
)
def test_experts_routes_and_combine_match_frozen_reference(fmt, tokens, top_k):
    experts, args = _make_experts(fmt, num_experts=8, top_k=top_k, seed=7)
    mx.random.seed(10 + tokens + top_k)
    x = (mx.random.normal((tokens, args.hidden_size)) * 0.5).astype(mx.bfloat16)
    inds = mx.stack(
        [mx.random.permutation(args.n_routed_experts)[:top_k] for _ in range(tokens)]
    ).astype(mx.uint32)
    weights = mx.random.uniform(0.1, 1.0, (tokens, top_k))

    z = experts.routes(x, inds)
    assert z.shape == (tokens, top_k, args.hidden_size)
    assert z.dtype == x.dtype

    split_out = experts.combine(z, inds, weights, x.dtype)
    call_out = experts(x, inds, weights)
    ref_out = _frozen_experts_call(experts, x, inds, weights)
    mx.eval(split_out, call_out, ref_out)

    assert mx.array_equal(split_out, ref_out)
    assert mx.array_equal(call_out, ref_out)
    assert mx.abs(ref_out.astype(mx.float32)).max().item() > 0.0

    if fmt == "nvfp4":
        # Confirm combine genuinely reads w2.global_scale rather than ignoring it.
        orig_gs = experts.w2.global_scale
        experts.w2.global_scale = orig_gs * 2.0
        scaled_out = experts.combine(z, inds, weights, x.dtype)
        mx.eval(scaled_out)
        assert not mx.array_equal(scaled_out, split_out)


@pytest.mark.parametrize("fmt", ["mxfp8", "nvfp4", "affine"])
def test_moe_route_inputs_and_experts_out_match_reference(fmt):
    spec = (
        {"mode": "affine", "group_size": 64, "bits": 4}
        if fmt == "affine"
        else inkling2.expert_quant_spec(fmt)
    )
    args = inkling2.TextArgs(
        hidden_size=128,
        intermediate_size=64,
        n_routed_experts=8,
        num_experts_per_tok=2,
    )
    moe = inkling2.Inkling2MoE(args, spec, spec)
    experts, _ = _make_experts(fmt, num_experts=8, top_k=2, seed=11)
    moe.experts = experts
    mx.random.seed(12)
    moe.router_weight = mx.random.normal(moe.router_weight.shape) * 0.1
    moe.router_bias = mx.random.normal(moe.router_bias.shape) * 0.05
    mx.eval(moe.parameters())

    for shape in ((1, 1, 128), (2, 4, 128), (1, 40, 128)):
        x = (mx.random.normal(shape) * 0.5).astype(mx.bfloat16)
        flat, inds, weights, orig_shape = moe.route_inputs(x)
        assert orig_shape == shape
        assert flat.shape == (shape[0] * shape[1], shape[2])
        assert inds.shape == (shape[0] * shape[1], 2)
        assert weights.shape == (shape[0] * shape[1], 2)

        got = moe.experts_out(flat, inds, weights, orig_shape)
        ref = _frozen_experts_call(
            moe.experts, x.reshape(-1, shape[-1]), *moe.route(x.reshape(-1, shape[-1]))
        ).reshape(shape)
        mx.eval(got, ref)
        assert mx.array_equal(got, ref)

        seen = []

        def custom_fn(fx, i, w, log=seen):
            log.append((fx.shape, i.shape, w.shape))
            return moe.experts(fx, i, w)

        custom_out = moe.experts_out(flat, inds, weights, orig_shape, experts=custom_fn)
        mx.eval(custom_out)
        assert len(seen) == 1
        assert mx.array_equal(custom_out, ref)


@pytest.mark.parametrize("cast_bf16", [False, True])
def test_split_layer_and_model_eager_match_stock_eager(cast_bf16):
    stock, split = _build_mixed_pair(cast_bf16=cast_bf16)

    # Layer-level check across boundary+sliding (0), non-boundary+sliding (1),
    # and full-attention (5, affine).
    h_dtype = mx.bfloat16 if cast_bf16 else mx.float32
    for layer_idx in (0, 1, 5):
        stock_layer = stock.layers[layer_idx]
        split_layer = split.layers[layer_idx]
        mx.random.seed(50 + layer_idx)
        length = 40  # > 16 (eager) and 40*2=80 >= 64 (sorted)
        h = (mx.random.normal((1, length, 128)) * 0.2).astype(h_dtype)
        n_blocks = max(0, layer_idx // 6)
        blocks_stock = [
            inkling2._unit_rms(mx.random.normal((1, length, 128)))
            for _ in range(n_blocks)
        ]
        blocks_split = [mx.array(b) for b in blocks_stock]
        cache_stock = stock.make_cache()[layer_idx]
        cache_split = split.make_cache()[layer_idx]

        called = []

        def wrap_experts(fn, store=called):
            def inner(fx, inds, w):
                mx.eval(inds)
                store.append(len(inds))
                return fn(fx, inds, w)

            return inner

        split_layer._split_experts = wrap_experts(split_layer.mlp.experts)
        out_stock = stock_layer(h, blocks_stock, cache=cache_stock)
        out_split = split_layer(h, blocks_split, cache=cache_split)
        mx.eval(out_stock, out_split)

        assert called == [length]
        assert mx.array_equal(out_split, out_stock)
        assert len(blocks_split) == len(blocks_stock)
        for bs, bp in zip(blocks_stock, blocks_split):
            assert mx.array_equal(bp, bs)
        for slot in range(4):
            assert mx.array_equal(cache_split[1][slot], cache_stock[1][slot])
        k_s, v_s = cache_stock[0].keys_and_values()
        k_p, v_p = cache_split[0].keys_and_values()
        assert cache_split[0].offset == cache_stock[0].offset
        assert mx.array_equal(k_p, k_s)
        assert mx.array_equal(v_p, v_s)

    # Full-model eager check at T = 24 (unsorted routes) and T = 40 (sorted routes).
    total_calls = [0]
    for layer in split.layers:
        exp = layer.mlp.experts

        def make_fn(e):
            def fn(fx, inds, w):
                total_calls[0] += 1
                mx.eval(inds)
                return e(fx, inds, w)

            return fn

        layer._split_experts = make_fn(exp)

    for length in (24, 40):
        mx.random.seed(200 + length)
        ids = mx.random.randint(0, 300, (1, length))
        cs = stock.make_cache()
        cp = split.make_cache()
        total_calls[0] = 0
        logits_stock = stock(ids, cache=cs)[..., :300]
        logits_split = split(ids, cache=cp)[..., :300]
        mx.eval(logits_stock, logits_split)
        assert total_calls[0] == len(split.layers)
        assert mx.array_equal(logits_split, logits_stock)

    # Also verify cache=None works cleanly on the split path.
    total_calls[0] = 0
    logits_stock_nc = stock(ids)[..., :300]
    logits_split_nc = split(ids)[..., :300]
    mx.eval(logits_stock_nc, logits_split_nc)
    assert total_calls[0] == len(split.layers)
    assert mx.array_equal(logits_split_nc, logits_stock_nc)


@pytest.mark.parametrize(
    "cast_bf16,num_experts,top_k",
    [
        (False, 8, 2),
        (True, 32, 4),
    ],
)
def test_split_compiled_decode_and_verify_match_fused_stock(
    cast_bf16, num_experts, top_k
):
    stock, split = _build_mixed_pair(
        cast_bf16=cast_bf16,
        n_routed_experts=num_experts,
        num_experts_per_tok=top_k,
    )
    expert_calls = [0]
    for layer in split.layers:
        exp = layer.mlp.experts

        def make_fn(e):
            def fn(fx, inds, w):
                expert_calls[0] += 1
                # Calling eval/tolist inside stage B verifies it runs eagerly
                # outside the compiled _post_attn / _post_tail stages.
                mx.eval(inds)
                _ = inds.tolist()
                return e(fx, inds, w)

            return fn

        layer._split_experts = make_fn(exp)

    mx.random.seed(303)
    ids = mx.random.randint(0, 300, (1, 40))

    cs = stock.make_cache()
    cp = split.make_cache()
    stock(ids[:, :18], cache=cs)
    split(ids[:, :18], cache=cp)
    expert_calls[0] = 0

    max_decode_diff = 0.0
    for t in range(18, 23):
        ls = stock(ids[:, t : t + 1], cache=cs)[..., :300]
        lp = split(ids[:, t : t + 1], cache=cp)[..., :300]
        mx.eval(ls, lp)
        diff = mx.abs(lp - ls).max().item()
        max_decode_diff = max(max_decode_diff, diff)
    assert expert_calls[0] == 5 * len(split.layers)
    assert max_decode_diff <= 5e-4, f"decode max |delta logit|={max_decode_diff}"

    max_verify_diff = 0.0
    for verify_len in (4, 8, 12, 16):
        cs2 = stock.make_cache()
        cp2 = split.make_cache()
        stock(ids[:, :18], cache=cs2)
        split(ids[:, :18], cache=cp2)
        expert_calls[0] = 0
        ls = stock(ids[:, 18 : 18 + verify_len], cache=cs2)[..., :300]
        lp = split(ids[:, 18 : 18 + verify_len], cache=cp2)[..., :300]
        mx.eval(ls, lp)
        assert expert_calls[0] == len(split.layers)
        diff = mx.abs(lp - ls).max().item()
        max_verify_diff = max(max_verify_diff, diff)
    assert max_verify_diff <= 5e-4, f"verify max |delta logit|={max_verify_diff}"


def test_split_verify_rollback_matches_sequential_split_decode():
    stock, split = _build_mixed_pair()
    for layer in split.layers:
        layer._split_experts = layer.mlp.experts

    mx.random.seed(404)
    ids = mx.random.randint(0, 300, (1, 30))

    seq_cache = split.make_cache()
    split(ids[:, :20], cache=seq_cache)
    for t in range(20, 24):
        split(ids[:, t : t + 1], cache=seq_cache)
    expected = split(ids[:, 24:25], cache=seq_cache)[..., :300]

    rb_cache = split.make_cache()
    split(ids[:, :20], cache=rb_cache)
    split(ids[:, 20:28], cache=rb_cache, return_hidden=True, capture_layer_ids=[5, 12])
    assert split.mtp_partial_rollback(rb_cache, 3, 7)
    got = split(ids[:, 24:25], cache=rb_cache)[..., :300]

    stock_cache = stock.make_cache()
    stock(ids[:, :20], cache=stock_cache)
    stock(ids[:, 20:28], cache=stock_cache, return_hidden=True)
    assert stock.mtp_partial_rollback(stock_cache, 3, 7)
    stock_got = stock(ids[:, 24:25], cache=stock_cache)[..., :300]

    mx.eval(expected, got, stock_got)
    assert rb_cache[0][0].offset == seq_cache[0][0].offset == 25
    assert mx.abs(got - expected).max().item() < 1e-4
    assert mx.abs(got - stock_got).max().item() <= 5e-4


def test_split_mode_toggle_restores_stock_exactness():
    fresh, toggled = _build_mixed_pair()
    assert all(layer._split_experts is None for layer in toggled.layers)
    fresh_param_names = [k for k, _ in tree_flatten(fresh.parameters())]

    for layer in toggled.layers:
        layer._split_experts = layer.mlp.experts
    assert [k for k, _ in tree_flatten(toggled.parameters())] == fresh_param_names

    # Exercise compiled and eager split paths before toggling back off.
    mx.random.seed(505)
    warmup = mx.random.randint(0, 300, (1, 28))
    wc = toggled.make_cache()
    mx.eval(
        toggled(warmup[:, :18], cache=wc),
        toggled(warmup[:, 18:19], cache=wc),
        toggled(warmup[:, 19:27], cache=wc),
    )

    for layer in toggled.layers:
        layer._split_experts = None
    assert [k for k, _ in tree_flatten(toggled.parameters())] == fresh_param_names

    ids = mx.random.randint(0, 300, (1, 32))
    cf = fresh.make_cache()
    ct = toggled.make_cache()
    checks = [
        (fresh(ids[:, :20], cache=cf), toggled(ids[:, :20], cache=ct)),
        (fresh(ids[:, 20:21], cache=cf), toggled(ids[:, 20:21], cache=ct)),
        (fresh(ids[:, 21:22], cache=cf), toggled(ids[:, 21:22], cache=ct)),
        (fresh(ids[:, 22:30], cache=cf), toggled(ids[:, 22:30], cache=ct)),
    ]
    for a, b in checks:
        mx.eval(a, b)
        assert mx.array_equal(a, b)

    for entry_f, entry_t in zip(cf, ct):
        kf, vf = entry_f[0].keys_and_values()
        kt, vt = entry_t[0].keys_and_values()
        assert entry_f[0].offset == entry_t[0].offset
        assert mx.array_equal(kf, kt)
        assert mx.array_equal(vf, vt)
        for slot in range(4):
            assert mx.array_equal(entry_f[1][slot], entry_t[1][slot])


def test_split_mode_model_freed_without_cyclic_gc():
    model = _inkling2_tiny()
    for layer in model.layers:
        layer._split_experts = layer.mlp.experts
    ids = mx.random.randint(0, 300, (1, 4))
    mx.eval(model(ids, cache=model.make_cache()))
    ref = weakref.ref(model.model.layers[3])
    gc.disable()
    try:
        del model
        assert ref() is None
    finally:
        gc.enable()
