# SPDX-License-Identifier: MIT
"""Decode read/compute overlap preserves per-route arithmetic and slot lifetimes."""

import logging
import os
import subprocess
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.inkling2 import moe_offload as mo
from omlx.patches.moe_expert_offload import materialize_offload_state
from tests.test_inkling2_moe_offload import (
    _build_tiny_model,
    _format_spec,
    _reload_inkling2,
    _save_runtime_checkpoint,
)


@pytest.fixture
def pair(tmp_path):
    models = []

    def make(fmt="nvfp4", dtype=mx.bfloat16, capacity=4):
        stock = _build_tiny_model(
            (_format_spec(fmt),),
            num_experts=16,
            top_k=4,
            cast_bf16=dtype == mx.bfloat16,
        )
        # Quantize nonzero finite weights in every format; exercise scales too.
        mx.random.seed(71)
        for name in ("w13", "w2"):
            proj = getattr(stock.model.layers[0].mlp.experts, name)
            shape = (*proj.weight.shape[:2], proj.scales.shape[-1] * proj.group_size)
            q = mx.quantize(
                (mx.random.normal(shape) * 0.1).astype(dtype),
                group_size=proj.group_size,
                bits=proj.bits,
                mode=proj.mode,
            )
            proj.weight, proj.scales = q[:2]
            if len(q) == 3:
                proj.biases = q[2]
        mx.eval(stock.parameters())
        _save_runtime_checkpoint(tmp_path, stock)
        experts = []
        for _ in range(2):
            model = _reload_inkling2(tmp_path, lazy=True)
            models.append(model)
            assert (
                mo.apply_inkling2_moe_expert_offload(model, tmp_path, capacity / 16)
                == 1
            )
            mx.eval(model.parameters())
            materialize_offload_state(model)
            experts.append(model.model.layers[0]._split_experts)
        return experts

    yield make
    for model in models:
        mo.close_inkling2_moe_expert_offload(model)


def _inputs(ids, dtype=mx.bfloat16):
    inds = mx.array(ids, dtype=mx.uint32)
    mx.random.seed(19)
    x = (mx.random.normal((*inds.shape[:-1], 128)) * 0.4).astype(dtype)
    w = mx.softmax(mx.random.normal(inds.shape), axis=-1)
    mx.eval(x, inds, w)
    return x, inds, w


def _bits(a):
    # array_equal alone does not distinguish signed zero.
    assert bool(mx.all(mx.isfinite(a)))
    return np.array(a.view(mx.uint8)).tobytes()


def _timing(monkeypatch):
    monkeypatch.setattr(mo, "TIMING", True)
    monkeypatch.setattr(mo, "_T", {})
    monkeypatch.setattr(mo, "TIMING_INTERVAL_S", float("inf"))


@pytest.mark.parametrize("fmt", ["affine", "nvfp4", "mxfp8"])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize(
    "warm",
    [[], [3, 1], [3, 1, 0, 2], [3, 9, 1, 10]],
    ids=["all_misses", "some_misses", "no_misses", "eviction"],
)
def test_decode_overlap_bit_identity(pair, monkeypatch, fmt, dtype, warm):
    old, new = pair(fmt, dtype)
    for expert in (old, new):
        expert.cache.ensure_ids(warm)
        mx.eval(expert.cache.experts.parameters(), expert.cache.map)
    args = _inputs([[2, 3, 0, 1]], dtype)
    monkeypatch.setattr(mo, "DECODE_OVERLAP", False, raising=False)
    ref = old(*args)
    mx.eval(ref)
    _timing(monkeypatch)
    monkeypatch.setattr(mo, "DECODE_OVERLAP", True, raising=False)
    got = new(*args)
    mx.eval(got)
    assert _bits(got) == _bits(ref)
    assert list(old.cache.slot_of) == list(new.cache.slot_of)
    assert (old.cache.hits, old.cache.misses, old.cache.fetched_bytes) == (
        new.cache.hits,
        new.cache.misses,
        new.cache.fetched_bytes,
    )
    assert mo._T.get("dec_overlap_calls", 0) == int(
        len(warm) > 0 and warm != [3, 1, 0, 2]
    )


@pytest.mark.parametrize("shape", [(1, 8), (2, 8), (1, 3, 8), (7, 8)])
@pytest.mark.parametrize("fmt", ["affine", "nvfp4", "mxfp8"])
def test_decode_overlap_repeated_routes_and_top8(pair, monkeypatch, shape, fmt):
    old, new = pair(fmt, capacity=8)
    ids = np.resize([2, 3, 0, 1, 6, 5, 4, 7], shape).tolist()
    for expert in (old, new):
        expert.cache.ensure_ids([3, 10, 1, 9, 5, 11, 7, 12])
        mx.eval(expert.cache.experts.parameters(), expert.cache.map)
    args = _inputs(ids)
    monkeypatch.setattr(mo, "DECODE_OVERLAP", False, raising=False)
    ref = old(*args)
    mx.eval(ref)
    _timing(monkeypatch)
    monkeypatch.setattr(mo, "DECODE_OVERLAP", True, raising=False)
    got = new(*args)
    mx.eval(got)
    assert _bits(got) == _bits(ref)
    assert mo._T.get("dec_overlap_calls", 0) == 1


def test_reads_overlap_resident_eval_and_ready_expert_runs_first(pair, monkeypatch):
    old, expert = pair()
    c = expert.cache
    c.ensure_ids([3, 9, 1, 10])
    mx.eval(c.experts.parameters(), c.map)
    args = _inputs([[2, 3, 0, 1]])
    old.cache.ensure_ids([3, 9, 1, 10])
    monkeypatch.setattr(mo, "DECODE_OVERLAP", False, raising=False)
    ref = old(*args)
    mx.eval(ref)
    monkeypatch.setattr(mo, "DECODE_OVERLAP", True, raising=False)
    events = []
    pending = {}
    read = mo.CheckpointExpertStore.read
    plans = {e: c._plans(e) for e in (2, 0)}
    expert_of = {plan: e for e, group in plans.items() for _, plan in group}

    class Pool:
        def submit(self, fn, plan):
            e = expert_of[plan]
            future = Future()
            pending[plan] = future
            events.append(("submit", e))
            return future

    def finish(e):
        for _, plan in plans[e]:
            pending[plan].set_result(read(plan))

    monkeypatch.setattr(mo, "_io_pool", lambda: Pool())
    routes = mo.Inkling2Experts.routes
    evaluate = mx.eval
    convert = c.view.to_slot_array

    def spy_routes(self, x, inds, sorted_hint=None):
        slots = inds.reshape(-1).tolist()
        routed = [int(c.slot_expert[s]) for s in slots]
        events.append(("routes", tuple(routed)))
        assert all(p in pending for group in plans.values() for _, p in group)
        if 0 in routed:
            # The second requested miss finishes first; it must compute without
            # waiting for expert 2, and its output must be evaluated before writes.
            assert not any(pending[p].done() for _, p in plans[2])
        return routes(self, x, inds, sorted_hint=sorted_hint)

    def spy_eval(*arrays):
        result = evaluate(*arrays)
        if events and events[-1] == ("routes", (3, 1)):
            events.append(("eval", "resident"))
            finish(0)
        elif events and events[-1] == ("routes", (0,)):
            events.append(("eval", "miss"))
            finish(2)
        return result

    def spy_convert(field, plan, raw):
        assert ("eval", "resident") in events, "slot write before resident eval"
        if expert_of[plan] == 2:
            assert ("eval", "miss") in events, "slot write before previous miss eval"
        assert {1, 3}.issubset(c.slot_of), "evicted a current resident"
        events.append(("write", expert_of[plan]))
        return convert(field, plan, raw)

    monkeypatch.setattr(mo.Inkling2Experts, "routes", spy_routes)
    monkeypatch.setattr(mx, "eval", spy_eval)
    monkeypatch.setattr(c.view, "to_slot_array", spy_convert)
    # Fail promptly on the old path instead of blocking on unfinished futures.
    monkeypatch.setattr(
        c, "ensure_ids", lambda *a, **kw: pytest.fail("serial ensure used")
    )
    got = expert(*args)
    evaluate(got)
    assert _bits(got) == _bits(ref)
    assert [v for kind, v in events if kind == "routes"] == [(3, 1), (0,), (2,)]
    assert list(c.slot_of) == [3, 1, 2, 0]
    assert all(f.done() for f in pending.values())


@pytest.mark.parametrize("window", [1, 2])
def test_read_window_refills_without_exceeding_bound(pair, monkeypatch, window):
    old, new = pair(capacity=8)
    for expert in (old, new):
        expert.cache.ensure_ids([1, 3, 5, 7, 9, 10, 11, 12])
        mx.eval(expert.cache.experts.parameters(), expert.cache.map)
    args = _inputs([[2, 3, 0, 1, 6, 5, 4, 7]])
    monkeypatch.setattr(mo, "DECODE_OVERLAP", False, raising=False)
    ref = old(*args)
    mx.eval(ref)
    c = new.cache
    active = set()
    maximum = 0
    plan_experts = {p: e for e in (2, 0, 6, 4) for _, p in c._plans(e)}

    class Pool:
        def submit(self, fn, plan):
            nonlocal maximum
            active.add(plan_experts[plan])
            maximum = max(maximum, len(active))
            assert len(active) <= window
            future = Future()
            future.set_result(fn(plan))
            return future

    convert = c.view.to_slot_array
    counts = {}

    def spy_convert(field, plan, raw):
        e = plan_experts[plan]
        counts[e] = counts.get(e, 0) + 1
        if counts[e] == len(c._writes):
            active.remove(e)
        return convert(field, plan, raw)

    monkeypatch.setattr(mo, "_io_pool", lambda: Pool())
    monkeypatch.setattr(c, "io_window", lambda: window)
    monkeypatch.setattr(c.view, "to_slot_array", spy_convert)
    monkeypatch.setattr(mo, "DECODE_OVERLAP", True, raising=False)
    _timing(monkeypatch)
    got = new(*args)
    mx.eval(got)
    assert _bits(got) == _bits(ref)
    assert maximum == window and not active
    assert mo._T["dec_overlap_calls"] == 1


def test_final_miss_batch_stays_lazy_until_combine(pair, monkeypatch):
    _, expert = pair()
    c = expert.cache
    c.ensure_ids([1, 3])
    mx.eval(c.experts.parameters(), c.map)
    args = _inputs([[2, 3, 0, 1]])

    class Pool:
        def submit(self, fn, plan):
            future = Future()
            future.set_result(fn(plan))
            return future

    evaluated = []
    evaluate = mx.eval

    def spy_eval(*arrays):
        evaluated.extend(arrays)
        evaluate(*arrays)

    monkeypatch.setattr(mo, "_io_pool", lambda: Pool())
    monkeypatch.setattr(mo, "DECODE_OVERLAP", True, raising=False)
    monkeypatch.setattr(mo, "TIMING", False)
    monkeypatch.setattr(mx, "eval", spy_eval)
    monkeypatch.setattr(
        c, "ensure_ids", lambda *a, **kw: pytest.fail("serial ensure used")
    )
    out = expert(*args)
    # Only the resident batch needs a barrier: no slot write follows the final
    # miss batch, so the caller can evaluate that batch together with combine.
    assert len(evaluated) == 1
    evaluate(out)


def test_partial_submission_failure_drains_started_reads(pair, monkeypatch):
    _, expert = pair()
    expert.cache.ensure_ids([1, 3])
    mx.eval(expert.cache.experts.parameters(), expert.cache.map)
    submitted = []

    class CancelFuture(Future):
        def cancel(self):
            cancelled = super().cancel()
            if cancelled:
                # A real executor acknowledges cancellation when dequeuing work.
                self.set_running_or_notify_cancel()
            return cancelled

    class Pool:
        def submit(self, fn, plan):
            if submitted:
                raise RuntimeError("forced submit failure")
            future = CancelFuture()
            submitted.append(future)
            return future

    monkeypatch.setattr(mo, "_io_pool", lambda: Pool())
    monkeypatch.setattr(mo, "DECODE_OVERLAP", True, raising=False)
    with pytest.raises(RuntimeError, match="forced submit failure"):
        expert(*_inputs([[2, 3, 0, 1]]))
    assert all(f.cancelled() for f in submitted)


def test_full_model_decode_logits_are_bit_identical(tmp_path, monkeypatch):
    stock = _build_tiny_model(
        ("nvfp4", "mxfp8", _format_spec("affine")), num_experts=16, top_k=4
    )
    for layer in stock.model.layers:
        layer.mlp.router_bias = mx.array([100.0] * 4 + [-100.0] * 12)
    _save_runtime_checkpoint(tmp_path, stock)
    models = [_reload_inkling2(tmp_path, lazy=True) for _ in range(2)]
    try:
        for model in models:
            assert mo.apply_inkling2_moe_expert_offload(model, tmp_path, 0.5) == 3
            mx.eval(model.parameters())
            materialize_offload_state(model)
        caches = [model.make_cache() for model in models]
        _timing(monkeypatch)
        for token in (3, 7, 11, 13):
            # The tiny model's routers select experts 0..3. Seed two resident
            # routes before every token, with other experts occupying all slots.
            for model in models:
                for layer in model.model.layers:
                    c = layer.mlp.cache
                    c.ensure_ids([1, 3, 8, 9, 10, 11, 12, 13])
                    mx.eval(c.experts.parameters(), c.map)
            outputs = []
            for enabled, model, cache in zip((False, True), models, caches):
                monkeypatch.setattr(mo, "DECODE_OVERLAP", enabled, raising=False)
                # The padded vocabulary tail is deliberately masked with -inf.
                out = model(mx.array([[token]]), cache=cache)[..., :300]
                mx.eval(out)
                outputs.append(out)
            assert _bits(outputs[0]) == _bits(outputs[1])
        assert mo._T.get("dec_overlap_calls", 0) > 0
    finally:
        for model in models:
            mo.close_inkling2_moe_expert_offload(model)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("warm", [[], [1, 3]], ids=["all_misses", "mixed"])
def test_read_exception_drains_current_and_other_experts(
    pair, monkeypatch, enabled, warm
):
    _, expert = pair()
    c = expert.cache
    c.ensure_ids(warm)
    mx.eval(c.experts.parameters(), c.map)
    args = _inputs([[2, 3, 0, 1]])
    monkeypatch.setattr(mo, "DECODE_OVERLAP", enabled, raising=False)
    futures = []
    release = threading.Event()
    started = threading.Event()
    pool = ThreadPoolExecutor(max_workers=1)

    class Pool:
        def submit(self, fn, plan):
            if not futures:
                future = Future()
                future.set_exception(OSError("forced slab failure"))
            else:

                def slow():
                    started.set()
                    assert release.wait(5), "drain never reached the running read"
                    return fn(plan)

                future = pool.submit(slow)
                assert started.wait(2)
            futures.append(future)
            return future

    drain = mo._drain

    def spy_drain(pending):
        retained = [f for group in pending.values() for _, _, f in group]
        assert set(futures).issubset(retained), "lost the failing expert's other reads"
        release.set()
        drain(pending)

    monkeypatch.setattr(mo, "_io_pool", lambda: Pool())
    monkeypatch.setattr(mo, "_drain", spy_drain)
    try:
        with pytest.raises(OSError, match="forced slab failure"):
            expert(*args)
        assert started.is_set()
        assert all(f.done() for f in futures)
    finally:
        release.set()
        pool.shutdown(wait=True)


@pytest.mark.parametrize("reason", ["env_off", "sorted", "no_pool", "warm"])
def test_fallback_uses_original_single_gather(pair, monkeypatch, reason):
    old, new = pair(capacity=16 if reason == "warm" else 4)
    for expert in (old, new):
        expert.cache.ensure_ids(list(range(16)) if reason == "warm" else [1, 3])
        mx.eval(expert.cache.experts.parameters(), expert.cache.map)
    args = _inputs([[2, 3, 0, 1]] * (16 if reason == "sorted" else 1))
    monkeypatch.setattr(mo, "DECODE_OVERLAP", False, raising=False)
    ref = old(*args)
    mx.eval(ref)
    monkeypatch.setattr(mo, "DECODE_OVERLAP", reason != "env_off", raising=False)
    if reason == "no_pool":
        monkeypatch.setattr(mo, "_io_pool", lambda: None)
    calls = []
    routes = mo.Inkling2Experts.routes

    def spy(self, x, inds, sorted_hint=None):
        calls.append((inds.shape, sorted_hint))
        return routes(self, x, inds, sorted_hint=sorted_hint)

    monkeypatch.setattr(mo.Inkling2Experts, "routes", spy)
    got = new(*args)
    mx.eval(got)
    assert _bits(got) == _bits(ref)
    assert calls == [
        (
            args[1].shape,
            True if reason == "sorted" else None if reason == "warm" else False,
        )
    ]


@pytest.mark.parametrize("setting, expected", [(None, True), ("0", False), ("1", True)])
def test_decode_overlap_environment(setting, expected):
    env = dict(os.environ)
    if setting is None:
        env.pop("OMLX_INKLING2_DECODE_OVERLAP", None)
    else:
        env["OMLX_INKLING2_DECODE_OVERLAP"] = setting
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from omlx.patches.inkling2.moe_offload import DECODE_OVERLAP; print(DECODE_OVERLAP)",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == str(expected)


def test_overlap_timing_counters_and_flush(pair, monkeypatch, caplog):
    _, expert = pair()
    expert.cache.ensure_ids([1, 3])
    mx.eval(expert.cache.experts.parameters(), expert.cache.map)
    args = _inputs([[2, 3, 0, 1]])
    _timing(monkeypatch)
    monkeypatch.setattr(mo, "DECODE_OVERLAP", True, raising=False)
    mx.eval(expert(*args))
    assert mo._T["dec_overlap_calls"] == 1
    assert mo._T["dec_calls"] == 1
    assert mo._T["dec_misses"] == 2
    for key in (
        "dec_overlap_res_eval",
        "dec_overlap_read_wait",
        "dec_overlap_miss_eval",
        "dec_overlap_install",
    ):
        assert mo._T[key] >= 0
    monkeypatch.setattr(mo, "TIMING_INTERVAL_S", 0)
    with caplog.at_level(logging.INFO, logger=mo.logger.name):
        mo._tflush()
    assert "overlap_calls=1" in caplog.text
    assert "res_eval=" in caplog.text and "miss_eval=" in caplog.text
    assert not mo._T
