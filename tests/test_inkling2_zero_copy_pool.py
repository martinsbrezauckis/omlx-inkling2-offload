# SPDX-License-Identifier: MIT
"""Cross-call direct buffers must outlive readers and preserve evaluated routes."""

import gc
import logging
import os
import subprocess
import sys
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.inkling2 import moe_offload as mo
from omlx.patches.moe_expert_offload import _ReadPlan, materialize_offload_state
from tests import test_inkling2_zero_copy as zero_copy_tests
from tests.test_inkling2_moe_offload import (
    _build_tiny_model,
    _format_spec,
    _reload_inkling2,
    _save_runtime_checkpoint,
)
from tests.test_inkling2_prefill_no_evict import _diverse
from tests.test_inkling2_zero_copy import bits, prefill

model_cache = zero_copy_tests.model_cache


@pytest.fixture(autouse=True)
def clean_pool(monkeypatch):
    # Allow behavioral regressions to run against the per-call allocation path, which has no pool.
    monkeypatch.setattr(mo, "ZERO_COPY_POOL_BYTES", 2560 << 20, raising=False)
    pool = getattr(mo, "_prefill_buffers", None)
    if pool is not None:
        pool.clear()
    yield
    if pool is not None:
        pool.clear()
    mo._T.clear()


def track_allocations(monkeypatch):
    allocated = []
    original = mo._PrefillBuffer

    def allocate(*args):
        buffer = original(*args)
        # Keep allocations alive so the allocator cannot fake address reuse.
        allocated.append(buffer)
        return buffer

    monkeypatch.setattr(mo, "_PrefillBuffer", allocate)
    return allocated


def addresses(buffer):
    return tuple(
        np.frombuffer(host, np.uint8).__array_interface__["data"][0]
        for host in buffer.host.values()
    )


def test_many_layers_reuse_with_lazy_final_outputs(model_cache, monkeypatch):
    model, _ = model_cache
    caches = [layer.mlp.cache for layer in model.model.layers]
    monkeypatch.setattr(mo, "PREFILL_GROUP", 3)
    monkeypatch.setattr(mo, "_io_pool", lambda: None)
    for cache in caches:
        cache.ensure_ids(range(3))
        mx.eval(cache.experts.parameters())
    monkeypatch.setattr(mo, "ZERO_COPY", False)
    expected = [bits(prefill(cache)) for cache in caches]
    monkeypatch.setattr(mo, "ZERO_COPY", True)
    monkeypatch.setattr(mo, "ZERO_COPY_POOL_BYTES", 0)
    assert [bits(prefill(cache)) for cache in caches] == expected
    allocated = track_allocations(monkeypatch)
    monkeypatch.setattr(mo, "ZERO_COPY_POOL_BYTES", 2560 << 20)
    monkeypatch.setattr(mo, "TIMING", True)
    mo._T.clear()
    # All final concatenate/take/combine graphs stay lazy while different
    # layers overwrite both full groups and odd tails of the same buffer.
    outputs = [prefill(cache) for _ in range(4) for cache in caches]
    assert [bits(output) for output in outputs] == expected * 4
    assert len(allocated) == 1
    assert mo._T["em_buffer_allocations"] == 1
    assert mo._T["em_buffer_reuses"] == 23
    elapsed = mo._T["em_buffer"]
    assert bits(prefill(caches[0])) == expected[0]
    assert mo._T["em_buffer"] == elapsed


def test_two_pipeline_buffers_reused_and_disabled_is_s20(model_cache, monkeypatch):
    _, cache = model_cache
    monkeypatch.setattr(mo, "PREFILL_GROUP", 3)
    monkeypatch.setattr(mo, "ZERO_COPY", True)
    allocated = track_allocations(monkeypatch)
    before = bits(prefill(cache))
    first_addresses = [addresses(buffer) for buffer in allocated]
    assert len(first_addresses) == 2
    assert bits(prefill(cache)) == before
    assert [addresses(buffer) for buffer in allocated] == first_addresses
    monkeypatch.setattr(mo, "ZERO_COPY_POOL_BYTES", 0)
    for _ in range(3):
        assert bits(prefill(cache)) == before
    assert len(allocated) == 8


@pytest.mark.parametrize("close", ["cache", "model"])
def test_close_releases_buffers(model_cache, monkeypatch, close):
    model, cache = model_cache
    monkeypatch.setattr(mo, "_io_pool", lambda: None)
    refs = []
    original = mo._PrefillBuffer

    def allocate(*args):
        buffer = original(*args)
        refs.append(weakref.ref(buffer))
        return buffer

    monkeypatch.setattr(mo, "_PrefillBuffer", allocate)
    bits(prefill(cache))
    assert refs and all(ref() is not None for ref in refs)
    if close == "cache":
        cache.close()
    else:
        mo.close_inkling2_moe_expert_offload(model)
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert mo._prefill_buffers.nbytes == 0


def layout(*, shape=(4,), dtype=np.uint32, view=None, source=False, field="weight"):
    cache = SimpleNamespace(view=SimpleNamespace(source_layout=source))
    plan = _ReadPlan(
        -1, 0, int(np.prod(shape)) * np.dtype(dtype).itemsize, dtype, view, shape
    )
    return cache, [(("w13", field), plan)]


@pytest.mark.parametrize(
    "change", ["shape", "dtype", "view", "source", "field", "count"]
)
def test_pool_key_captures_every_layout_difference(change):
    pool = mo._prefill_buffers
    cache, plans = layout()
    first = pool.acquire(cache, 3, plans)
    pool.release(first)
    variants = {
        "shape": {"shape": (2, 2)},
        "dtype": {"dtype": np.int32},
        "view": {"view": mx.float32},
        "source": {"source": True},
        "field": {"field": "scales"},
        "count": {},
    }
    cache2, plans2 = layout(**variants[change])
    second = pool.acquire(cache2, 2 if change == "count" else 3, plans2)
    assert second is not first
    pool.release(second)
    assert pool.acquire(cache, 3, plans) is first
    pool.release(first)


def test_byte_cap_counts_leases_and_evicts_idle_buffers(monkeypatch):
    pool = mo._prefill_buffers
    monkeypatch.setattr(mo, "ZERO_COPY_POOL_BYTES", 96)
    cache, plans = layout()
    a = pool.acquire(cache, 3, plans)
    b = pool.acquire(cache, 3, plans)
    spill = pool.acquire(cache, 3, plans)
    assert len({addresses(a), addresses(b), addresses(spill)}) == 3
    assert pool.nbytes == 96
    for buffer in (a, b, spill):
        pool.release(buffer)
    assert pool.nbytes == 96
    other_cache, other_plans = layout(dtype=np.float32)
    other = pool.acquire(other_cache, 5, other_plans)
    assert pool.nbytes == 80
    pool.release(other)
    too_large = pool.acquire(other_cache, 7, other_plans)
    pool.release(too_large)
    assert pool.nbytes <= 96


def test_two_per_key_even_with_large_byte_budget():
    pool = mo._prefill_buffers
    cache, plans = layout()
    buffers = [pool.acquire(cache, 3, plans) for _ in range(5)]
    assert len({addresses(buffer) for buffer in buffers}) == 5
    assert pool.nbytes == 96
    for buffer in buffers:
        pool.release(buffer)
    assert pool.nbytes == 96


def test_clear_does_not_readmit_outstanding_leases():
    pool = mo._prefill_buffers
    cache, plans = layout()
    before = pool.acquire(cache, 3, plans)
    pool.clear()
    after = pool.acquire(cache, 3, plans)
    pool.release(before)
    pool.release(after)
    assert pool.nbytes == 48
    assert pool.acquire(cache, 3, plans) is after
    pool.release(after)


def test_abandoned_generator_does_not_reuse_lazy_consumer(model_cache, monkeypatch):
    _, cache = model_cache
    monkeypatch.setattr(mo, "_io_pool", lambda: None)
    monkeypatch.setattr(mo, "PREFILL_GROUP", 3)
    allocated = track_allocations(monkeypatch)
    direct = mo._zero_copy_groups(cache, [[0, 1, 2]])
    fields = next(direct)
    weight = fields[("w13", "weight")]
    snapshot = mx.array(np.array(weight))
    lazy = mx.sum(weight.astype(mx.float32))
    direct.close()
    # Closing at a yield is an aborted evaluation, not a reuse barrier.
    bits(prefill(cache))
    assert len(allocated) == 2
    assert bits(lazy.reshape(1)) == bits(mx.sum(snapshot.astype(mx.float32)).reshape(1))
    assert mo._prefill_buffers.nbytes == 3 * cache.expert_bytes


def test_concurrent_calls_keep_buffers_exclusive(model_cache, monkeypatch):
    model, _ = model_cache
    caches = [layer.mlp.cache for layer in model.model.layers[:2]]
    monkeypatch.setattr(mo, "_io_pool", lambda: None)
    monkeypatch.setattr(mo, "PREFILL_GROUP", 3)
    expected = [bits(prefill(cache)) for cache in caches]
    mo._prefill_buffers.clear()
    allocated = track_allocations(monkeypatch)
    real_eval, real_read, real_routes = mx.eval, mo._read_into, mo._TempExperts.routes
    barrier, lock = threading.Barrier(2), threading.Lock()
    busy, seen = {}, set()
    width = caches[0].experts.w2.weight.shape[1]

    def routes(self, *args):
        ident = threading.get_ident()
        ranges = []
        for proj in (self.w13, self.w2):
            for field in mo._fields(proj):
                array = np.asarray(proj[field].view(mx.uint8))
                start = array.__array_interface__["data"][0]
                ranges.append((start, start + array.nbytes))
        with lock:
            busy[ident] = ranges
            first = ident not in seen
            seen.add(ident)
        if first:
            barrier.wait(timeout=15)
        return real_routes(self, *args)

    def read(plan, dest):
        address = np.frombuffer(dest, np.uint8).__array_interface__["data"][0]
        with lock:
            assert not any(
                lo <= address < hi for ranges in busy.values() for lo, hi in ranges
            )
        return real_read(plan, dest)

    def evaluate(*args):
        result = real_eval(*args)
        if any(
            isinstance(a, mx.array) and a.ndim == 2 and a.shape[-1] == width
            for a in args
        ):
            with lock:
                busy.pop(threading.get_ident(), None)
        return result

    def run(cache):
        try:
            with mx.stream(mx.new_stream(mx.gpu)):
                return bits(prefill(cache))
        finally:
            mx.synchronize()
            mx.clear_streams()

    monkeypatch.setattr(mo._TempExperts, "routes", routes)
    monkeypatch.setattr(mo, "_read_into", read)
    monkeypatch.setattr(mx, "eval", evaluate)
    with ThreadPoolExecutor(max_workers=2) as executor:
        assert list(executor.map(run, caches)) == expected
    assert len(allocated) == 2
    assert set(addresses(allocated[0])).isdisjoint(addresses(allocated[1]))
    monkeypatch.setattr(mo._TempExperts, "routes", real_routes)
    assert bits(prefill(caches[0])) == expected[0]
    assert len(allocated) == 2


@pytest.mark.parametrize("hook", ["pressure", "headroom"])
def test_scheduler_pressure_releases_live_pool(monkeypatch, hook):
    from omlx.scheduler import Scheduler

    pool = mo._prefill_buffers
    cache, plans = layout()
    buffer = pool.acquire(cache, 3, plans)
    pool.release(buffer)
    scheduler = SimpleNamespace(
        _pending_pressure_clear=True, _current_usage_bytes=lambda: 123
    )
    if hook == "pressure":
        assert Scheduler._consume_pressure_clear(scheduler)
    else:

        def clear(_):
            assert (
                pool.nbytes == 0
            ), "live arrays must be released before MLX cache clear"

        monkeypatch.setattr(Scheduler, "_clear_cache", clear)
        assert Scheduler._reclaim_prefill_headroom(scheduler) == 123
    assert pool.nbytes == 0


@pytest.mark.parametrize(
    "setting, expected", [(None, 2560 << 20), ("0", 0), ("1.5", 1572864)]
)
def test_pool_env(setting, expected):
    env = dict(os.environ)
    env.pop("OMLX_INKLING2_ZERO_COPY_POOL_MB", None)
    if setting is not None:
        env["OMLX_INKLING2_ZERO_COPY_POOL_MB"] = setting
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from omlx.patches.inkling2.moe_offload "
            "import ZERO_COPY_POOL_BYTES; print(ZERO_COPY_POOL_BYTES)",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == str(expected)


def test_mixed_layout_full_model_repeated_calls(tmp_path, monkeypatch):
    formats = tuple(_format_spec(fmt) for fmt in ("nvfp4", "affine", "mxfp8") * 2)
    stock = _build_tiny_model(formats, num_experts=16, top_k=2)
    _diverse(stock, "nvfp4")
    _save_runtime_checkpoint(tmp_path, stock)
    model = _reload_inkling2(tmp_path, lazy=True)
    mo.apply_inkling2_moe_expert_offload(model, tmp_path, 0.25)
    mx.eval(model.parameters())
    materialize_offload_state(model)
    monkeypatch.setattr(mo, "PREFILL_GROUP", 3)
    monkeypatch.setattr(mo, "_io_pool", lambda: None)
    tokens = [
        mx.arange(offset, offset + 48, dtype=mx.int32)[None] for offset in range(4)
    ]
    try:
        monkeypatch.setattr(mo, "ZERO_COPY", False)
        expected = [bits(model(ids)) for ids in tokens]
        monkeypatch.setattr(mo, "ZERO_COPY", True)
        monkeypatch.setattr(mo, "ZERO_COPY_POOL_BYTES", 0)
        assert [bits(model(ids)) for ids in tokens] == expected
        monkeypatch.setattr(mo, "ZERO_COPY_POOL_BYTES", 2560 << 20)
        allocated = track_allocations(monkeypatch)
        outputs = [model(ids) for _ in range(3) for ids in tokens]
        assert [bits(output) for output in outputs] == expected * 3
        assert len(allocated) == 3
    finally:
        mo.close_inkling2_moe_expert_offload(model)


def test_model_unload_drops_pool_without_explicit_close(tmp_path, monkeypatch):
    stock = _build_tiny_model(("nvfp4",) * 6, num_experts=16, top_k=2)
    _save_runtime_checkpoint(tmp_path, stock)
    model = _reload_inkling2(tmp_path, lazy=True)
    mo.apply_inkling2_moe_expert_offload(model, tmp_path, 0.25)
    mx.eval(model.parameters())
    materialize_offload_state(model)
    cache = model.model.layers[0].mlp.cache
    monkeypatch.setattr(mo, "_io_pool", lambda: None)
    bits(prefill(cache))
    pool = mo._prefill_buffers
    assert pool.nbytes > 0
    refs = [weakref.ref(buffer) for buffer in pool._entries]
    store_ref = weakref.ref(cache.view.store)
    del cache, model
    gc.collect()
    assert store_ref() is None
    assert pool.nbytes == 0
    assert all(ref() is None for ref in refs)


def test_group_eval_failure_discards_buffers(model_cache, monkeypatch):
    _, cache = model_cache
    monkeypatch.setattr(mo, "_io_pool", lambda: None)
    monkeypatch.setattr(mo, "PREFILL_GROUP", 3)
    allocated = track_allocations(monkeypatch)
    original = mx.eval
    width = cache.experts.w2.weight.shape[1]
    consumers = []

    def fail(*args):
        for array in args:
            if (
                isinstance(array, mx.array)
                and array.ndim == 2
                and array.shape[-1] == width
            ):
                consumers.append(array)
                raise RuntimeError("injected group eval")
        return original(*args)

    monkeypatch.setattr(mx, "eval", fail)
    with pytest.raises(RuntimeError, match="injected group eval"):
        prefill(cache)
    monkeypatch.setattr(mx, "eval", original)
    before = [bytes(host) for host in allocated[0].host.values()]
    bits(prefill(cache))
    assert len(allocated) == 2
    assert [bytes(host) for host in allocated[0].host.values()] == before
    original(*consumers)
    assert mo._prefill_buffers.nbytes == 3 * cache.expert_bytes


def test_timing_log_reports_allocations_and_reuses(monkeypatch, caplog):
    mo._T.update(em_buffer_allocations=2, em_buffer_reuses=10)
    monkeypatch.setattr(mo, "TIMING_INTERVAL_S", 0)
    with caplog.at_level(logging.INFO, logger=mo.logger.name):
        mo._tflush()
    assert "alloc=2 reuse=10" in caplog.text
