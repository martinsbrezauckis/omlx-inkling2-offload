# SPDX-License-Identifier: MIT
"""Direct prefill reads preserve transport bits, compute groups, and lifetimes."""

import os
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event

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
    _save_source_checkpoint,
)
from tests.test_inkling2_prefill_no_evict import _diverse


@pytest.fixture(
    params=[
        ("affine", False),
        ("nvfp4", False),
        ("mxfp8", False),
        ("nvfp4", True),
        ("mxfp8", True),
    ]
)
def model_cache(request, tmp_path):
    fmt, source = request.param
    stock = _build_tiny_model((_format_spec(fmt),) * 6, num_experts=16, top_k=2)
    _diverse(stock, fmt)
    save = _save_source_checkpoint if source else _save_runtime_checkpoint
    save(tmp_path, stock)
    model = _reload_inkling2(tmp_path, lazy=True)
    assert mo.apply_inkling2_moe_expert_offload(model, tmp_path, 0.25) == 6
    mx.eval(model.parameters())
    materialize_offload_state(model)
    yield model, model.model.layers[0].mlp.cache
    mo.close_inkling2_moe_expert_offload(model)


def prefill(cache):
    ids = list(range(cache.n_experts)) * 12
    inds = mx.array(ids, dtype=mx.uint32).reshape(-1, 2)
    width = cache.experts.w2.weight.shape[1]
    x = (mx.arange(inds.shape[0] * width).reshape(-1, width) / 1000).astype(mx.bfloat16)
    weights = mx.full(inds.shape, 0.5, dtype=mx.float32)
    return mo.OffloadedInkling2Experts(cache)._forward_expert_major_no_evict(
        x, inds, weights, ids, 2, x.dtype
    )


def bits(value):
    mx.eval(value)
    return np.asarray(value.view(mx.uint8)).tobytes()


@pytest.mark.parametrize("group", [1, 3, 5, 32])
@pytest.mark.parametrize("resident", [0, 3])
def test_prefill_bits_and_no_conversion(model_cache, monkeypatch, group, resident):
    _, cache = model_cache
    cache.ensure_ids(range(resident))
    mx.eval(cache.experts.parameters())
    monkeypatch.setattr(mo, "PREFILL_GROUP", group)
    monkeypatch.setattr(mo, "ZERO_COPY", False)
    expected = bits(prefill(cache))
    before = dict(cache.slot_of)
    fetched = cache.fetched_bytes
    monkeypatch.setattr(mo, "ZERO_COPY", True)

    def forbidden(*args, **kwargs):
        pytest.fail("direct prefill used bytes, per-expert conversion, or stack")

    monkeypatch.setattr(mo.CheckpointExpertStore, "read", forbidden)
    monkeypatch.setattr(cache.view, "to_slot_array", forbidden)
    monkeypatch.setattr(mx, "stack", forbidden)
    assert bits(prefill(cache)) == expected
    assert cache.slot_of == before
    assert (
        cache.fetched_bytes - fetched
        == (cache.n_experts - resident) * cache.expert_bytes
    )


def test_full_model_prefill_bits(model_cache, monkeypatch):
    model, _ = model_cache
    monkeypatch.setattr(mo, "PREFILL_GROUP", 3)
    ids = mx.arange(48, dtype=mx.int32).reshape(1, -1)
    monkeypatch.setattr(mo, "ZERO_COPY", False)
    expected = bits(model(ids))
    monkeypatch.setattr(mo, "ZERO_COPY", True)
    assert bits(model(ids)) == expected


@pytest.mark.parametrize("setting, expected", [(None, True), ("0", False), ("1", True)])
def test_zero_copy_env(setting, expected):
    env = dict(os.environ)
    env.pop("OMLX_INKLING2_ZERO_COPY", None)
    if setting is not None:
        env["OMLX_INKLING2_ZERO_COPY"] = setting
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from omlx.patches.inkling2.moe_offload import ZERO_COPY; print(ZERO_COPY)",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == str(expected)


@pytest.mark.parametrize("failure", ["read", "submit", "routes"])
def test_direct_failure_drains_reads(model_cache, monkeypatch, failure):
    _, cache = model_cache
    monkeypatch.setattr(mo, "ZERO_COPY", True)
    monkeypatch.setattr(mo, "PREFILL_GROUP", 2)
    started, release = Event(), Event()
    futures, blocked = [], []
    real_read, real_wait = mo._read_into, mo.wait
    first = cache._plans(0)[0][1]
    # Hold the next group's read when failing during GPU graph construction.
    hold = cache._plans(2 if failure == "routes" else 0)[1][1]

    def read(plan, dest):
        if plan == hold:
            started.set()
            assert release.wait(10), "cleanup did not drain the read"
        if failure == "read" and plan == first:
            assert started.wait(10)
            raise OSError("injected read")
        return real_read(plan, dest)

    def drain(fs):
        fs = list(fs)
        if blocked and blocked[0] in fs:
            release.set()
        return real_wait(fs)

    with ThreadPoolExecutor(max_workers=24) as pool:

        class Reader:
            def submit(self, fn, plan, dest):
                if failure == "submit" and len(futures) == 2:
                    assert started.wait(10)
                    raise RuntimeError("injected submit")
                f = pool.submit(fn, plan, dest)
                futures.append(f)
                if plan == hold:
                    blocked.append(f)
                return f

        monkeypatch.setattr(mo, "_io_pool", lambda: Reader())
        monkeypatch.setattr(mo, "_read_into", read)
        monkeypatch.setattr(mo, "wait", drain)
        if failure == "routes":

            def routes(*args):
                assert started.wait(10)
                raise ValueError("injected routes")

            monkeypatch.setattr(mo._TempExperts, "routes", routes)
        try:
            with pytest.raises((OSError, RuntimeError, ValueError), match="injected"):
                prefill(cache)
            assert futures and all(f.done() for f in futures)
        finally:
            release.set()


def test_buffer_reuse_waits_for_output(model_cache, monkeypatch):
    _, cache = model_cache
    monkeypatch.setattr(mo, "ZERO_COPY", True)
    monkeypatch.setattr(mo, "PREFILL_GROUP", 3)
    real_read, real_routes, real_eval = mo._read_into, mo._TempExperts.routes, mx.eval
    in_use, groups, evaluations = [], [], []
    width = cache.experts.w2.weight.shape[1]

    def read(plan, dest):
        address = np.frombuffer(dest, dtype=np.uint8).__array_interface__["data"][0]
        assert not any(
            lo <= address < hi for lo, hi in in_use
        ), "CPU overwrote a buffer before its output evaluation"
        return real_read(plan, dest)

    def routes(self, x, inds):
        assert not in_use, "previous group was not evaluated"
        owners = set()
        for proj in (self.w13, self.w2):
            for field in mo._fields(proj):
                array = proj[field].view(mx.uint8)
                address = np.asarray(array).__array_interface__["data"][0]
                owners.add(address)
                in_use.append((address, address + array.nbytes))
        groups.append(owners)
        return real_routes(self, x, inds)

    def evaluate(*args):
        is_output = any(
            isinstance(a, mx.array) and a.ndim == 2 and a.shape[-1] == width
            for a in args
        )
        # Reads may run while eval blocks. Release ownership only after it
        # returns, and only for the group's output (never its input buffers).
        result = real_eval(*args)
        if is_output and in_use:
            evaluations.append(len(groups))
            in_use.clear()
        return result

    monkeypatch.setattr(mo, "_read_into", read)
    monkeypatch.setattr(mo._TempExperts, "routes", routes)
    monkeypatch.setattr(mx, "eval", evaluate)
    actual = bits(prefill(cache))
    assert len(groups) == 6
    assert groups[0] == groups[2] == groups[4]
    assert groups[1] == groups[3] == groups[5]
    assert groups[0].isdisjoint(groups[1])
    assert evaluations == list(range(1, 7))
    monkeypatch.setattr(mo, "ZERO_COPY", False)
    assert actual == bits(prefill(cache))


def test_timing_includes_buffer_preparation(monkeypatch, caplog):
    import logging

    mo._T.clear()
    mo._T["em_buffer"] = 1.25
    monkeypatch.setattr(mo, "TIMING_INTERVAL_S", 0)
    with caplog.at_level(logging.INFO, logger=mo.logger.name):
        mo._tflush()
    assert "buffer=1.2s" in caplog.text


@pytest.mark.parametrize("budget", [0, 3, 4, 6, 64])
def test_direct_byte_window(model_cache, monkeypatch, budget):
    _, cache = model_cache
    monkeypatch.setattr(mo, "ZERO_COPY", True)
    monkeypatch.setattr(mo, "PREFILL_GROUP", 3)
    monkeypatch.setattr(mo, "PREFILL_IO_BYTES", budget * cache.expert_bytes)
    monkeypatch.setattr(mo, "TIMING", True)
    mo._T.clear()
    outstanding = peak = 0
    submitted, owners = [], set()

    class Read(Future):
        def __init__(self, fn, plan, dest):
            super().__init__()
            self.plan, self.consumed = plan, False
            fn(plan, dest)
            self.set_result(None)

        def result(self, timeout=None):
            nonlocal outstanding
            if not self.consumed:
                self.consumed = True
                outstanding -= self.plan.nbytes
            return super().result(timeout)

    class Reader:
        def submit(self, fn, plan, dest):
            nonlocal outstanding, peak
            submitted.append(plan)
            owners.add(id(dest.obj))
            outstanding += plan.nbytes
            peak = max(peak, outstanding)
            return Read(fn, plan, dest)

    monkeypatch.setattr(mo, "_io_pool", lambda: Reader())
    monkeypatch.setattr(
        mo, "wait", lambda _: pytest.fail("successful reads should not be drained")
    )
    actual = bits(prefill(cache))
    assert outstanding == 0
    assert peak == min(6, max(3, budget)) * cache.expert_bytes
    assert submitted == [p for e in range(cache.n_experts) for _, p in cache._plans(e)]
    assert len(owners) == len(cache._writes) * (1 if budget <= 3 else 2)
    assert mo._T["em_bytes"] == cache.n_experts * cache.expert_bytes
    assert mo._T["em_groups"] == 6
    assert mo._T["em_buffer"] > 0
    assert mo._T.get("em_convert", 0) == 0
    monkeypatch.setattr(mo, "_io_pool", lambda: None)
    assert bits(prefill(cache)) == actual
    monkeypatch.setattr(mo, "ZERO_COPY", False)
    assert bits(prefill(cache)) == actual
    mo._T.clear()


@pytest.mark.parametrize("eof", [False, True])
def test_direct_read_short_chunks_and_interrupts(tmp_path, monkeypatch, eof):
    from omlx.patches.moe_expert_offload import _ReadPlan

    path = tmp_path / "slab.bin"
    payload = bytes(range(128))
    path.write_bytes(b"prefix" + payload)
    fd = os.open(path, os.O_RDONLY)
    plan = _ReadPlan(fd, 6, len(payload) + int(eof), np.uint8, None, (128 + int(eof),))
    dest = memoryview(bytearray(plan.nbytes))
    original = os.preadv
    offsets = []

    def short(fd, buffers, offset):
        offsets.append(offset)
        if len(offsets) == 1:
            raise InterruptedError()
        return original(fd, [buffers[0][:7]], offset)

    monkeypatch.setattr(os, "preadv", short)
    try:
        if eof:
            with pytest.raises(OSError, match="short read"):
                mo._read_into(plan, dest)
        else:
            mo._read_into(plan, dest)
            assert dest.tobytes() == payload
        assert offsets[:3] == [6, 6, 13]
        assert os.lseek(fd, 0, os.SEEK_CUR) == 0
    finally:
        os.close(fd)


@pytest.mark.parametrize(
    "dest",
    [memoryview(bytes(4)), memoryview(bytearray(3)), memoryview(bytearray(8))[::2]],
)
def test_direct_read_rejects_invalid_destination(dest):
    from omlx.patches.moe_expert_offload import _ReadPlan

    plan = _ReadPlan(-1, 0, 4, np.uint8, None, (4,))
    with pytest.raises(ValueError, match="destination"):
        mo._read_into(plan, dest)
