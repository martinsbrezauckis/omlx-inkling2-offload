# SPDX-License-Identifier: MIT
"""Read lifetime and byte bounds for the Inkling expert pipelines."""

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from omlx.patches.inkling2 import moe_offload as mo
from omlx.patches.inkling2.model import Inkling2Experts, TextArgs


@pytest.fixture
def slot_cache(tmp_path):
    args = TextArgs(hidden_size=64, intermediate_size=64, n_routed_experts=8)
    spec = {"mode": "affine", "bits": 4, "group_size": 64}
    experts = Inkling2Experts(args, spec, spec)
    mx.random.seed(14)
    for proj in (experts.w13, experts.w2):
        shape = (*proj.weight.shape[:2], 64)
        weights = (mx.random.normal(shape) * 0.05).astype(mx.bfloat16)
        proj.weight, proj.scales, proj.biases = mx.quantize(weights, **spec)
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"),
        {
            f"experts.{name}": value
            for name, value in tree_flatten(experts.parameters())
        },
    )
    store = mo.CheckpointExpertStore(tmp_path)
    cache = mo._SlotCache(experts, 4, mo._Inkling2StoreView(store, "experts"))
    mx.eval(experts.parameters())
    yield cache
    cache.close()


def _prefill(cache):
    # Exercise multiple routes per token and per expert across several groups.
    ids = list(range(cache.n_experts)) * 12
    inds = mx.array(ids, dtype=mx.uint32).reshape(-1, 2)
    x = mx.arange(inds.shape[0] * 64).reshape(-1, 64).astype(mx.bfloat16) / 1000
    weights = mx.full(inds.shape, 0.5, dtype=mx.float32)
    return mo.OffloadedInkling2Experts(cache)._forward_expert_major_no_evict(
        x, inds, weights, ids, 2, x.dtype
    )


@pytest.mark.parametrize("path", ["ensure", "ensure_ahead", "prefill"])
@pytest.mark.parametrize("failure", ["read", "convert", "submit"])
def test_exception_drains_every_submitted_read(slot_cache, monkeypatch, path, failure):
    """A running read in a popped group must finish before closing its store."""
    cache = slot_cache
    # These injections exercise the retained bytes/conversion path.
    monkeypatch.setattr(mo, "ZERO_COPY", False)
    monkeypatch.setattr(mo, "PREFILL_GROUP", 2)
    started, release = Event(), Event()
    submitted = []
    blocked = []
    real_read, real_wait = mo.CheckpointExpertStore.read, mo.wait
    first_plan = cache._plans(0)[0][1]
    # ensure_ids resolves all of expert 0 before conversion; hold expert 1 there.
    blocked_expert = 1 if failure == "convert" and path != "prefill" else 0
    blocked_plan = cache._plans(blocked_expert)[1][1]

    def read(plan):
        if plan == blocked_plan:
            started.set()
            assert release.wait(10), "cleanup did not drain the blocked read"
        if failure == "read" and plan == first_plan:
            assert started.wait(10)
            raise OSError("injected read failure")
        return real_read(plan)

    def drain(futures):
        futures = list(futures)
        # Release only when cleanup actually owns the running read. This makes
        # the old pop-before-result leak deterministic without timing sleeps.
        if blocked and blocked[0] in futures:
            release.set()
        return real_wait(futures)

    with ThreadPoolExecutor(max_workers=32) as pool:
        class Reader:
            def submit(self, fn, plan):
                if failure == "submit" and len(submitted) == 2:
                    assert started.wait(10)
                    raise RuntimeError("injected submit failure")
                future = pool.submit(fn, plan)
                submitted.append(future)
                if plan == blocked_plan:
                    blocked.append(future)
                return future

        monkeypatch.setattr(mo, "_io_pool", lambda: Reader())
        monkeypatch.setattr(mo, "_io_batch", lambda: 4)
        monkeypatch.setattr(mo.CheckpointExpertStore, "read", staticmethod(read))
        monkeypatch.setattr(mo, "wait", drain)
        if failure == "convert":
            def convert(*args):
                assert started.wait(10)
                raise ValueError("injected conversion failure")

            monkeypatch.setattr(cache.view, "to_slot_array", convert)
        try:
            with pytest.raises((OSError, ValueError, RuntimeError), match="injected"):
                if path == "prefill":
                    _prefill(cache)
                elif path == "ensure_ahead":
                    pending = cache.read_ahead([0, 1], 2)
                    cache.ensure_ids([0, 1], pending)
                else:
                    cache.ensure_ids([0, 1])
            assert submitted
            assert all(f.done() for f in submitted), "read outlived the call"
            cache.close()
        finally:
            release.set()


@pytest.mark.parametrize("budget_experts", [0, 2, 3, 4, 64])
def test_prefill_byte_cap_preserves_compute_groups(
    slot_cache, monkeypatch, budget_experts
):
    cache = slot_cache
    monkeypatch.setattr(mo, "ZERO_COPY", False)
    group_size = 2
    monkeypatch.setattr(mo, "PREFILL_GROUP", group_size)
    monkeypatch.setattr(
        mo, "PREFILL_IO_BYTES", budget_experts * cache.expert_bytes, raising=False
    )
    submitted, consumed, groups = [], set(), []
    outstanding_bytes = peak_bytes = 0

    class Read(Future):
        def __init__(self, plan):
            super().__init__()
            self.plan = plan
            self.set_result(mo.CheckpointExpertStore.read(plan))

        def result(self, timeout=None):
            nonlocal outstanding_bytes
            if self not in consumed:
                consumed.add(self)
                outstanding_bytes -= self.plan.nbytes
            return super().result(timeout)

    class Reader:
        def submit(self, fn, plan):
            nonlocal outstanding_bytes, peak_bytes
            future = Read(plan)
            submitted.append(plan)
            outstanding_bytes += plan.nbytes
            peak_bytes = max(peak_bytes, outstanding_bytes)
            return future

    original_routes = mo._TempExperts.routes

    def routes(self, x, inds):
        groups.append(self.w13.weight.shape[0])
        return original_routes(self, x, inds)

    monkeypatch.setattr(mo._TempExperts, "routes", routes)
    monkeypatch.setattr(mo, "_io_pool", lambda: None)
    expected = _prefill(cache)
    mx.eval(expected)
    expected_groups = list(groups)
    groups.clear()
    monkeypatch.setattr(mo, "_io_pool", lambda: Reader())
    # Successful consumption must not add a wait/synchronize at function exit.
    monkeypatch.setattr(mo, "wait", lambda _: pytest.fail("unexpected read wait"))
    actual = _prefill(cache)
    mx.eval(actual)

    assert peak_bytes <= max(group_size, budget_experts) * cache.expert_bytes
    expected_peak = min(2 * group_size, max(group_size, budget_experts))
    assert peak_bytes == expected_peak * cache.expert_bytes
    assert outstanding_bytes == 0
    assert submitted == [p for e in range(cache.n_experts) for _, p in cache._plans(e)]
    assert groups == expected_groups == [group_size] * 4
    assert mx.array_equal(actual, expected).item()

