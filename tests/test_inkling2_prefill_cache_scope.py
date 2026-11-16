# SPDX-License-Identifier: MIT
"""Over-capacity Inkling 2 prefill bounds MLX's allocator pool and restores the limit."""
import threading

import mlx.core as mx
import pytest

from omlx.patches.inkling2 import moe_offload as mo


class _Limits:
    def __init__(self, monkeypatch, start=1 << 40):
        self.value = start
        self.calls = []
        monkeypatch.setattr(mo.mx, "set_cache_limit", self.set)

    def set(self, limit):
        prev, self.value = self.value, int(limit)
        self.calls.append(int(limit))
        return prev


def test_scope_sets_cap_and_restores(monkeypatch):
    lim = _Limits(monkeypatch)
    monkeypatch.setattr(mo, "get_max_working_set_bytes", lambda: 240 << 30)
    with mo._prefill_cache_scope():
        assert lim.value == min(mo.PREFILL_CACHE_CAP, (240 << 30) // 16)
    assert lim.value == 1 << 40
    assert mo._cache_scope_depth == 0


def test_scope_nested_sets_once(monkeypatch):
    lim = _Limits(monkeypatch)
    with mo._prefill_cache_scope():
        with mo._prefill_cache_scope():
            pass
        assert lim.value != 1 << 40
    assert lim.calls.count(1 << 40) == 1 and len(lim.calls) == 2
    assert lim.value == 1 << 40


def test_scope_restores_on_error(monkeypatch):
    lim = _Limits(monkeypatch)
    with pytest.raises(RuntimeError):
        with mo._prefill_cache_scope():
            raise RuntimeError("boom")
    assert lim.value == 1 << 40 and mo._cache_scope_depth == 0


def test_scope_disabled(monkeypatch):
    lim = _Limits(monkeypatch)
    monkeypatch.setattr(mo, "PREFILL_CACHE_CAP", 0)
    with mo._prefill_cache_scope():
        pass
    assert lim.calls == []


def test_scope_overlapping_threads(monkeypatch):
    lim = _Limits(monkeypatch)
    inside = threading.Barrier(2)
    def run():
        with mo._prefill_cache_scope():
            inside.wait(5)
    ts = [threading.Thread(target=run) for _ in range(2)]
    [t.start() for t in ts]; [t.join(10) for t in ts]
    assert lim.value == 1 << 40 and len(lim.calls) == 2


def test_expert_major_runs_inside_scope(monkeypatch):
    seen = []
    class Probe(mo.OffloadedInkling2Experts):
        def __init__(self):
            pass
        def _forward_expert_major_no_evict(self, *a):
            seen.append(mo._cache_scope_depth)
            return "ok"
        def _forward_expert_major_install(self, *a):
            seen.append(("install", mo._cache_scope_depth))
            return "ok"
    _Limits(monkeypatch)
    assert Probe()._forward_expert_major(None, None, None, [], 8, mx.float32) == "ok"
    monkeypatch.setattr(mo, "PREFILL_NO_EVICT", False)
    assert Probe()._forward_expert_major(None, None, None, [], 8, mx.float32) == "ok"
    assert seen == [1, ("install", 1)]
    assert mo._cache_scope_depth == 0
