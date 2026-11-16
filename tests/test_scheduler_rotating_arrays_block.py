# SPDX-License-Identifier: MIT
"""Rotating-window + ArraysCache hybrids widen the paged block to the prefill step."""
from types import SimpleNamespace

from omlx.scheduler import Scheduler


def _sched(step, window, arrays, block=256):
    s = Scheduler.__new__(Scheduler)
    s.config = SimpleNamespace(
        paged_ssd_cache_dir="/tmp/x", paged_cache_block_size=block, prefill_step_size=step
    )
    s._detect_rotating_window_sizes = lambda: {window}
    s._detect_pooling_cache = lambda: False
    s._is_mimo_hybrid = lambda: False
    s._model_has_arrays_cache = lambda: arrays
    s._qwen35_prefill_floor = 0
    return s


def test_hybrid_block_follows_step():
    s = _sched(8192, 1024, True)
    Scheduler._align_block_size_with_rotating_window(s)
    assert s.config.paged_cache_block_size == 8192


def test_hybrid_block_rounds_up_to_window_multiple():
    s = _sched(3000, 1024, True)
    Scheduler._align_block_size_with_rotating_window(s)
    assert s.config.paged_cache_block_size == 3072


def test_hybrid_small_step_keeps_window():
    s = _sched(512, 1024, True)
    Scheduler._align_block_size_with_rotating_window(s)
    assert s.config.paged_cache_block_size == 1024


def test_rotating_only_unchanged():
    s = _sched(8192, 1024, False)
    Scheduler._align_block_size_with_rotating_window(s)
    assert s.config.paged_cache_block_size == 1024
