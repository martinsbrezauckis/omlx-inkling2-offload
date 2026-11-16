# SPDX-License-Identifier: MIT
"""Prefix transfer must preserve both logical position and rotating storage."""

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, CacheList, RotatingKVCache

from omlx.patches.inkling2 import model as inkling2
from omlx.prefill_helper import wire
from omlx.prefill_helper.adapters.inkling2 import Inkling2Adapter


@pytest.mark.parametrize("keep", [0, 2])
@pytest.mark.parametrize("chunks", [[21], [8, 8, 5], [1] * 21, [1] * 5])
def test_rotating_prefix_round_trip_preserves_next_attention(monkeypatch, keep, chunks):
    source = RotatingKVCache(max_size=8, keep=keep)
    for length in chunks:
        keys = mx.arange(source.offset, source.offset + length).reshape(1, 1, -1, 1)
        source.update_and_fetch(
            keys.astype(mx.float32), (keys + 100).astype(mx.float32)
        )
    conv = ArraysCache(size=4)
    for i in range(4):
        conv[i] = mx.full((1, 3, 1), i, mx.float32)
    adapter = Inkling2Adapter()
    state = adapter.prefix_state([CacheList(source, conv)], 1)
    state = {
        name: wire.from_host(*wire.to_host(value)) for name, value in state.items()
    }
    restored = RotatingKVCache(max_size=8)
    destination = [CacheList(restored, ArraysCache(size=4))]
    adapter.load_prefix(destination, state)

    assert restored.offset == source.offset == sum(chunks)
    assert (restored.keep, restored.max_size, restored._idx) == (
        source.keep, source.max_size, source._idx
    )
    for actual, expected in zip(restored.keys_and_values(), source.keys_and_values()):
        assert mx.array_equal(actual, expected).item()
    for i in range(4):
        assert mx.array_equal(destination[0][1][i], conv[i]).item()

    # Observe the actual attention inputs, including temporal reordering after
    # the next write wraps the ring; no attention approximation is involved.
    calls = []

    def attend(q, k, v, *args, **kwargs):
        calls.append((k, v))
        return q

    monkeypatch.setattr(inkling2.kernels, "banded_sink_attention", attend)
    attn = inkling2.Inkling2Attention(inkling2.TextArgs(hidden_size=64), 0)
    for length in [1, 1, 3, 1]:
        keys = (
            mx.arange(source.offset, source.offset + length)
            .reshape(1, 1, -1, 1)
            .astype(mx.float32)
        )
        calls.clear()
        for cache in (source, restored):
            attn.attend(keys, keys, keys + 100, None, None, cache)
        assert source.offset == restored.offset
        for actual, expected in zip(calls[1], calls[0]):
            assert mx.array_equal(actual, expected).item()

