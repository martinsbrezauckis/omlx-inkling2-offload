# SPDX-License-Identifier: MIT
"""Tests for window-bounded RotatingKVCache on Inkling 2 sliding layers (S7)."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchRotatingKVCache,
    CacheList,
    KVCache,
    RotatingKVCache,
)
from transformers import AutoTokenizer

from omlx.cache._rotating_subclass import PrefillReadyRotatingKVCache
from omlx.models.vlm import VLMModelAdapter
from omlx.patches.inkling2 import model as inkling2
from omlx.patches.inkling2.moe_offload import close_inkling2_moe_expert_offload
from omlx.patches.inkling2.vlm import Model as Inkling2VLMModel
from omlx.patches.moe_expert_offload import (
    apply_moe_expert_offload,
    materialize_offload_state,
)
from omlx.prefill_helper.adapters.inkling2 import Inkling2Adapter, _cache_state
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig
from omlx.utils.model_loading import materialize_lazy_state
from tests.test_inkling2_moe_offload import _build_tiny_model, _format_spec
from tests.test_inkling2_offload_wiring import (
    _load_inkling2_vlm,
    _save_wiring_checkpoint,
)
from tests.test_inkling_vlm_mtp import _inkling2_tiny


def _full_kv_cache(model) -> list[CacheList]:
    """Reference stock full-KVCache structure across all layers."""
    return [CacheList(KVCache(), ArraysCache(4)) for _ in model.model.layers]


def _greedy_generate(
    model,
    prompt_ids: list[int],
    *,
    steps: int = 20,
    chunk_size: int | None = None,
    use_full_kv: bool = False,
    unpadded_vocab: int = 300,
) -> tuple[list[int], list[CacheList], mx.array]:
    """Prefill (optionally chunked) + greedy decode, returning tokens, cache, last logits."""
    cache = _full_kv_cache(model) if use_full_kv else model.make_cache()
    x = mx.array(prompt_ids, dtype=mx.int32)[None, :]
    step = chunk_size or len(prompt_ids)
    logits = None
    for i in range(0, len(prompt_ids), step):
        logits = model(x[:, i : i + step], cache=cache)
    assert logits is not None
    prefill_last = logits[:, -1, :unpadded_vocab]
    tok = int(mx.argmax(prefill_last, axis=-1).item())
    out = [tok]
    for _ in range(steps - 1):
        step_logits = model(mx.array([[tok]], dtype=mx.int32), cache=cache)
        tok = int(mx.argmax(step_logits[:, -1, :unpadded_vocab], axis=-1).item())
        out.append(tok)
    mx.eval(out, prefill_last)
    return out, cache, prefill_last


def _sliding_kv_resident_bytes(model, cache: list[CacheList]) -> int:
    """Actual tensor bytes held inside keys_and_values() on sliding layers."""
    total = 0
    for layer, entry in zip(model.model.layers, cache, strict=True):
        if not layer.sliding:
            continue
        k, v = entry[0].keys_and_values()
        total += k.nbytes + v.nbytes
    return total


def _make_bf16_tiny(window: int = 32, seed: int = 0) -> inkling2.Model:
    mx.random.seed(seed)
    m = _inkling2_tiny(
        formats=("mxfp8",) * 6,
        sliding_window_size=window,
        rel_extent=window,
    )
    m.load_weights(
        [
            (
                k,
                (
                    v.astype(mx.bfloat16)
                    if m.cast_predicate(k) and v.dtype == mx.float32
                    else v
                ),
            )
            for k, v in tree_flatten(m.parameters())
        ]
    )
    mx.eval(m.parameters())
    return m


def test_make_cache_uses_rotating_only_for_sliding_layers():
    """Global layers (q_proj/k_proj/v_proj) keep KVCache; sliding layers use RotatingKVCache."""
    model = _inkling2_tiny(
        formats=("mxfp8",) * 12,
        sliding_window_size=24,
        rel_extent=24,
    )
    cache = model.make_cache()
    assert len(cache) == 12
    for idx, (layer, entry) in enumerate(zip(model.model.layers, cache, strict=True)):
        assert isinstance(entry, CacheList)
        assert isinstance(entry[1], ArraysCache)
        expected_sliding = bool((idx + 1) % 6)
        assert layer.sliding is expected_sliding
        if expected_sliding:
            assert hasattr(layer.attn, "in_proj")
            assert type(entry[0]) is RotatingKVCache
            assert entry[0].max_size == 24
        else:
            assert hasattr(layer.attn, "q_proj")
            assert hasattr(layer.attn, "k_proj")
            assert hasattr(layer.attn, "v_proj")
            assert type(entry[0]) is KVCache

    # VLM subclass inherits/shares the exact same make_cache behavior.
    vlm_cache = Inkling2VLMModel.make_cache(model)
    assert [type(c[0]) for c in vlm_cache] == [type(c[0]) for c in cache]


@pytest.mark.parametrize("prompt_len", [10, 16, 17, 53, 87])
@pytest.mark.parametrize("chunk_size", [None, 5, 16, 25])
def test_greedy_equality_vs_full_cache_across_lengths_and_chunks(
    prompt_len: int, chunk_size: int | None
):
    """Sliding RotatingKVCache matches full KVCache greedily across prompt & chunk sizes."""
    window = 16
    model = _inkling2_tiny(
        formats=("mxfp8",) * 6,
        sliding_window_size=window,
        rel_extent=window,
    )
    mx.eval(model.parameters())
    prompt = [((i * 13 + 7) % 250) + 10 for i in range(prompt_len)]
    ref_tokens, ref_cache, _ = _greedy_generate(
        model, prompt, steps=24, use_full_kv=True
    )
    rot_tokens, rot_cache, _ = _greedy_generate(
        model, prompt, steps=24, chunk_size=chunk_size, use_full_kv=False
    )
    assert rot_tokens == ref_tokens

    # Check sliding KV allocation is strictly bounded by window while global tracks full seq.
    total_seq = prompt_len + 24 - 1
    sliding_layers = sum(1 for layer in model.model.layers if layer.sliding)
    kv_heads = model.args.text.swa_num_key_value_heads
    head_dim = model.args.text.swa_head_dim
    elem_bytes = rot_cache[0][0].keys.dtype.size
    max_sliding_bytes = sliding_layers * 2 * kv_heads * window * head_dim * elem_bytes
    assert _sliding_kv_resident_bytes(model, rot_cache) <= max_sliding_bytes
    for layer, r_entry, f_entry in zip(
        model.model.layers, rot_cache, ref_cache, strict=True
    ):
        assert r_entry[0].offset == total_seq
        if layer.sliding:
            assert r_entry[0].keys.shape[2] == min(total_seq, window)
            if total_seq > window:
                assert r_entry[0].nbytes < f_entry[0].nbytes
        else:
            assert r_entry[0].keys.shape[2] >= total_seq


def test_flash_banded_bf16_chunking_above_and_below_window():
    """Exercise bf16 flash_banded_attention (window=32, L>=32) with chunking <, ==, > window."""
    window = 32
    model = _make_bf16_tiny(window=window, seed=7)
    prompt = [((i * 19 + 1) % 250) + 10 for i in range(105)]

    for chunk_size in (None, 15, 32, 40):
        f_toks, _, f_logits = _greedy_generate(
            model, prompt, steps=30, chunk_size=chunk_size, use_full_kv=True
        )
        r_toks, r_cache, r_logits = _greedy_generate(
            model, prompt, steps=30, chunk_size=chunk_size, use_full_kv=False
        )
        diff = float(
            mx.max(
                mx.abs(f_logits.astype(mx.float32) - r_logits.astype(mx.float32))
            ).item()
        )
        # chunk=40 > window=32 shifts BK=32 flash tile alignment after RotatingKVCache trim,
        # which can flip ~0.0005 bf16-ULP random near-ties; assert < 2e-3 logit agreement
        # and exact greedy parity for chunk <= window.
        assert diff <= 2e-3
        if chunk_size is None or chunk_size <= window:
            assert r_toks == f_toks
        assert r_cache[0][0].keys.shape[2] == window

    # In float32 (_masked_attention), chunk > window matches full KV greedily step-for-step.
    m_f32 = _inkling2_tiny(
        formats=("mxfp8",) * 6,
        sliding_window_size=window,
        rel_extent=window,
    )
    mx.eval(m_f32.parameters())
    ref_f32, _, _ = _greedy_generate(
        m_f32, prompt, steps=30, chunk_size=40, use_full_kv=True
    )
    rot_f32, _, _ = _greedy_generate(
        m_f32, prompt, steps=30, chunk_size=40, use_full_kv=False
    )
    assert rot_f32 == ref_f32


def test_batched_unequal_lengths_left_padding_filter_and_batch_generator():
    """Batch=2 and batch=3 unequal lengths match single-row full KV and survive filter/extract."""
    window = 16
    model = _inkling2_tiny(
        formats=("mxfp8",) * 6,
        sliding_window_size=window,
        rel_extent=window,
    )
    mx.eval(model.parameters())

    prompts = [
        [((i * 7 + 3) % 250) + 10 for i in range(11)],  # < window
        [((i * 11 + 5) % 250) + 10 for i in range(16)],  # == window
        [((i * 17 + 9) % 250) + 10 for i in range(53)],  # > 3x window
    ]
    refs = [_greedy_generate(model, p, steps=32, use_full_kv=True)[0] for p in prompts]

    # 1. Batch=2 via singleton prefill + CacheList.merge + multi-step decode past wrap,
    #    and batched L > 1 chunking with unequal left_padding.
    p_short, p_long = prompts[0], prompts[2]
    c_short, c_long = model.make_cache(), model.make_cache()
    model(mx.array(p_short[:-4], dtype=mx.int32)[None, :], cache=c_short)
    model(mx.array(p_long[:-4], dtype=mx.int32)[None, :], cache=c_long)
    bc2 = [CacheList.merge([a, b]) for a, b in zip(c_short, c_long, strict=True)]
    for layer, entry in zip(model.model.layers, bc2, strict=True):
        if layer.sliding:
            assert type(entry[0]) is BatchRotatingKVCache
        else:
            assert type(entry[0]) is BatchKVCache
        entry.prepare(lengths=[4, 4], right_padding=[0, 0])

    logits_chunk = model(
        mx.array([p_short[-4:], p_long[-4:]], dtype=mx.int32), cache=bc2
    )
    for entry in bc2:
        entry.finalize()
    t0 = int(mx.argmax(logits_chunk[0, -1, :300], axis=-1).item())
    t1 = int(mx.argmax(logits_chunk[1, -1, :300], axis=-1).item())
    out0, out1 = [t0], [t1]
    for _ in range(31):
        step_logits = model(mx.array([[t0], [t1]], dtype=mx.int32), cache=bc2)
        t0 = int(mx.argmax(step_logits[0, -1, :300], axis=-1).item())
        t1 = int(mx.argmax(step_logits[1, -1, :300], axis=-1).item())
        out0.append(t0)
        out1.append(t1)
    assert out0 == refs[0]
    assert out1 == refs[2]
    assert bc2[0][0].keys.shape[2] == window

    # 2. Batch=3 -> filter down to 2 survivors (dropping middle), continue decoding & verify.
    singles = []
    first_toks = []
    for p in prompts:
        c = model.make_cache()
        logits = model(mx.array(p, dtype=mx.int32)[None, :], cache=c)
        first_toks.append(int(mx.argmax(logits[0, -1, :300], axis=-1).item()))
        singles.append(c)
    bc3 = [
        CacheList.merge([c[i] for c in singles]) for i in range(len(model.model.layers))
    ]
    outs3 = [[t] for t in first_toks]
    cur = list(first_toks)
    for step in range(31):
        if step == 10:
            # Drop row 1 mid-decode; survivors are row 0 (short) and row 2 (long).
            for entry in bc3:
                entry.filter([0, 2])
            cur = [cur[0], cur[2]]
        step_logits = model(mx.array(cur, dtype=mx.int32)[:, None], cache=bc3)
        mx.eval(step_logits)
        nxt = [
            int(mx.argmax(step_logits[b, -1, :300], axis=-1).item())
            for b in range(len(cur))
        ]
        if step < 10:
            for b, t in enumerate(nxt):
                outs3[b].append(t)
        else:
            outs3[0].append(nxt[0])
            outs3[2].append(nxt[1])
        cur = nxt
    assert outs3[0] == refs[0]
    assert outs3[2] == refs[2]

    # 3. Extract row 0 after rotation and verify continued single-row decode matches.
    extracted0 = [entry.extract(0) for entry in bc3]
    assert type(extracted0[0][0]) is RotatingKVCache
    assert type(extracted0[-1][0]) is KVCache
    logits_ext = model(mx.array([[cur[0]]], dtype=mx.int32), cache=extracted0)
    tok_ext = int(mx.argmax(logits_ext[0, -1, :300], axis=-1).item())
    ref_ext = _greedy_generate(model, prompts[0] + refs[0], steps=1, use_full_kv=True)[
        0
    ][0]
    assert tok_ext == ref_ext

    # 4. mlx_lm BatchGenerator: RotatingKVCache must match full KVCache bit-for-bit
    #    when prefilling & decoding two unequal prompts together.
    def _run_batch_gen(use_full: bool):
        orig_make = model.make_cache
        if use_full:
            model.make_cache = lambda: _full_kv_cache(model)
        try:
            gen = BatchGenerator(model, max_tokens=24, stop_tokens=set())
            uids = gen.insert([p_short, p_long], [24, 24])
            seqs = {u: [] for u in uids}
            while True:
                res = gen.next_generated()
                if not res:
                    break
                for r in res:
                    seqs[r.uid].append(r.token)
            return [seqs[uids[0]], seqs[uids[1]]]
        finally:
            model.make_cache = orig_make

    assert _run_batch_gen(use_full=False) == _run_batch_gen(use_full=True)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8", "affine"])
def test_expert_offload_residency_half_matches_full_kv(tmp_path: Path, fmt: str):
    """Expert offload (residency=0.5) + RotatingKVCache matches full KVCache."""
    base = _build_tiny_model((_format_spec(fmt),) * 6, num_experts=8, top_k=2, seed=42)
    ckpt = tmp_path / f"offload_{fmt}"
    _save_wiring_checkpoint(ckpt, base, layout="runtime")
    off_model, _ = _load_inkling2_vlm(ckpt, lazy=True)
    assert apply_moe_expert_offload(off_model, ckpt, 0.5) == 6
    assert materialize_offload_state(off_model) == 6
    materialize_lazy_state(off_model)

    try:
        # _build_tiny_model uses sliding_window_size=8; exceed window by > 4x.
        window = off_model.args.text.sliding_window_size
        prompt = [((i * 13 + 5) % 250) + 10 for i in range(4 * window + 3)]
        ref_toks, _, _ = _greedy_generate(off_model, prompt, steps=16, use_full_kv=True)
        rot_toks, rot_cache, _ = _greedy_generate(
            off_model, prompt, steps=16, use_full_kv=False
        )
        assert rot_toks == ref_toks
        assert rot_cache[0][0].keys.shape[2] == window
    finally:
        close_inkling2_moe_expert_offload(off_model)


def test_scheduler_prefix_cache_reuse_and_memory_monitor(tmp_path: Path):
    """End-to-end Scheduler prefix-cache store/restore/partial+tail hit & concurrent batch=2."""
    window = 16
    tiny = _inkling2_tiny(
        formats=("mxfp8",) * 6,
        sliding_window_size=window,
        rel_extent=window,
    )
    ckpt = tmp_path / "sched_ckpt"
    _save_wiring_checkpoint(ckpt, tiny, layout="runtime")
    model, _ = _load_inkling2_vlm(ckpt, lazy=False)
    adapter = VLMModelAdapter(model)
    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    def _run_requests(scheduler: Scheduler, specs: list[tuple[str, list[int]]]):
        reqs = []
        for rid, tok_ids in specs:
            req = Request(
                request_id=rid,
                prompt=" ".join(f"t{t}" for t in tok_ids),
                prompt_token_ids=list(tok_ids),
                sampling_params=SamplingParams(temperature=0.0, max_tokens=20),
            )
            scheduler.add_request(req)
            reqs.append(req)
        deadline = time.monotonic() + 30.0
        while not all(r.is_finished() for r in reqs):
            scheduler.step()
            assert time.monotonic() < deadline
        for _ in range(5):
            scheduler.step()
            time.sleep(0.02)
        ssd = (
            scheduler.block_aware_cache.paged_ssd_cache
            if scheduler.block_aware_cache
            else None
        )
        if ssd is not None and hasattr(ssd, "flush"):
            ssd.flush(timeout=10.0)
        return reqs

    cold_sched = Scheduler(
        model=adapter,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            paged_ssd_cache_dir=None,
            max_num_seqs=4,
            prefill_step_size=64,
            model_name="tiny-inkling2",
        ),
    )
    warm_sched = Scheduler(
        model=adapter,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            paged_ssd_cache_dir=str(tmp_path / "ssd_cache"),
            hot_cache_only=True,
            hot_cache_max_size=64 * 1024**2,
            max_num_seqs=4,
            prefill_step_size=64,
            model_name="tiny-inkling2",
        ),
    )
    block_size = warm_sched.config.paged_cache_block_size
    assert block_size == 512  # 16 aligned into [512, 1024]

    # MemoryMonitor separates global KVCache from RotatingKVCache sliding layers.
    mm = warm_sched.memory_monitor
    assert mm._num_kv_cache_layers == 1
    assert mm._rotating_layer_specs == ((5, window),)

    prefix = [((i * 13 + 7) % 250) + 10 for i in range(520)]
    prompt_a = prefix + [((i * 17 + 3) % 250) + 10 for i in range(30)]
    prompt_b = prefix + [((i * 23 + 9) % 250) + 10 for i in range(30)]
    prompt_ext = prompt_a + [200, 201, 202, 203, 204]

    [cold_b] = _run_requests(cold_sched, [("cold-b", prompt_b)])
    [cold_ext] = _run_requests(cold_sched, [("cold-ext", prompt_ext)])

    [warm_a] = _run_requests(warm_sched, [("warm-a", prompt_a)])
    assert warm_a.cached_tokens == 0

    [warm_b] = _run_requests(warm_sched, [("warm-b", prompt_b)])
    assert warm_b.cached_tokens == block_size
    assert warm_b.output_token_ids == cold_b.output_token_ids
    assert warm_sched.block_aware_cache.get_stats().hits >= 1

    # Inspect reconstructed cache directly on the shared block-aligned prefix
    # (512 matched tokens + uncached suffix): sliding layers rebuild
    # PrefillReadyRotatingKVCache bounded at window, global keeps KVCache.
    probe_prefix = prefix[:block_size] + [295] * 16
    fetched_table, _ = warm_sched.block_aware_cache.fetch_cache("probe", probe_prefix)
    assert fetched_table is not None
    assert fetched_table.num_tokens == block_size
    rebuilt = warm_sched.block_aware_cache.reconstruct_cache(fetched_table)
    assert rebuilt is not None
    assert type(rebuilt[0][0]) is PrefillReadyRotatingKVCache
    assert rebuilt[0][0].offset == block_size
    assert rebuilt[0][0].keys.shape[2] == window
    assert type(rebuilt[-1][0]) is KVCache
    assert rebuilt[-1][0].offset == block_size

    # Tail-block extension hit (exact prefix of prompt_a stored at 549 tokens).
    [warm_ext] = _run_requests(warm_sched, [("warm-ext", prompt_ext)])
    assert warm_ext.cached_tokens == 549
    assert warm_ext.output_token_ids == cold_ext.output_token_ids

    # Multi-block (>= 2 blocks = 1024 tokens) partial prefix hit: stitches two
    # KVCache blocks on global layers while taking block 1's window on sliding layers.
    prefix_2blk = [((i * 11 + 3) % 250) + 10 for i in range(1030)]
    p2_a = prefix_2blk + [((i * 17 + 1) % 250) + 10 for i in range(25)]
    p2_b = prefix_2blk + [((i * 19 + 2) % 250) + 10 for i in range(25)]
    [cold_2b] = _run_requests(cold_sched, [("cold-2b", p2_b)])
    _run_requests(warm_sched, [("warm-2a", p2_a)])
    [warm_2b] = _run_requests(warm_sched, [("warm-2b", p2_b)])
    assert warm_2b.cached_tokens == 2 * block_size
    assert warm_2b.output_token_ids == cold_2b.output_token_ids

    # Concurrent batch=2 in Scheduler mixing a short uncached prompt (< window)
    # and a prefix-hit prompt (> block_size).
    p_short = [((i * 5 + 2) % 250) + 10 for i in range(11)]
    p_long = prefix + [((i * 29 + 11) % 250) + 10 for i in range(45)]
    [ref_s] = _run_requests(cold_sched, [("ref-s", p_short)])
    [ref_l] = _run_requests(cold_sched, [("ref-l", p_long)])
    got_s, got_l = _run_requests(
        warm_sched, [("batch-s", p_short), ("batch-l", p_long)]
    )
    assert got_s.output_token_ids == ref_s.output_token_ids
    assert got_l.cached_tokens == block_size
    assert got_l.output_token_ids == ref_l.output_token_ids


def test_mtp_partial_rollback_and_prefill_helper_past_window():
    """Verify mtp_partial_rollback & Inkling2Adapter._cache_state past sliding window."""
    window = 8
    model = _inkling2_tiny(
        formats=("mxfp8",) * 6,
        sliding_window_size=window,
        rel_extent=window,
    )
    mx.eval(model.parameters())

    prompt = [((i * 7 + 3) % 250) + 10 for i in range(20)]
    verify_tokens = [41, 42, 43, 44]  # draft_len = 3 -> verify L = 4
    accepted = 1  # accept token 41 + 42 (keep = 2), reject 43, 44

    # Reference: sequential prefill + keeping only accepted prefix [41, 42].
    ref_cache = model.make_cache()
    model(mx.array(prompt, dtype=mx.int32)[None, :], cache=ref_cache)
    ref_logits = model(
        mx.array([verify_tokens[: accepted + 1]], dtype=mx.int32), cache=ref_cache
    )
    ref_next = model(mx.array([[55]], dtype=mx.int32), cache=ref_cache)

    # Test cache: prefill, run full L=4 verify, roll back, then step next token.
    test_cache = model.make_cache()
    model(mx.array(prompt, dtype=mx.int32)[None, :], cache=test_cache)
    model(mx.array([verify_tokens], dtype=mx.int32), cache=test_cache)
    assert model.mtp_partial_rollback(test_cache, accepted=accepted, num_drafts=3)
    test_next = model(mx.array([[55]], dtype=mx.int32), cache=test_cache)
    mx.eval(ref_next, test_next)
    assert (
        float(
            mx.max(
                mx.abs(
                    ref_logits[:, -1, :300] * 0
                    + ref_next[:, -1, :300]
                    - test_next[:, -1, :300]
                )
            ).item()
        )
        < 1e-5
    )
    assert test_cache[0][0].offset == ref_cache[0][0].offset == 23

    # Prefill helper _cache_state after rotation/chunked prefill past window.
    chunk1 = [((i * 3 + 1) % 250) + 10 for i in range(12)]
    chunk2 = [((i * 5 + 2) % 250) + 10 for i in range(6)]
    helper_cache = model.make_cache()
    model(mx.array(chunk1, dtype=mx.int32)[None, :], cache=helper_cache)
    model(mx.array(chunk2, dtype=mx.int32)[None, :], cache=helper_cache)
    extracted = _cache_state(0, helper_cache[0], since=len(chunk1))
    assert extracted["0.k"].shape[2] == len(chunk2)
    dst = model.make_cache()
    model(mx.array(chunk1, dtype=mx.int32)[None, :], cache=dst)
    Inkling2Adapter().apply_state(dst, 1, extracted)
    assert dst[0][0].offset == len(chunk1) + len(chunk2)
    assert mx.array_equal(
        dst[0][0].keys_and_values()[0], helper_cache[0][0].keys_and_values()[0]
    )


def test_real_inkling2_small_config_memory_monitor_estimate():
    """Verify MemoryMonitor reflects 8 global + 40 rotating@1024 layers at 131072 tokens."""
    ta = inkling2.TextArgs()
    args = inkling2.ModelArgs(model_type="inkling2", text_config=vars(ta))

    class _MockRealModel:
        def __init__(self, margs):
            self.args = margs
            self.model = self
            self.model_type = "inkling2"
            self.config = {"model_type": "inkling2", "text_config": vars(ta)}
            self.layers = [
                type("L", (), {"sliding": margs.text.is_sliding(i)})()
                for i in range(margs.text.num_hidden_layers)
            ]

        def make_cache(self):
            return inkling2.Model.make_cache(self)

    tok = MagicMock()
    tok.encode.return_value = [1]
    sched = Scheduler(
        model=_MockRealModel(args),
        tokenizer=tok,
        config=SchedulerConfig(paged_ssd_cache_dir=None, model_name="inkling2-small"),
    )
    mm = sched.memory_monitor
    assert mm._num_layers == 48
    assert mm._num_kv_cache_layers == 8
    assert mm._rotating_layer_specs == ((40, 1024),)

    n = 131072
    bytes_per_layer_tok = 2 * 8 * 128 * 2  # 4096 B
    expected_global = n * 8 * bytes_per_layer_tok  # 4_294_967_296 B
    expected_sliding = 1024 * 40 * bytes_per_layer_tok  # 167_772_160 B
    assert mm.estimate_prompt_kv_bytes(n) == expected_global
    assert mm.estimate_resident_kv_bytes(n, chunk_tokens=1) == (
        expected_global + expected_sliding
    )
    # Must be strictly under 5 GiB vs 24 GiB (25.77 GB) before this change.
    assert mm.estimate_resident_kv_bytes(n, chunk_tokens=1) < 5 * 1024**3
