# omlx-inkling2-offload

Routed-expert SSD offload for Inkling 2 Small in oMLX 0.8.0rc1 (main 624cff0e).

## Requirements

- macOS, Apple Silicon.
- oMLX `624cff0e` source tree.
- Installer: Python >= 3.10; standard library only.
- oMLX runtime: Python >= 3.11, < 3.14; installed oMLX dependencies and native kernels.
- RAM: tested on 256 GB.
- Model: [mlx-community/Inkling-2-Small-oQ4e](https://huggingface.co/mlx-community/Inkling-2-Small-oQ4e) (318 GB).
- Package version: `0.2.0`.

## Install / check / uninstall

```bash
python3 install.py --omlx /path/to/omlx --check
python3 install.py --omlx /path/to/omlx
python3 install.py --omlx /path/to/omlx --check
python3 install.py --omlx /path/to/omlx --uninstall
```

- Exact SHA-256 verification: 8 patched files, 2 replaced fixtures, 14 added files.
- Original fixture bytes: compressed backups in `.omlx-inkling2-offload-record.json`.
- Double install: refused.
- Modified files: `--check` exits 1; install/uninstall refuse changes.
- Discard post-install edits: `--uninstall --force`.

## Settings

| Scope | Setting | Value | Fact |
| --- | --- | --- |
| `model_settings` | `moe_expert_offload_enabled` | `true` | Routed-expert offload |
| `model_settings` | `moe_expert_offload_resident_fraction` | `0.55` | Resident expert fraction |
| `model_settings` | `mtp_enabled` | `false` | Required with offload |
| Global | `scheduler.prefill_priority` | `"speed"` | Prefill priority |
| Global | `memory.memory_guard_tier` | `"aggressive"` | Memory guard |
| Global | `cache.ssd_cache_max_size` | `"100GB"` | SSD cache limit |
| Environment | `OMLX_PREFILL_STEP_SIZE` | `8192` | Recommended override; default `2048` |
| Optional environment | `OMLX_INKLING2_TIMING` | `1` | Default `0`; stage timing logs at most every 30 s during activity |
| Optional environment | `OMLX_MOE_IO_WORKERS` | `12` | Default; `0` or `1` for serial reads |
| Optional environment | `OMLX_INKLING2_DECODE_OVERLAP` | `1` | Default; `0` disables overlap |
| Optional environment | `OMLX_INKLING2_PREFILL_CACHE_CAP_GB` | `4` | Default; GiB; also bounded by 1/16 of working-set limit |
| Optional environment | `OMLX_INKLING2_PREFILL_NO_EVICT` | `1` | Default; preserve resident experts during prefill |
| Optional environment | `OMLX_INKLING2_PREFILL_GROUP` | `32` | Default; experts per compute group |
| Optional environment | `OMLX_INKLING2_PREFILL_IO_MB` | `2048` | Default; MiB; minimum one compute group |

```bash
export OMLX_PREFILL_STEP_SIZE=8192
export OMLX_INKLING2_TIMING=1
```

## Verify

```bash
python3 install.py --omlx /path/to/omlx --check

grep -F 'paged SSD cache enabled:' /path/to/omlx.log | grep -F 'block_size=8192 tokens'
grep -F 'inkling2 moe expert offload: wrapped ' /path/to/omlx.log
grep -F 'Inkling2 offload timing: wall=' /path/to/omlx.log

cd /path/to/omlx
PYTHONPATH="$PWD" /path/to/venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_moe_expert_offload.py tests/test_moe_expert_offload_live.py \
  tests/test_moe_offload_compat.py tests/test_deepseek_v4_offload.py \
  tests/test_glm_moe_dsa_offload.py tests/test_mlx_vlm_inkling_compat.py \
  tests/test_inkling_vlm_mtp.py tests/test_inkling2_*.py \
  tests/test_moe_io_workers_env.py tests/test_scheduler*.py tests/test_paged*.py \
  tests/test_prefix*.py tests/test_prefill_helper*.py
```

## Results

- Hardware: Mac Studio M5 Ultra, 256 GB unified memory.
- Model: Inkling-2-Small oQ4e, 318 GB; 55 % of routed experts resident.
- Generation tok/s = completion tokens / generation time; prompt tok/s = uncached prompt tokens / prompt time.

| Workload | Prompt tok/s | Generation tok/s |
| --- | --- | --- |
| Replay, 27 recorded agentic requests | 229.6 | 12.49 |
| Agentic coding run, first 123 requests | 130.7 | 11.90 |

## Limits

- MTP / DSpark rejected with offload.
- Prefill helper rejected with offload.
- Outputs not bit-identical across chunk sizes.
- Tested model: Inkling-2-Small oQ4e only.
- Memory guard: other apps lower oMLX's limit; prefill then slows or a too-large prompt is refused.

## Files

| Delivery | File | License |
| --- | --- | --- |
| Patch hunk | `omlx/engine/vlm.py` | Apache-2.0 |
| Patch hunk | `omlx/model_settings.py` | Apache-2.0 |
| Patch hunk | `omlx/patches/inkling2/model.py` | Apache-2.0 |
| Copied | `module/omlx/patches/inkling2/moe_offload.py` | MIT |
| Patch hunk | `omlx/patches/moe_expert_offload.py` | Apache-2.0 |
| Patch hunk | `omlx/patches/moe_offload_compat.py` | Apache-2.0 |
| Patch hunk | `omlx/prefill_helper/adapters/inkling2.py` | Apache-2.0 |
| Patch hunk | `omlx/scheduler.py` | Apache-2.0 |
| Patch hunk | `omlx/utils/model_loading.py` | Apache-2.0 |
| Replaced fixture | `tests/fixtures/prefill_contract/inkling2/case.npz` | Apache-2.0 |
| Replaced fixture | `tests/fixtures/prefill_contract/inkling2/meta.json` | Apache-2.0 |
| Copied | `tests/test_inkling2_decode_overlap.py` | MIT |
| Copied | `tests/test_inkling2_helper_cache.py` | MIT |
| Copied | `tests/test_inkling2_moe_offload.py` | MIT |
| Copied | `tests/test_inkling2_offload_io.py` | MIT |
| Copied | `tests/test_inkling2_offload_timing.py` | MIT |
| Copied | `tests/test_inkling2_offload_wiring.py` | MIT |
| Copied | `tests/test_inkling2_prefill_cache_scope.py` | MIT |
| Copied | `tests/test_inkling2_prefill_no_evict.py` | MIT |
| Copied | `tests/test_inkling2_split_experts.py` | MIT |
| Copied | `tests/test_inkling2_window_cache.py` | MIT |
| Copied | `tests/test_moe_io_workers_env.py` | MIT |
| Copied | `tests/test_scheduler_prefill_step_size.py` | Apache-2.0 |
| Copied | `tests/test_scheduler_rotating_arrays_block.py` | MIT |

## License

- MIT: module, new tests except the file below, installer, documentation; [LICENSE](LICENSE).
- Apache-2.0: patch hunks, upstream fixture replacements, and the existing notice in `tests/test_scheduler_prefill_step_size.py`; [Apache-2.0](LICENSES/Apache-2.0.txt).
- Attribution and file inventory: [NOTICE](NOTICE).
