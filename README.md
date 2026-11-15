# omlx-inkling2-offload

Routed-expert offloading for Inkling 2 MoE models in oMLX.

## What it does

Inkling 2's routed MoE experts make up 330.23 GB of the 338.06 GB oQ4e index
(with 7.83 GB of non-expert weights). This package keeps a configurable
fraction of each layer's routed experts in memory slots (`[capacity, ...]`) and
streams non-resident experts on demand from the checkpoint's `.safetensors`
shards over oMLX's shared `CheckpointExpertStore` and I/O pool, allowing the
~338 GB oQ4e model (and ~393.5 GB original checkpoint) to run on Macs with less
memory.

## Supported oMLX version

- Upstream `jundot/omlx` `main` commit `624cff0e23519ba46a4d5ed2802c2c236ad1a9ac` (`624cff0e`).

`install.py` verifies exact SHA-256 checksums of the 6 patched upstream files
before modifying anything and works on both git checkouts and plain unpacked
source trees without `.git`.

## Supported checkpoints

Both checkpoint layouts are handled per layer:

- **Thinking Machines original layout** (`model.llm.layers.N.mlp.experts.*`):
  `*_weight` / `*_weight_scale` tensors in `U8` / `F8_E4M3` with derived NVFP4
  `global_scale` (NVFP4 in layers 0–39, MXFP8 in layers 40–47).
- **oQ4e runtime layout** (`model.layers.N.mlp.experts.{w13,w2}.*`):
  stacked `weight`, `scales`, optional `biases`, and `global_scale` (NVFP4 in
  layers 0–39, affine 4-bit gs64 in layers 40–47).
- **MXFP8-converted runtime layout**: supported through the same runtime code
  path and expected to work (covered on tiny test models; not yet benchmarked
  on the full ~622.8 GB checkpoint).

## Installation, check, and uninstall

Requires Python >= 3.10 (stdlib only for `install.py`).

Check status without modifying the target tree:

```bash
python3 install.py --omlx /path/to/omlx --check
```

Install into an oMLX `624cff0e` tree:

```bash
python3 install.py --omlx /path/to/omlx
```

Run the Inkling 2 offload tests and baseline offload/Inkling tests inside the
installed tree (run from inside `/path/to/omlx` with `PYTHONPATH=.` so Python
imports the target tree):

```bash
cd /path/to/omlx
PYTHONPATH=. pytest \
  tests/test_inkling2_split_experts.py \
  tests/test_inkling2_moe_offload.py \
  tests/test_inkling2_offload_wiring.py
```

Uninstall (restores the 6 patched files byte-identically, removes the added
module, tests, and `.omlx-inkling2-offload-record.json`; refuses if any touched
file was modified after install unless `--force` is given):

```bash
python3 install.py --omlx /path/to/omlx --uninstall
```

Double install is refused cleanly without modifying any files.

## Configuration and estimated memory

Enable offloading in oMLX model settings using:

- `moe_expert_offload_enabled`: `true` (default `false`)
- `moe_expert_offload_resident_fraction`: float in `(0, 1]` (default `0.25`;
  slot capacity per layer is `min(E, max(top_k, round(E * fraction)))`)

The table below gives **ESTIMATE** figures (calculated from checkpoint tensor
sizes for 384 routed experts per layer in oQ4e; not yet measured on the real
model). KV cache (~192 KiB/token across 48 layers, ~6 GiB at 32K) is additional.

| `moe_expert_offload_resident_fraction` | Slots (of 384) | Estimated Resident Weights |
| --- | --- | --- |
| `0.50` | 192 | 172.9 GB / 161.1 GiB (ESTIMATE) |
| `0.65` | 250 | 222.8 GB / 207.5 GiB (ESTIMATE) |
| `0.75` | 288 | 255.5 GB / 238.0 GiB (ESTIMATE) |

## v1 limits

- **No MTP / DSpark with offload**: `mtp_enabled`, `vlm_mtp_enabled`, and
  `dflash_enabled` are rejected on `inkling2` when offload is enabled.
- **No Prefill Accelerator with offload**: setting `prefill_helper_node_id` on
  `inkling2` with offload enabled is rejected at settings validation and load.
- **No float16 activation recast**: `activation_fp16_enabled` is rejected with
  MoE expert offload (upstream rule).
- **Speed unmeasured**: end-to-end throughput on a real Inkling 2 model has not
  yet been measured.

## Numerical contract

- **In capacity** (decode, batch, and prefill fitting within `capacity`):
  bit-identical (`array_equal`) to the split stock `Inkling2Layer` path, and
  within small floating-point rounding tolerance (`<= 5e-4` logits on tiny
  models) of the fused compiled stock path.
- **Over-capacity prefill**: evaluated via expert-major chunks of at most
  `capacity` experts; crossing the 64-route sort threshold introduces a small
  numerical tolerance (`< 2e-2` max abs error in unit tests).

## Adopting upstream in oMLX

1. Copy `module/omlx/patches/inkling2/moe_offload.py` (910 lines, MIT) into
   `omlx/patches/inkling2/moe_offload.py`, and copy the 3 test files in `tests/`
   into `tests/`.
2. Apply the 6 hook edits in `patches/omlx-624cff0e.patch` (+118 / -17 lines
   total, Apache-2.0):

| File | Diff (`+` / `-`) | Hook summary |
| --- | --- | --- |
| `omlx/patches/inkling2/model.py` | +78 / -12 | Add `routes()`/`combine()`, `route_inputs()`/`experts_out()`, and opt-in `_split_experts` stage (`_post_attn` / eager experts / `_post_tail`) |
| `omlx/patches/moe_expert_offload.py` | +11 / -2 | Add `F8_E4M3` byte-view dtype, dispatch `apply_inkling2_moe_expert_offload`, and add `inkling2` estimator branch |
| `omlx/engine/vlm.py` | +10 / -1 | Pass `lazy=bool(moe_expert_offload_enabled)` to `inkling2_vlm.load` |
| `omlx/model_settings.py` | +9 / -2 | Reject `prefill_helper_node_id` when offload is enabled on `inkling2` |
| `omlx/patches/moe_offload_compat.py` | +7 / -0 | Register `inkling2` in `_SUPPORTED_TYPES` and inspect via `estimate_inkling2_expert_savings` |
| `omlx/utils/model_loading.py` | +3 / -0 | Forward `prefill_helper_node_id` into `validate_moe_expert_offload` |

## Licensing

- `module/`, `tests/`, `install.py`, `README.md`, and `CHANGELOG.md` are
  licensed under the MIT License (`LICENSE`).
- `patches/omlx-624cff0e.patch` modifies Apache-2.0 oMLX files and is provided
  under the Apache License, Version 2.0 (`NOTICE`, `LICENSES/Apache-2.0.txt`).
