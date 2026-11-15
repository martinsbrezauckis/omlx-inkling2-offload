#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Installer for omlx-inkling2-offload against oMLX commit 624cff0e."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import tempfile
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PACKAGE_NAME = "omlx-inkling2-offload"
PACKAGE_VERSION = "0.1.0"
UPSTREAM_COMMIT = "624cff0e23519ba46a4d5ed2802c2c236ad1a9ac"
UPSTREAM_SHORT = "624cff0e"
RECORD_FILENAME = ".omlx-inkling2-offload-record.json"
PATCH_REL_PATH = Path("patches/omlx-624cff0e.patch")

# Expected SHA-256 digests of the 6 upstream files at commit 624cff0e.
UPSTREAM_SHA256: dict[str, str] = {
    "omlx/engine/vlm.py": (
        "d6a3dc462c73f17d7928904b0ae6a167c6d5f22ff39b3fff4a92b8681e008f0f"
    ),
    "omlx/model_settings.py": (
        "322d6bde1d861d2acad797fba9176f9c68c41f78a6e6aa322dd85e5f05386ceb"
    ),
    "omlx/patches/inkling2/model.py": (
        "aa2031cb7adac6100b77402efaab010b861b36a0edb4775c86a506e4e01281d4"
    ),
    "omlx/patches/moe_expert_offload.py": (
        "8826c11b2bcb6f7532b9be9a95822299c1e4e810e6c73fc0424861b5e3405e37"
    ),
    "omlx/patches/moe_offload_compat.py": (
        "841db0a7747e3b3067396c0cd0da14aa4ed558ce624874c9460b80b7dd9db595"
    ),
    "omlx/utils/model_loading.py": (
        "0b5441a0edae640cf39f37cf187b34c1193a40a30e042fc132cb65f20ece77ad"
    ),
}

# Expected SHA-256 digests of the 6 files after applying patches/omlx-624cff0e.patch.
PATCHED_SHA256: dict[str, str] = {
    "omlx/engine/vlm.py": (
        "30d9db177e936e45c2aa8d6a0ed6f162903373e5a8c101998765546459757b17"
    ),
    "omlx/model_settings.py": (
        "4fb61d00bd17e69c22df46f2b6ea153122d4f29b2acc1f794b9a01fb03452d83"
    ),
    "omlx/patches/inkling2/model.py": (
        "ddfee0730524a85bd21e3d46e30b0c4c105bebe6c5bbab36160d4e67eebdd137"
    ),
    "omlx/patches/moe_expert_offload.py": (
        "b38b306e004dcd41879354731ee73a879e1a846f9537db19239ec33c1a288eed"
    ),
    "omlx/patches/moe_offload_compat.py": (
        "93a58442ae7979c1ba4c6fc937f760d34b66dd48fa2dfda1e5c48e00b1afd9cb"
    ),
    "omlx/utils/model_loading.py": (
        "b52639d54e7fb027cfbfaddf2c81b8852147fd9f68708824a0dc5a9d071eabd9"
    ),
}

# Mapping from source path inside this package to destination path inside oMLX.
COPIED_FILES: dict[str, str] = {
    "module/omlx/patches/inkling2/moe_offload.py": (
        "omlx/patches/inkling2/moe_offload.py"
    ),
    "tests/test_inkling2_split_experts.py": "tests/test_inkling2_split_experts.py",
    "tests/test_inkling2_moe_offload.py": "tests/test_inkling2_moe_offload.py",
    "tests/test_inkling2_offload_wiring.py": "tests/test_inkling2_offload_wiring.py",
}

_HUNK_HEADER_RE = re.compile(rb"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?$")


class InstallError(RuntimeError):
    """Raised when preflight verification or patching fails cleanly."""


@dataclass(frozen=True)
class Hunk:
    """Single unified-diff hunk."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    ops: tuple[tuple[bytes, bytes], ...]


@dataclass(frozen=True)
class FilePatch:
    """Unified diff for a single modified file."""

    rel_path: str
    hunks: tuple[Hunk, ...]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _atomic_write(dest: Path, data: bytes) -> None:
    """Write ``data`` to ``dest`` via a same-directory temporary file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(dest.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_path, dest)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _strip_diff_prefix(raw_path: bytes) -> str:
    text = raw_path.decode("utf-8")
    if text.startswith(("a/", "b/")):
        return text[2:]
    return text


def _strip_line_ending(line: bytes) -> bytes:
    if line.endswith(b"\r\n"):
        return line[:-2]
    if line.endswith((b"\n", b"\r")):
        return line[:-1]
    return line


def _parse_unified_diff(patch_bytes: bytes) -> list[FilePatch]:
    """Parse a standard ``git diff`` unified patch into ``FilePatch`` entries."""
    lines = patch_bytes.splitlines(keepends=True)
    patches: list[FilePatch] = []
    idx = 0
    total = len(lines)

    while idx < total:
        line = lines[idx]
        if not line.startswith(b"diff --git "):
            idx += 1
            continue
        idx += 1
        old_path: str | None = None
        new_path: str | None = None
        while idx < total and not lines[idx].startswith(b"--- "):
            if lines[idx].startswith((b"diff --git ", b"Binary files ")):
                raise InstallError("Unexpected non-text diff entry in patch.")
            idx += 1
        if idx + 1 >= total or not lines[idx + 1].startswith(b"+++ "):
            raise InstallError("Malformed unified diff header (missing ---/+++).")
        old_path = _strip_diff_prefix(_strip_line_ending(lines[idx][4:]))
        new_path = _strip_diff_prefix(_strip_line_ending(lines[idx + 1][4:]))
        if old_path != new_path:
            raise InstallError(
                f"Patch renames or adds/deletes unsupported path: {old_path} -> "
                f"{new_path}"
            )
        idx += 2

        hunks: list[Hunk] = []
        while idx < total and not lines[idx].startswith(b"diff --git "):
            header_line = _strip_line_ending(lines[idx])
            match = _HUNK_HEADER_RE.match(header_line)
            if match is None:
                raise InstallError(
                    f"Malformed hunk header in {new_path}: {header_line!r}"
                )
            old_start = int(match.group(1))
            old_count = int(match.group(2)) if match.group(2) is not None else 1
            new_start = int(match.group(3))
            new_count = int(match.group(4)) if match.group(4) is not None else 1
            idx += 1

            ops: list[tuple[bytes, bytes]] = []
            seen_old = 0
            seen_new = 0
            while seen_old < old_count or seen_new < new_count:
                if idx >= total:
                    raise InstallError(f"Truncated hunk in {new_path}.")
                body_line = lines[idx]
                idx += 1
                tag = body_line[:1]
                payload = body_line[1:]
                if tag not in (b" ", b"-", b"+"):
                    raise InstallError(
                        f"Unexpected hunk line in {new_path}: {body_line!r}"
                    )
                no_nl = b"\\ No newline at end of file"
                if idx < total and lines[idx].startswith(no_nl):
                    payload = _strip_line_ending(payload)
                    idx += 1
                ops.append((tag, payload))
                if tag in (b" ", b"-"):
                    seen_old += 1
                if tag in (b" ", b"+"):
                    seen_new += 1

            if seen_old != old_count or seen_new != new_count:
                raise InstallError(f"Hunk line count mismatch in {new_path}.")
            hunks.append(
                Hunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    ops=tuple(ops),
                )
            )

        if not hunks:
            raise InstallError(f"No hunks found for {new_path}.")
        patches.append(FilePatch(rel_path=new_path, hunks=tuple(hunks)))

    if not patches:
        raise InstallError("Patch file contained no file diffs.")
    return patches


def _apply_file_patch(
    original: bytes, file_patch: FilePatch, *, reverse: bool = False
) -> bytes:
    """Apply (or reverse-apply) one ``FilePatch`` strictly with zero fuzz."""
    lines = original.splitlines(keepends=True)
    delta = 0

    for hunk in file_patch.hunks:
        if reverse:
            src_start, src_count = hunk.new_start, hunk.new_count
            dst_count = hunk.old_count
            expected = [payload for tag, payload in hunk.ops if tag in (b" ", b"+")]
            replacement = [payload for tag, payload in hunk.ops if tag in (b" ", b"-")]
        else:
            src_start, src_count = hunk.old_start, hunk.old_count
            dst_count = hunk.new_count
            expected = [payload for tag, payload in hunk.ops if tag in (b" ", b"-")]
            replacement = [payload for tag, payload in hunk.ops if tag in (b" ", b"+")]

        base_idx = (src_start - 1) if src_count > 0 else src_start
        pos = base_idx + delta
        if pos < 0 or pos + src_count > len(lines):
            raise InstallError(
                f"Hunk out of range in {file_patch.rel_path} at line {src_start}."
            )
        actual = lines[pos : pos + src_count]
        if actual != expected:
            raise InstallError(
                f"Context mismatch in {file_patch.rel_path} at line {src_start}."
            )
        lines[pos : pos + src_count] = replacement
        delta += dst_count - src_count

    return b"".join(lines)


def _validate_omlx_root(target_root: Path) -> None:
    if not target_root.exists():
        raise InstallError(f"Target path does not exist: {target_root}")
    if not target_root.is_dir():
        raise InstallError(f"Target path is not a directory: {target_root}")
    if not (target_root / "omlx").is_dir():
        if (target_root / "engine" / "vlm.py").is_file():
            raise InstallError(
                f"Target {target_root} looks like the inner omlx/ package directory; "
                "pass the oMLX repository root containing omlx/ and tests/."
            )
        raise InstallError(f"Target {target_root} does not contain an omlx/ directory.")


def _load_package_assets(pkg_root: Path) -> tuple[list[FilePatch], dict[str, bytes]]:
    patch_path = pkg_root / PATCH_REL_PATH
    if not patch_path.is_file():
        raise InstallError(f"Missing package patch file: {patch_path}")
    file_patches = _parse_unified_diff(patch_path.read_bytes())
    patch_targets = {fp.rel_path for fp in file_patches}
    expected_targets = set(UPSTREAM_SHA256)
    if patch_targets != expected_targets:
        raise InstallError(
            "Patch file target list does not match expected upstream files: "
            f"{sorted(patch_targets)}"
        )

    copied_data: dict[str, bytes] = {}
    for src_rel, dst_rel in COPIED_FILES.items():
        src_path = pkg_root / src_rel
        if not src_path.is_file():
            raise InstallError(f"Missing package source file: {src_path}")
        copied_data[dst_rel] = src_path.read_bytes()

    return file_patches, copied_data


def _load_record(record_path: Path) -> dict:
    try:
        data = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InstallError(f"Invalid install record {record_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise InstallError(f"Invalid install record {record_path}: expected object.")
    return data


def _check_status(pkg_root: Path, target_root: Path) -> tuple[str, list[str]]:
    """Inspect ``target_root`` without modifying it; return (status, details)."""
    _validate_omlx_root(target_root)
    _, copied_data = _load_package_assets(pkg_root)
    record_path = target_root / RECORD_FILENAME

    if not record_path.exists():
        mismatches: list[str] = []
        for rel, expected_sha in UPSTREAM_SHA256.items():
            path = target_root / rel
            if not path.is_file():
                mismatches.append(f"missing upstream file: {rel}")
            else:
                actual = _sha256_file(path)
                if actual != expected_sha:
                    mismatches.append(
                        f"modified upstream file: {rel} "
                        f"(expected {expected_sha[:12]}, got {actual[:12]})"
                    )
        existing_added = [
            dst for dst in COPIED_FILES.values() if (target_root / dst).exists()
        ]
        if existing_added:
            for dst in existing_added:
                mismatches.append(f"unexpected existing target file: {dst}")
        if not mismatches:
            return "not installed", [
                f"Target matches oMLX commit {UPSTREAM_SHORT} cleanly."
            ]
        return "modified", mismatches

    record = _load_record(record_path)
    patched_rec = record.get("patched_files", {})
    copied_rec = record.get("copied_files", {})
    diffs: list[str] = []

    for rel in UPSTREAM_SHA256:
        path = target_root / rel
        expected_after = patched_rec.get(rel, {}).get("after_sha256")
        if not path.is_file():
            diffs.append(f"missing patched file: {rel}")
            continue
        actual = _sha256_file(path)
        if actual != expected_after or actual != PATCHED_SHA256[rel]:
            diffs.append(
                f"modified after install: {rel} "
                f"(recorded {str(expected_after)[:12]}, got {actual[:12]})"
            )

    for dst_rel, pkg_bytes in copied_data.items():
        path = target_root / dst_rel
        expected_copy = copied_rec.get(dst_rel, {}).get("sha256")
        if not path.is_file():
            diffs.append(f"missing installed file: {dst_rel}")
            continue
        actual = _sha256_file(path)
        pkg_sha = _sha256_bytes(pkg_bytes)
        if actual != expected_copy or actual != pkg_sha:
            diffs.append(
                f"modified after install: {dst_rel} "
                f"(recorded {str(expected_copy)[:12]}, got {actual[:12]})"
            )

    if not diffs:
        return "installed", [
            f"All 6 patched files and 4 added files match install record "
            f"({RECORD_FILENAME})."
        ]
    return "modified", diffs


def _cmd_check(pkg_root: Path, target_root: Path) -> int:
    status, details = _check_status(pkg_root, target_root)
    print(f"Status: {status}")
    for line in details:
        print(f"  - {line}")
    return 0


def _cmd_install(pkg_root: Path, target_root: Path) -> int:
    _validate_omlx_root(target_root)
    file_patches, copied_data = _load_package_assets(pkg_root)
    record_path = target_root / RECORD_FILENAME

    if record_path.exists():
        raise InstallError(
            f"{PACKAGE_NAME} is already installed in {target_root} "
            f"({RECORD_FILENAME} exists). Run --check to inspect or "
            "--uninstall to remove. Nothing changed."
        )

    existing_added = [
        dst for dst in COPIED_FILES.values() if (target_root / dst).exists()
    ]
    if existing_added:
        raise InstallError(
            "Destination file(s) already exist in target tree: "
            f"{', '.join(existing_added)}. Nothing changed."
        )

    originals: dict[str, bytes] = {}
    mismatches: list[str] = []
    for rel, expected_sha in UPSTREAM_SHA256.items():
        target_file = target_root / rel
        if not target_file.is_file():
            mismatches.append(f"{rel} (missing)")
            continue
        raw = target_file.read_bytes()
        actual_sha = _sha256_bytes(raw)
        if actual_sha != expected_sha:
            mismatches.append(
                f"{rel} (expected {expected_sha[:12]}, got {actual_sha[:12]})"
            )
            continue
        originals[rel] = raw

    if mismatches:
        raise InstallError(
            f"Target tree does not match supported oMLX commit {UPSTREAM_SHORT}. "
            "Mismatching file(s): " + "; ".join(mismatches) + ". Nothing changed."
        )

    # Dry-run apply the unified diff completely in memory before touching disk.
    patched_contents: dict[str, bytes] = {}
    for fp in file_patches:
        patched = _apply_file_patch(originals[fp.rel_path], fp, reverse=False)
        after_sha = _sha256_bytes(patched)
        expected_after = PATCHED_SHA256[fp.rel_path]
        if after_sha != expected_after:
            raise InstallError(
                f"Dry-run patch output SHA-256 mismatch for {fp.rel_path}: "
                f"expected {expected_after[:12]}, got {after_sha[:12]}. "
                "Nothing changed."
            )
        patched_contents[fp.rel_path] = patched

    # All verification and dry-run checks passed; commit changes atomically.
    patched_record: dict[str, dict[str, str]] = {}
    originals_b64: dict[str, str] = {}
    for rel, patched in patched_contents.items():
        _atomic_write(target_root / rel, patched)
        patched_record[rel] = {
            "before_sha256": UPSTREAM_SHA256[rel],
            "after_sha256": PATCHED_SHA256[rel],
        }
        compressed = zlib.compress(originals[rel], level=9)
        originals_b64[rel] = base64.b64encode(compressed).decode("ascii")

    copied_record: dict[str, dict[str, str]] = {}
    for dst_rel, data in copied_data.items():
        _atomic_write(target_root / dst_rel, data)
        copied_record[dst_rel] = {"sha256": _sha256_bytes(data)}

    record = {
        "package": PACKAGE_NAME,
        "version": PACKAGE_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "installed_at": datetime.now(timezone.utc).isoformat(),
        "patched_files": patched_record,
        "copied_files": copied_record,
        "originals_zlib_b64": originals_b64,
    }
    _atomic_write(
        record_path,
        (json.dumps(record, indent=2) + "\n").encode("utf-8"),
    )

    print(
        f"Installed {PACKAGE_NAME} {PACKAGE_VERSION} into {target_root} "
        f"(6 patched, 4 added)."
    )
    print(f"Install record written to {record_path}")
    print("\nNext steps — run tests inside the target tree:")
    print(f"  cd {target_root}")
    print(
        "  PYTHONPATH=. pytest "
        "tests/test_inkling2_split_experts.py "
        "tests/test_inkling2_moe_offload.py "
        "tests/test_inkling2_offload_wiring.py"
    )
    print(
        "  PYTHONPATH=. pytest "
        "tests/test_moe_expert_offload.py "
        "tests/test_moe_expert_offload_live.py "
        "tests/test_moe_offload_compat.py "
        "tests/test_deepseek_v4_offload.py "
        "tests/test_glm_moe_dsa_offload.py "
        "tests/test_mlx_vlm_inkling_compat.py "
        "tests/test_inkling_vlm_mtp.py"
    )
    return 0


def _restore_bytes_for_file(
    fp: FilePatch,
    current_bytes: bytes | None,
    record: dict,
    *,
    force: bool,
) -> bytes:
    """Compute and verify original bytes for ``fp.rel_path`` during uninstall."""
    expected_orig = UPSTREAM_SHA256[fp.rel_path]
    if current_bytes is not None:
        try:
            reversed_bytes = _apply_file_patch(current_bytes, fp, reverse=True)
            if _sha256_bytes(reversed_bytes) == expected_orig:
                return reversed_bytes
        except InstallError:
            if not force:
                raise

    if not force:
        raise InstallError(
            f"Reverse patch did not restore original SHA-256 for {fp.rel_path}."
        )

    encoded = record.get("originals_zlib_b64", {}).get(fp.rel_path)
    if not isinstance(encoded, str):
        raise InstallError(
            f"Cannot --force restore {fp.rel_path}: no backup data in install record."
        )
    restored = zlib.decompress(base64.b64decode(encoded.encode("ascii")))
    if _sha256_bytes(restored) != expected_orig:
        raise InstallError(
            f"Backup SHA-256 mismatch for {fp.rel_path}; refusing to overwrite."
        )
    return restored


def _cmd_uninstall(pkg_root: Path, target_root: Path, *, force: bool) -> int:
    _validate_omlx_root(target_root)
    file_patches, _ = _load_package_assets(pkg_root)
    record_path = target_root / RECORD_FILENAME

    if not record_path.is_file():
        raise InstallError(
            f"{PACKAGE_NAME} is not installed in {target_root} "
            f"(missing {RECORD_FILENAME}). Nothing changed."
        )

    record = _load_record(record_path)
    patched_rec = record.get("patched_files", {})
    copied_rec = record.get("copied_files", {})

    modified: list[str] = []
    current_patched: dict[str, bytes | None] = {}
    for rel in UPSTREAM_SHA256:
        path = target_root / rel
        expected_after = patched_rec.get(rel, {}).get("after_sha256")
        if not path.is_file():
            modified.append(f"{rel} (missing)")
            current_patched[rel] = None
            continue
        raw = path.read_bytes()
        current_patched[rel] = raw
        actual = _sha256_bytes(raw)
        if actual != expected_after:
            modified.append(
                f"{rel} (recorded {str(expected_after)[:12]}, got {actual[:12]})"
            )

    for dst_rel in COPIED_FILES.values():
        path = target_root / dst_rel
        expected_copy = copied_rec.get(dst_rel, {}).get("sha256")
        if not path.is_file():
            modified.append(f"{dst_rel} (missing)")
            continue
        actual = _sha256_file(path)
        if actual != expected_copy:
            modified.append(
                f"{dst_rel} (recorded {str(expected_copy)[:12]}, got {actual[:12]})"
            )

    if modified and not force:
        raise InstallError(
            "File(s) changed after install: "
            + "; ".join(modified)
            + ". Re-run with --force to discard post-install edits and restore "
            "the 624cff0e originals. Nothing changed."
        )

    if modified and force:
        print(
            "Warning: --force specified; reverting post-install edits in: "
            + ", ".join(modified),
            file=sys.stderr,
        )

    restored_contents: dict[str, bytes] = {}
    for fp in file_patches:
        restored_contents[fp.rel_path] = _restore_bytes_for_file(
            fp, current_patched[fp.rel_path], record, force=force
        )

    for rel, original_bytes in restored_contents.items():
        dest = target_root / rel
        _atomic_write(dest, original_bytes)
        if _sha256_file(dest) != UPSTREAM_SHA256[rel]:
            raise InstallError(f"Post-write SHA-256 verification failed for {rel}.")

    for dst_rel in COPIED_FILES.values():
        (target_root / dst_rel).unlink(missing_ok=True)
    record_path.unlink(missing_ok=True)

    print(
        f"Uninstalled {PACKAGE_NAME} from {target_root} "
        "(restored 6 files byte-identically, removed 4 files and install record)."
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"Install, check, or uninstall {PACKAGE_NAME} on oMLX {UPSTREAM_SHORT}."
        )
    )
    parser.add_argument(
        "--omlx",
        required=True,
        help="Path to the target oMLX repository root or unpacked source tree.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Report status (not installed / installed / modified) without changes.",
    )
    mode.add_argument(
        "--uninstall",
        action="store_true",
        help="Restore upstream files byte-identically and remove installed files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --uninstall, restore 624cff0e originals even if files changed.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.force and not args.uninstall:
        parser.error("--force can only be used together with --uninstall")

    pkg_root = Path(__file__).resolve().parent
    target_root = Path(args.omlx).expanduser().resolve()

    try:
        if args.check:
            return _cmd_check(pkg_root, target_root)
        if args.uninstall:
            return _cmd_uninstall(pkg_root, target_root, force=args.force)
        return _cmd_install(pkg_root, target_root)
    except InstallError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
