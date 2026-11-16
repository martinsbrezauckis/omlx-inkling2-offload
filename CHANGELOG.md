# Changelog

## 0.2.0 (2026-11-16)

- Decode read/compute overlap.
- Bounded prefill cache and I/O read-ahead.
- Prefill expert groups without resident eviction.
- Window-bounded KV cache and speculative rollback.
- Prefill helper contract v2 and rotating-cache fixtures.
- Configurable prefill step size and paged-cache block alignment.
- Configurable I/O worker count.
- Offload stage timing.
- Regression tests for all changes.
- Byte-exact fixture replacement and uninstall backups.
- Nonzero status checks for modified files.

## 0.1.0 (2026-11-15)

- Initial Inkling 2 routed-expert offload module, tests, and installer for oMLX `624cff0e`.
