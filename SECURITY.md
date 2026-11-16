# Security

## Reporting a problem

Please report security problems privately, not in a public issue:

- through GitHub's private vulnerability reporting on this repository (Security › Report a vulnerability), or
- by email to martins.brezauckis@gmail.com.

Say what you found, how to reproduce it, and which version and system you used. You will get an answer within a
few days. Fixes are released as soon as they are ready, and the release notes credit you unless you ask otherwise.

Problems in oMLX itself belong to the oMLX project: report them there. If you are not sure which side a problem is
on, report it here and it will be passed on.

## What is in scope

This repository: `install.py`, the offload module under `module/`, the patch under `patches/`, and the tests.

## What the package does

- `install.py` uses only the Python standard library, changes only files inside the oMLX tree you point it at, and
  verifies the SHA-256 of every file before and after patching. It refuses to install over modified files.
- The offload module reads expert weights from the model's own `.safetensors` files with `pread`; it opens no
  network connections and writes no files outside oMLX's normal logging.
