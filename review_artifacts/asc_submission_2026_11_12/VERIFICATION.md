# Verification record

Date: 2026-11-12 (Asia/Shanghai).

The publicly downloaded v2 packet was 2,114,605 bytes and matched SHA-256 `2989f90bfbc9f77115b3cf4b4412bc349e049332b7444f81f3751bdb2b6d0a2c`. ZIP CRC verification passed. After fresh extraction, `python run_checks.py --figures` exited 0; all six check groups passed.

Tested environment: Python 3.12.14; NumPy 2.3.5; SciPy 1.17.0; Matplotlib 3.10.8.

The revised PDF (1,219,685 bytes) and source ZIP (800,954 bytes) were uploaded as Git blobs and their returned blob SHA-1 values matched the local files:
- PDF: `78d5b4791572a84ee5bb79feebf5dfa35c4cd2f2`
- Source: `c6c85c80afb0732d53457bb4353d2e7b75b1efe0`

The LaTeX build passed, with no undefined references or overfull boxes. 628 mathematical expressions and table numeric rows were unchanged relative to the supplied colleague-editing project. The PDF contains 68 pages.

## Public download verification — PASS

The three binary files and SHA256SUMS were downloaded without authentication from commit-pinned public raw URLs at commit `22a58617b3bbdbddee9a047c5c5721824cbc6912`. All three SHA-256 checks matched; both ZIP CRC checks passed. The publicly downloaded reproducibility ZIP was freshly extracted, and `python run_checks.py --figures` exited 0 with all six groups PASS. No previously generated verification_outputs were used.

See REPRODUCTION_SCOPE.md for the limits of the frozen-record checks.
