# Public download verification — PASS

Verified 2026-11-12 (Asia/Shanghai). Packet commit: `3d4b80988b6a55f70e20e81c6c25e8feb96ec800`.

The commit-pinned public raw download of `ASC_REVIEW_REPRODUCIBILITY_v2.zip` was 2,114,605 bytes. SHA-256: `2989f90bfbc9f77115b3cf4b4412bc349e049332b7444f81f3751bdb2b6d0a2c`, matching the published SHA256SUMS and locally tested packet. Git blob SHA-1: `4d6f2ab06f430ee83c1f3a3cfc6b38d9821dddc9`.

ZIP CRC verification passed. After fresh extraction of the public copy, `python run_checks.py --figures` exited 0. All six check groups passed: independent candidate summary/protocol audit, recorded timing arithmetic, recorded envelope summary arithmetic, object decision diagnostics matched against saved results, family statistics recomputation matched against saved results, and presentation figure regeneration. The runner verified its internal SHA-256 manifest and used an isolated temporary input copy. Dependencies available in this verification environment: NumPy, SciPy and Matplotlib.

This is a rebuilt packet from ASC_LDL_integrated_source (6).zip, not a byte recovery of v1 and not a claim that the original 13-check runner has been restored. The checks operate on frozen records and do not rerun training, feature selection, raw-data prediction, image inference, error-vector replay, or hardware timing. Original-query-pool disjointness is not independently verified without the external return archive. Refer to README.md for scope. The historical v1 failure report remains valid for v1.
