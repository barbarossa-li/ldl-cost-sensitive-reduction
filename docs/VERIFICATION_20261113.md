# Independent verification — 13 November 2026

Checked snapshot: `1887e40dba08cef452b130b869f67c8fdfe143c1`.

## Version identity

The pre-revision manuscript source ZIP matched the user-designated terminology revision byte for byte: SHA-256 `930c77375181f84d969e6f5fa45d136d76eb77374c6f71df9b99840e6b1f5e81`, Git blob `c6c85c80afb0732d53457bb4353d2e7b75b1efe0`. The reviewer packet SHA-256 is `2989f90bfbc9f77115b3cf4b4412bc349e049332b7444f81f3751bdb2b6d0a2c`.

## Checks run

- Freshly extracted reviewer packet: all six groups from `python run_checks.py --figures` passed. Bundled `verification_outputs` were excluded from the test input copy by the runner; checks were executed again.
- All 275 displayed numerical/missing-value cells in the main acquisition-cost and Brier tables agree with the corresponding frozen result records at manuscript precision.
- All 936 frozen final-result entries have matching selected-subset records; valid attribute counts agree.
- Recalculation of the two 192-object mean-versus-objectwise diagnostics reproduces 9 mean-selector violations on Yeast-cold and 24 on SCUT-FBP, with zero O1 violations in those diagnostics.
- The frozen formal-method check passed 100 nonmonotone vector-oracle cases, including fixed-reference tolerance handling. The complete-sweep check passed 200 six-attribute nonmonotone feasibility tables and their terminal deletion certificates.

Environment: Python 3.12.14; NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8. These checks did not install or change the frozen experiment dependencies.

## Interpretation

The statistical and diagnostic calculations above are recomputations from saved records. They do not establish that every saved record was independently reproduced from original observations. The 512-candidate and storage-envelope checks use saved candidate summaries, not independent SQLite error-vector replay or query-pool disjointness reconstruction. Timing checks use the recorded seven rounds and do not constitute new hardware measurements. Full 25-task selection, original-data prediction and image inference were not rerun.

The core reference geometry, safe extension and deletion modules are available inside the frozen installer, rather than at the old repository-root paths. The cleanup itself made no mathematical or experimental changes. The subsequent manuscript-only revision corrects the implementation description and disclosure; see the dated submission directory verification and source revision notes. Frozen experiment implementations and results remain unchanged.

## Cleanup scope

Retained: the single dated manuscript/reviewer directory, all original runner/amendment files, the complete 25-task light results ZIP, license, ignore rules and this index/check record. Removed working-tree files are available at the pinned pre-cleanup commit; no commits, branches or Git history were erased. Original retained files are unchanged.

## Subsequent review-aligned manuscript revision

The current 72-page PDF and source were revised after the record checks above. Independent node eligibility, the distance-ordered anchored oracle, parameter/metric defaults, the zero-reference exclusion policy, a formal margin result, complexity and evidence limits are now specified. The current checksums are in the submission directory; source/plot numeric invariance and the restored synthetic example were checked. Original-data experiment reruns and missing-resource measurements are not claimed.
