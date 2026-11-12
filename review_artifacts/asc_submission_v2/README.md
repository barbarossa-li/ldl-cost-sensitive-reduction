# ASC reviewer reproducibility packet v2

Rebuilt from the user-supplied ASC_LDL_integrated_source (6).zip on 2026-11-12. This is a new packet; it does not recover the old v1 bytes or certify the old v1 checks.

Install requirements.txt, then run `python run_checks.py --figures`. The runner verifies the manifest and runs checks in an isolated temporary copy. Fresh logs and generated figures are placed in verification_outputs/.

Checks: the 512 independent candidate protocol/summary audit (without the original-return query-pool disjointness check); seven-round recorded timing arithmetic; 2043-candidate/six-endpoint envelope summary arithmetic; frozen-object decision diagnostics compared with the saved summary; and family-blocked statistical recomputation compared with saved results (49999 permutations). The optional figure check regenerates the presentation figures from frozen inputs. This packet contains no manuscript or author declarations.

These are frozen-record checks, not training, feature-selection reruns, independent error-vector replay, original-data prediction replay, image inference, new hardware timing, mathematical proof review, or bibliography checking. Scripts for deeper reproduction may require original datasets, code, return archives, model weights and hardware. In particular, the stored digest-reproduced flags in envelope summaries are checked as records; their underlying error vectors are not replayed here. Historical audit files remain historical evidence only.
