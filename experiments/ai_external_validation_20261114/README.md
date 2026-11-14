# External-default comparisons, prospective Ren variant and negative controls

Completed and independently audited E11--E15 research sequence. All original settings, budgets, raw/prior-adapted baseline outputs, negative results and numerical termination diagnostics are retained. The original78-page manuscript with Declarations remains preserved.

- [Chinese report](REPORT.md)
- [Complete aggregate results and audit scripts](AI_LDL_external_validation_results.zip)
- [Parent audit](EVIDENCE_AUDIT.json), [negative-control/prior audit](SUPPLEMENT_AUDIT.json), [convergence/state audit](CONVERGENCE_AUDIT.json)
- [Frozen E11/E12 source/protocol](../../research_plans/ai_external_v1_20261114/)
- [E13 pre-outcome negative controls](../../research_plans/ai_e13_20261114/)
- [E14 pre-outcome common-prior overlay](../../research_plans/ai_e14_20261114/)
- [E15 post-outcome numerical diagnostic](../../research_plans/ai_e15_20261114/)

## Main findings

The prespecified geometricGG50/half-prior setting improves mean Brier over the fixed external defaults on16 consulted feature families. On the prospectively fixed Ren_Cecps packaged variant, FullGG50 Brier is0.590613 versus AA-kNN20 0.587933 and native-toolkit RkNN0.581965. The more favorable secondary25% point does not replace the primary. All common-prior baseline overlays remain alongside raw outputs.

FullGG50 uses50.48% of its own singleton numeric-state bytes and improves risk from0.600159 to0.590613. This is a same-layout prototype tradeoff, not universal memory superiority: the exported linear SA state can be much smaller. Negative controls favor correctly associated/aligned weights over five label/axis permutations' mean individual risks, but new-corpuscost40 GG50 is slightly worse than uniformUU50.

## Evidence and limits

3445 aggregate prediction rows and24115 metric values independently recomputed; maximum arithmetic difference8.9e-16. These are not3445 independent experiments. E11 uses the same48 consulted training-pool folds and exactly replays816 E10 prototype/mean reference arrays. E12 has five disjoint384-fit/512-test allocations from one32420-row/100feature/8label Ren packaged variant. Exact-feature duplicates never cross allocations; document/author IDs are unavailable. This is not the35096-row full corpus described in earlier literature or broad independent cross-dataset confirmation.

E13/E14 were added and frozen after parent prediction generation but before any parent outcome scoring/access. They retain all53 existing cases and add no new independent data. E15 was explicitly designed after first outcome access:94/106 original SA fits hit500iterations; raising only the ceiling to20000 yields66/106 stop-criterion successes,40still at the ceiling. Training objectives do not increase, but test risks often worsen. All original500iteration outputs remain.212 standalone state predictions exactly replay after fitting, and12 already-converged native results replay exactly.

External code is the pinned MIT PyLDL toolkit, not verified original-author implementations. Native RkNN query-bandwidth batch dependence is retained; a training-only-gamma adapter is separately labeled. Baseline hyperparameters have not been fairly nested-tuned. Direct ProLSFEO reduction comparison, more unconsulted native-LDL sources, identity/document grouping and a complete resource benchmark remain open. Raw matrices, actual row allocations, permutations, per-object predictions and inference weights are excluded from public artifacts. Public package entry names are allowlisted and JSON structures checked for row-index/prediction-array fields.
