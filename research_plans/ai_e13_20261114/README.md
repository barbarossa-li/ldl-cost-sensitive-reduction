# E13: label association and axis-permutation controls

Fixed geometric50percent/NN5/half-prior/zeta.5 setting;Full primary,cost40secondary. Five label permutations learn alternative G weights while the actual prototype labels and prediction prior stay original. Five axis permutations preserve the original weight histogram but break its feature alignment. All12 predictors per mask shareK and actual arrays. GG50 andUU50 must exactly replay parent references.

This supplement is frozen after the five E12 prediction bundles exist but before E11/E12 outcomes are scored or inspected. It does not change the E12 primary. It uses the same53 cases, so adds mechanism contrasts, not independent datasets. Average individual seed risks, not predictions;retain each seed and every row. No bestseed,ensemble substitution or confirmatory pvalue.

Obtain/extract the frozen source package under ../ai_external_v1_20261114/ at b03a0d8e7fdf578ab0fc6f39669501784c057bb5. Its manifest/protocol hashes are checked. Use the same pinned Python environment.

```
python controls.py check --external-source /absolute/external_source
python controls.py run --external-source /absolute/external_source --known-data /private/known --novel-data /private/ren --external-run /private/external_run --output /private/e13_run
python controls.py score --external-source /absolute/external_source --known-data /private/known --novel-data /private/ren --external-run /private/external_run --output /private/e13_run
```

Both parent studies and all53 control bundles must be frozen before scoring. Source labels/permutation indices/per-object predictions remain private. Public release contains only code,protocol,aggregate risks/checksums. Original contribution claims still require external tuning, more unconsulted domains and verified identity/document grouping.
