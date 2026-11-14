# E15: numerical convergence and standalone-state diagnostic for SA-BFGS

This follows, rather than replaces, the already-scored E11/E12/E13/E14 outcomes.94 of106 original SA-BFGS fits stopped at the500iteration limit. Increase only the supported optimizer iteration ceiling to20000, retaining original loss, tolerances, seeds, fitting rows, preprocessing and both raw/half-prior outputs. Freeze this source before diagnostic fitting and every diagnostic prediction before scoring. Any remaining nonconvergence must remain visible. This is post-outcome numerical development, not independent confirmation.

Each fit also exports a standalone numeric state (columnindices,minmaxlo/span,W;prior only forhalfversion). Its prediction must exactly equal the upstream predictor after fitting. Report this measured layout separately from fitting caches and from hardware memory/timing. A small linear predictor may use fewer numeric bytes than a prototype model; do not claim universal storage superiority or equate different architectures' capacities.

Use the unchanged pinned external-v1 vendor code at b03a0d8e7fdf578ab0fc6f39669501784c057bb5. Source protocol/manifest hashes are checked. Two single-threadCPUprocesses execute the same53existing cases. Per-object arrays, actualfit/testrows andinferenceweights remain in the authorized private cache. Publish only code,protocols,aggregate metrics/convergence/state-byte diagnostics andchecksums.

```
python convergence.py check --external-source /absolute/external_source
python convergence.py run --external-source /absolute/external_source --external-run /private/external_run --known-data /private/known --novel-data /private/ren --output /private/e15_run
```

Scoring/audit scripts are separate from fitting. The original prospective primary, fullbudget curves and every native/comparison result remain unchanged. Fair baseline hyperparameter tuning, direct ProLSFEO reduction comparison, further unconsulted data and verifiedidentity/document grouping remain open.
