# E14: common fitting-prior overlay

Pre-outcome secondary supplement to E11/E12. Own GG50 uses half of the fitting-label mean. To avoid attributing generic prior shrinkage to geometry, retain each unchanged external prediction and separately add .5*prediction+.5*fittingmean. The cached plain_mean arrays provide that prior; no new fit, target access or parameter choice enters prediction. Both native RkNN and its fixed-training-gamma adapter retain separate labels. All raw and adapted versions must be reported; the original primary/native comparisons are unchanged.

This is declared after parent prediction freezing but before any parent or E13 scores are computed or inspected. It adds no independent data. Both masks, five parent prediction variants and53cases are retained, yielding530 additional aggregate rows if no parent failures. Seven original metrics, Brier primary. No best-adapter selection or confirmatory pvalue. Ren remains one packaged dataset and all data limitations in the parent protocol continue to apply.

Use the pinned external-v1 source/environment at b03a0d8e7fdf578ab0fc6f39669501784c057bb5. Public artifacts contain code/protocol/aggregate results/checksums; object predictions and split records stay in the authorized private cache.

```
python prior_overlay.py run --external-source /absolute/external_source --external-run /private/external_run --output /private/e14_run
python prior_overlay.py score --external-source /absolute/external_source --external-run /private/external_run --e13-run /private/e13_run --known-data /private/known --novel-data /private/ren --output /private/e14_run
```

All53 derived outputs, all53 E13 bundles andboth parent manifests must freeze before any scoring.
