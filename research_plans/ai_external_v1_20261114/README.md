# E11 external-default development + E12 prospective Ren packaged-variant evaluation

Source and protocol must be committed before real-data prediction. Both studies must freeze all predictions before either new score is computed. Code dependencies are pinned and unchanged; own wrapper, diagnostics and explicit training-gamma adapter are labeled separately.

## E11

The same16 consulted families and48 E10 training-pool heldout folds. Compare five external prediction variants (AA-kNN5/20, SA-BFGS, native toolkit RkNN-LDL and separately labeled training-gamma RkNN adapter) with the fixed E10 UU/GG geometric25/50/75/100% alpha.5 references and mean baseline.27 outputs percase,1296 aggregate rows if no failures. Own references are copied and checked against frozen E10 predictions. No new original outer-test scoring, no tuned-baseline/SOTA claim.

## E12

One packaged native-LDL Ren_Cecps variant, actually32420x100 features and8labels, differs from the35096-row version described by DBC-LDL (IJCAI2019). No unknown filtering is attributed to the authors. Existing inspected26-dataset manifests and public paths contain no Ren evaluation records. This supports a prospectively frozen new-corpus evaluation, not broad independent cross-dataset confirmation.

Five disjoint allocations, each384 fit rows and512 test rows. Exact full-feature duplicate groups never cross fit/test or replicate allocations. Omitted members of consumed groups are not reused. Data has32203 such groups (largest10); original document/blog/author IDs are absent, so this is not document- or author-disjoint verification. Group allocation uses features only. Primary Full/GG50/alpha.5/zeta.5 configuration is fixed before any Ren prediction; cost40, other prespecified budgets and UU are secondary. No choosing the best budget after seeing test outcomes.

## Implementation and fairness limits

PyLDL toolkit code at80b2d188c11b640bce7accbc9a88dac89c48a56f, MIT code license preserved under vendor/LICENSE. These are not verified original-author implementations. RkNN-LDL reference DOI10.1016/j.patcog.2024.111006; AA/SA DOI10.1109/TKDE.2016.2545658. ProLSFEO-LDL (DOI10.3390/app10093089) remains an important direct reduction prior; a verified implementation is still pending.

External selected columns use fit-only minmax, constants mapped tozero, no clipping. AA uses Euclidean distance, SA is upstream softmax linear fitting without intercept; both are distinct from NN5. RkNN retains upstream defaults k20/alpha1e-3/beta1e-6/kernel/sparse/1000epochs/fullbatch/SGD1e-3. Native upstream query RBF bandwidth is estimated from the entire query-versus-training batch; disclose that query-batch dependence rather than silently fix it. The training-gamma adapter changes only query bandwidth, using the exact stored float32 training state; it shares the same fit and is not counted as a second fitted method. Upstream float32 outputs are evaluated as returned (mass tolerance1e-5), not renormalized. All errors/warnings and optimizer terminations retained. Fixed defaults are a baseline probe, not a substitute for fair nested hyperparameter tuning.

Metrics: Brier primary plus Chebyshev/Clark/Canberra/KL/cosine/intersection, formulas in runner.py. Three folds pooled by objects perfamily, then16 families equalweight. Ren pools5x512 objects for one dataset, not five independent datasets. Pairwise completeness and failures must be explicit; no confirmatory pvalues or bestpoint headline. Prototype fits reused in E11, so diagnostic timings cannot support a speed comparison. Numeric prototype bytes are not complete resource costs; cost40 own calibration still may useFull fit columns.

## Reproduce

Use Python3.12 and the pinned requirements. CPU PyTorch wheel index is in requirements.txt. Set KERAS_BACKEND=torch. Code package includes only code/license/protocol/checks, no historical data splits.

```bash
python -m pip install -r requirements.txt
python check.py
python prepare_data.py --known-data /private/known --novel-data /private/ren
python runner.py run --stage e11 --known-data /private/known --novel-data /private/ren --e10 /private/e10_run --output /private/external_run
python runner.py run --stage e12 --known-data /private/known --novel-data /private/ren --e10 /private/e10_run --output /private/external_run
python runner.py score --known-data /private/known --novel-data /private/ren --e10 /private/e10_run --output /private/external_run
```

The two run commands can execute independently; score requires both complete frozen manifests. Predictions, fit/test allocations, source matrices and detailed logs stay in private cache. Public artifacts contain only code, protocols, aggregate results and checksums. Outstanding: tuned comparison, ProLSFEO, more new native-LDL domains, subject/document grouping, full resource benchmark. Original78-page manuscript with Declarations remains preserved.
