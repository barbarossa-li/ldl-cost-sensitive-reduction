# Natural Scene: historical first real-LDL pilot

This report records the dependency-only stage. The predictive guard is now adopted in the working objective; see ../guarded_pilot/GUARDED_REPORT.md for the fresh follow-up.

The pilot found a weak feasible-reduction criterion, not a successful predictive reduction. The current method selected one feature with higher training dependency than all 294 features, but the downstream held-out distribution prediction was worse.

## Data and provenance

- Original dataset: 2,000 objects, 294 numerical attributes, 9-dimensional label distributions.
- Author-maintained repository: https://github.com/wangjing4research/LDL-LDM
- Pinned commit: 77b8b51f4d047c7b5cee063c2d545f4692d09f23
- Files: Scene/feature.npy and Scene/label.npy. Exact URLs and SHA-256 hashes are in data/sources.json.
- Dataset catalogue: https://palm.seu.edu.cn/xgeng/LDL/index.htm
- All arrays are finite; labels are nonnegative; maximum row-sum error is 6.66e-16. All 2,000 label rows have multiple positive components. No argmax conversion or synthetic label enhancement is used.
- Citation: Xin Geng, Label Distribution Learning, IEEE TKDE 28(7), 1734–1748, 2016. Repository is by the LDL-LDM research authors.

## Prespecified pilot protocol

One random permutation, seed 20260930: first 128 objects for training, next 256 for testing, 1,616 unused. These are subsamples of a real dataset, not its full benchmark protocol. Costs are simulated independent integers 1–10, seed 20260931, not measured acquisition prices.

Working defaults: zeta=.1, kappa=.1, beta=1, smin=2, alpha=1, zero split tolerances, lambda=-1, preservation tolerance zero. Numerical bandwidth is median positive pairwise normalized training difference. Geometry and feature scaling use training objects only. Feature search does not use test labels.

Downstream predictor is a uniform five-neighbor label-distribution regressor on the selected numerical features, using training-only min/max scaling. Predicted distributions average the five training label vectors. The no-feature predictor uses the training-global mean. Held-out metrics are mean squared label-vector error (Brier), KL with natural logarithms, TV and Jensen–Shannon divergence. Prediction is clipped at 1e-12 and renormalized for numerical logarithm safety.

## Observed results

| Method | Features | Simulated cost | Training dependency | Eligible coverage | Test Brier | Test KL |
|---|---:|---:|---:|---:|---:|---:|
| All attributes | 294 | 1648 | .09485 | 7.81% | .24382 | 1.09902 |
| Training global mean | 0 | 0 | 0 | 0% | .33234 | 1.05780 |
| Adopted heuristic | 1 | 1 | .15326 | 7.81% | .34869 | 1.45150 |
| Information-only ablation | 1 | 1 | .16069 | 10.94% | .37488 | 1.54551 |
| Cheapest same cardinality | 1 | 1 | .07216 | 8.59% | .39893 | 1.90339 |
| One random same-cardinality subset | 1 | 4 | .01289 | 6.25% | .35756 | 1.56201 |

The adopted selected feature is index 164 (zero-based, i.e. feature 165). Its Brier loss is 43.01% higher and KL loss 32.07% higher than the all-feature downstream predictor on this split. This is evidence about this pilot/predictor, not a significance claim across datasets.

Full-feature construction has 30 leaves, four eligible balls covering 10/128 objects, and 26 unresolved balls. The adopted single-feature construction has 42 leaves, five eligible balls covering 10/128, and 37 unresolved balls. Neither has singleton leaves. Removing singleton interpolation therefore did not suffice to prevent a weak reduction.

## Diagnose the objective, not just the search

Fifteen of 294 single features already satisfy Gamma({a}) >= Gamma(A). The adopted search finds a feasible cost-1 feature. The empty subset is infeasible, and every nonempty subset costs at least 1. Thus cost 1 is globally optimal over the full 294-feature space for THIS empirical feasibility criterion, without needing a completed 2^294 search. This certifies the cost objective, not predictive preservation.

The separate exact-search check uses eight training-screened candidates and its own full-candidate dependency target. Both searches return original feature index 209 at cost 5. The completed exact search visits 55 nodes and evaluates nine distinct subsets. This reference is scoped to eight candidates; it is not a full-space backtracking benchmark.

Training-only parameter diagnostics are saved. For example, at zeta=.25,kappa=.1 the full-feature score is zero, so the preservation target would be degenerate. This reinforces the need to report reference strength rather than treating every zero-cost result as success. No parameter setting was chosen from these test results.

## Next model change to evaluate

Retain directed geometry, balls, the dependency diagnostic and additive costs. Add a predictive guard assessed entirely inside the training procedure, such as R_inner(P) <= R_inner(A) + epsilon_pred, alongside Gamma(P) >= tau. Fold-specific preprocessing/statistics must be fitted on inner training objects. This would be an empirical model-selection constraint, not a generalization theorem. All-feature feasibility and cost-only exact pruning can remain intact if the guard compares the full set to itself.

A coverage threshold alone does not explain or exclude this failure: the full and selected subset cover the same 7.81%. A hard minimum feature count would conceal the weak objective. Further evaluation must use new untouched outer test objects/folds; this pilot test split has already informed diagnosis and must not serve as final evidence for a revised guard.

At this historical stage the guard was proposed. It is now adopted in the current Section 3 and tested on fresh objects in Section 6.

## Reproduce

From this directory, with NumPy/SciPy/Matplotlib installed:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_pilot.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python diagnose_training.py
python make_figures.py
```

Protocol indices and costs, CSV results, training diagnostics, search traces, exact comparison, input files and plots are included. The dense training cache and a single small split are not a scalable journal experiment suite.
