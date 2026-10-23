# Prediction-Constrained Cost-Sensitive Attribute Reduction for Label Distribution Learning via Directed Granular-Ball Computing

Reference implementation and frozen experimental records for the manuscript submitted to *Applied Soft Computing*.

## Overview

The framework selects a minimum-acquisition-cost attribute subset for label distribution learning (LDL) subject to an internal prediction-feasibility guard:

- a local distribution-information weight and a source-dependent directed dissimilarity,
- a deterministic granular-ball model with non-monotone dependency,
- a constructive greedy search (dependency-guided `G` / prediction-only `P`), an exact backtracking reference, and a 200-evaluation deletion/exchange refinement (`G+` / `P+`),
- a nested audit protocol: frozen 576-object development subsets (432 search / 144 audit, block-wise nested), three inner folds, audit labels excluded from every selection decision.

## Repository structure

```
download_data.py            fetch all public benchmark datasets
granulation_reference.py    directed geometry + granular-ball model
reduction_reference.py      greedy / exact search with the loss guard
predictive_revision/        guided search variant (G)
refinement_pilot/refine.py  budgeted deletion/exchange refinement
stability/                  Natural Scene frozen splits (initial replication)
cross_dataset/              SBU-3DFE / Yeast_cold / Human_Gene / M2B runs
large_holdout_replication/  frozen 576-protocol on the anchor datasets
common_budget_audit/        common-budget ranking ablation
matched_cost_audit/         label-free matched-cost controls
pareto_audit/               Pareto frontier records
baselines_published/        GBFRS / DSIM / FSLC scoring cores (2022-2023)
dataset_expansion/          23-dataset expansion: main sweep, NRS / FMI cores
                            (2024-2025), guard ablation, ridge predictor study,
                            budget sweep, analysis and figure scripts
ASC_LDL_paper_draft.tex     current manuscript LaTeX source (inputs paper_*.tex,
  + paper_*.tex             refs_updated.bib, elsarticle class, paper_figures/)
highlights.tex              Elsevier highlights page
```

Each experiment directory contains its own protocol description, per-split
checkpoint files (`split.json`, `selection_before_test.json`, `results.json`),
and frozen results. All statistics used for selection are computed on training
data only; reserved-audit and held-out test scores are recorded once per frozen
selection.

## Reproduction

Requires Python 3.10+ with `numpy`, `scipy`, `matplotlib`.

```bash
python download_data.py                 # 1. fetch datasets (public sources)
# 2. anchor-dataset pipeline (Natural Scene / SBU-3DFE / Yeast-cold)
#    see stability/, cross_dataset/, large_holdout_replication/
cd dataset_expansion
python run_expansion.py                 # 3. 23-dataset main sweep
python run_expansion_published.py       # 4. GBFRS/DSIM/FSLC matched to G+
python run_expansion_newcores.py        # 5. NRS/FMI cores
python run_ablations.py all             # 6. guard ablation + ridge probe
python run_ridge_protocol.py            # 7. ridge-predictor full protocol
python run_budget_sweep.py              # 8. budget caps {1,2,5,10,20,50}
python analyze_expansion.py             # 9. tables, statistics, main figures
python generate_ablation_figures.py     # 10. ablation figures
python generate_final_figures.py        # 11. budget sweep + ridge table
```

All sweeps checkpoint per split and can be resumed after interruption.
Result CSV/JSON files from the paper's runs are already included, so every
table and figure can be regenerated without re-running the experiments.

## Datasets

All 25 native LDL benchmarks are publicly available (PyLDL and LDL-LDM
repositories); `download_data.py` fetches them and records the exact sources.
Label rows are normalized to sum one. `M2B` is retained as a predictor
suitability boundary case, not a primary benchmark.

## Citation

If you use this code, please cite the manuscript (citation entry will be added
upon acceptance).

## License

MIT License (see LICENSE).
