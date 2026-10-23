# Dataset Expansion (ASC submission)

Twenty-three additional native LDL datasets executed under the frozen nested
audit protocol (576-object development subsets; single-192-block variant for
SJAFFE/Music/Painting; three repeats; identical guard, refinement cap,
predictor, and cost rule).

## Datasets
All from the PyLDL repository (SpriteMisaka/PyLDL, datasets/), converted with
scipy.io.loadmat keys features/labels and label rows normalized to sum one.
The five multi-target regression files (atp1d, enb, jura, scm1d, wq) and jaffe
were excluded as non-native LDL (raw_data keys, no distribution semantics).

## Reproduction order
1. `run_expansion.py`            — main sweep (checkpoints per repeat; resumable)
2. `run_expansion_published.py`  — GBFRS/DSIM/FSLC scoring cores, matched to G+
3. `analyze_expansion.py`        — summaries, figures (expansion_*), LaTeX tables

Outputs: `audit_results.csv`, `expansion_summary.csv`, `published_baselines/`,
`expansion_stats.json`; figures and tables are written to `../paper_figures/`
and `../paper_expansion*_table.tex`.

Key results: G+ cost reductions of 97.9-99.5% on the five informative datasets
(Natural Scene, SBU-3DFE, SJAFFE, fbp5500, RAF-ML); the internal guard
certifies the empty set on the other twenty native benchmarks, where the
global mean strictly beats the full-attribute kNN predictor on the audit
partition. Friedman over 69 published-core units: chi2=18.87, p=0.004;
no core separates from G+ by the Nemenyi CD (1.11).

## Round 2 additions (baselines expansion + ablations)

4. `run_expansion_newcores.py`   — NRS (neighborhood rough set, Wu et al. 2025 CPE)
                                     + FMI (fuzzy mutual information, You et al. 2024
                                     Inf. Sci.) scoring cores, matched-k/budget to G+.
5. `run_ablations.py noguard`    — guard-disabled forced-nonempty ablation on the 20
                                     guard-empty datasets (53/60 runs worse than the
                                     free empty set; 424 cost units spent).
6. `run_ablations.py ridge`      — predictor-sensitivity probe: ridge predictor revives
                                     9/20 datasets (>=2/3 repeats).
7. `run_ridge_protocol.py`       — full nested protocol under ridge on the 9 revived
                                     datasets; G+ dominates both Empty and Full on 6/9
                                     (FG-Net 0.0628@5 vs 0.0643@1412; MOR-PH 0.0707@1.7
                                     vs 0.0740@1162).
8. `run_budget_sweep.py`         — budget caps {1,2,5,10,20,50} ranking sweep on the
                                     informative datasets (rapid saturation).
9. `analyze_expansion.py`        — includes the 11-method Friedman/Nemenyi statistics
                                     (chi2=35.81, p=9e-5, CD=1.82; no core separates
                                     from G+).
10. `generate_ablation_figures.py` + `generate_final_figures.py` — all round-2 figures
    (noguard_guard, ridge_probe, refine_convergence, method_heatmap, audit_test,
    cost_strata, full_empty, budget_sweep) and paper_ridge_table.tex.
