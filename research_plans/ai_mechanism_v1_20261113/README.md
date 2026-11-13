# Fixed cross-family mechanism protocol v1

**Execution update, 13 November 2026:** [All 80 cases are complete and scored](../../experiments/ai_mechanism_v1_20261113/). The protocol and its originally packaged code remain frozen. The completion paragraph below records the initial freeze checkpoint.

The specification and runner are frozen before any new outer-test scoring; this is not a registry preregistration. Sixteen known feature families, five repeats, thirteen budget positions each: 80 cases and 1,040 positions. Fixed balanced zeta=.5, smoothing alpha=.5. Geometry and grouping have separate primary contrasts; see PROTOCOL.json. Old strict-O1 numbers and the new residual readout remain distinct.

After obtaining verified input NPY files from the authorized cache/upstream sources:

    python runner.py prepare --data /cache --output /private/run
    python runner.py worker --data /cache --output /private/run --family SCUT_FBP --repeat 1

Run worker once for each of the 16 families and repeat 1..5, preserving failures. Once all 80 prediction records exist:

    python runner.py freeze --data /cache --output /private/run
    python runner.py score --data /cache --output /private/run

No package commands download data or publish individual artifacts. `freeze` requires all 80 verified records before `score` can run. The source manifest pins code and design checks; actual generated splits/predictions stay private. The score output contains aggregate metric rows only. Dependencies: Python3.12/numpy2.3.5/scipy1.17.0; bundled numerical model, calibration, coverage controls, anchored oracle and frozen metrics.

Initial freeze checkpoint: design checks passed; 80 private splits prepared; 3/16 family inputs locally verified. One SCUT-FBP prediction case was executed to check the runner; no new outer-test loss was scored. Other inputs require cache-first resolution. Old Music floating-point training-size calculation is corrected to exact (7*N)//10=252 before main execution.

This protocol tests mechanism hypotheses on shared random masks. It does not establish feature-selector superiority, full-model compression or subject-disjoint generalization. External-selector comparisons need a separate frozen protocol.
