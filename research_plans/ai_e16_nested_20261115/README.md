# E16 restricted-grid training-only selection

This is a **development** analysis on all53 already-consulted E11/E12 cases. It adds shared three-fold inner selection, explicit prior-only options and a regularized linear reference. It is not independent confirmation or exhaustive SOTA tuning. Exact-feature duplicate grouping is not participant or document grouping.

Read PROTOCOL.json for all candidate menus, seeds, failure policies, storage boundaries and analysis restrictions. G_atU is the capacity-matched comparison; independent G/U selection does not ensure equal capacities. Preserve all earlier negative and fixed-primary results.

Parent source is frozen at commit b03a0d8e7fdf578ab0fc6f39669501784c057bb5 under research_plans/ai_external_v1_20261114/. Use its source ZIP and dependencies. Required versions are Python3.12, numpy2.3.5, scipy1.17.0, scikit-learn1.8.0, torch2.9.1+cpu and keras3.12.0. KERAS_BACKEND=torch. BLAS threads are limited to one; two case workers.

```
LDL_PARENT_SOURCE=/path/to/parent/source python check.py
python study.py run --external-source /path/to/parent/source --external-run /path/to/private/frozen-parent-predictions --known-data /path/to/known-npy --novel-data /path/to/Ren-npy --output /path/to/private/e16
```

Data and prior case records are obtained from the authorized private cache, with source hashes checked, not included in this public code. Keep per-object predictions, explicit partitions and model arrays private. The runner freezes predictions without outer scoring. Any subsequent scoring must happen only after all53 records are frozen and their hashes verified. Selection fits themselves necessarily evaluate only inner training-pool labels.

Technical checks use synthetic data and one/two epochs solely to verify code; real-data protocol uses100/500/1000 checkpoints. These short checks are not benchmark evidence.
