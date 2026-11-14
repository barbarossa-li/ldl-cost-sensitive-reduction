# E17 direct prototype-reduction pilot

This is an **explicit-assumption ProLSFEO paper-setting reconstruction**, not original-author code or a numerically verified reproduction. Source paper: https://doi.org/10.3390/app10093089 . A bounded search did not locate a verified author implementation. Read PROTOCOL.json for all assumptions, in particular validation-prototype exclusion, empty genomes, HUX pairing and restart semantics.

The pilot uses original population100/generations500/threshold10/k4 on only three consulted Full-feature development cases (Music/Painting/SJAFFE fold0), with one fixed seed. It is not the full published benchmark or independent evidence. If the3600s case budget prevents500 generations, that case is incomplete and must stay labeled as such. No reduced generation count is substituted.

```
python check.py
python run.py --parent-source /path/to/E11-source --parent-run /path/to/private/E11-E12-run --known-data /path/to/canonical-data --output /path/to/private/E17
```

Python3.12/numpy2.3.5. Data loading and validation use parent source/dependencies frozen at b03a0d8e7fdf578ab0fc6f39669501784c057bb5 (research_plans/ai_external_v1_20261114). Inner search code itself uses only numpy and explicit standalone state. All per-object data, partitions and genomes remain in the authorized private cache. Public results must show finite-fitness failures, completed generation count, prototype/feature/storage sizes and training cost, and preserve previous negative results.
