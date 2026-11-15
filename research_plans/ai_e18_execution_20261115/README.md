# E18 execution source v1

Implements frozen design6c60788e205491ec756d213629e777b9760b3a4c under research_plans/ai_e18_confirmation_20261115. This execution version must be committed before any real-data model fit. The design primary endpoint, all20partitions, target populations and candidate menus are unchanged.

Use the existing authorized input cache; no downloads occur in the runner. Base contains source/PROTOCOL.json, private/raw and private planned allocations from the design stage. Parent source is E11's frozen source bundle (b03a0d8e7fdf578ab0fc6f39669501784c057bb5). Every source/parent hash is checked before fitting. PyLDL toolkit commit80b2d188c11b640bce7accbc9a88dac89c48a56f and its code license remain in parent provenance.

```
LDL_PARENT_SOURCE=/path/to/parent/source python check_execution.py
python pipeline.py --base /path/to/E18 --parent-source /path/to/parent/source
```

Pipeline runs two case workers and waits for all15primary plus5rater-disjoint secondary records before any outer scoring. All17policies, failed cases and mean-only selections are retained;105training-only CV candidates per case. Encoder vocabulary/IDF/SVD refit independently inside each inner training fold. The native-fold text group allocations are loaded from the private commitment and never regenerated after outcomes. Portable inference state and storage accounting include required vocabulary/IDF/SVD files. Primary fixedGG50-vsUU50 differs from selected G_min-vsU_min, which may have unequal capacities; G_atU remains the matched selected comparison.

Read EXECUTION.json for warning compression and translation-stable Euclidean simplex projection (same mathematical objective), technical synthetic scope and precise resource boundaries. Original development predictions/results are unchanged. All raw texts/votes, IDs, allocations, encoder/model arrays and per-object outputs stay private; only code/protocol/checksums and aggregate evidence may be published. Two text application families, not three independent domains; rater transport is not old image-subject confirmation.

The local pipeline does not upload to external services or edit the manuscript. After audit completion, verify all outcomes/failures and synchronize using connected tools. Do not silently change frozen source after scoring. If necessary, record a distinct audited analysis correction preserving original evidence.
