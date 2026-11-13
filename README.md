# LDL geometry and granular-ball research materials

**Active research target: Applied Intelligence — redesign in progress.** The [fixed cross-family mechanism study](experiments/ai_mechanism_v1_20261113/) is complete: all 80 cases across 16 known feature families were frozen before scoring, producing 12,560 aggregate metric rows. Under the frozen settings, local geometry improves over uniform geometry in 6/16 families, and ball smoothing improves over the coverage-matched mean control in 4/16; both primary equal-family mean differences are positive. These results do not support a broad superiority claim for the current geometry/readout. All negative outcomes and development history are retained. The [frozen mechanism protocol](research_plans/ai_mechanism_v1_20261113/) is unchanged. The current 78-page manuscript includes Declarations and has not been rewritten around unestablished new claims.

**Current training-internal development:** [E0.6 budget-first prototype curves](experiments/ai_geometry_e06_20261113/) are complete (48 folds,12,720 audited metrics). Exact25/50/75/100% prototype budgets provide explicit risk/capacity curves. At the fixed25% G/uniform/half-prior setting, candidate errors are lower than same-capacity Lloyd in13/16 Full families and12/16 cost40 families, while label-priority benefits over geometry-only remain inconsistent. Compression carries a mean risk cost versus full same-prior prediction; no broad superiority or novelty is claimed. [Source-amplitude diagnosis](experiments/ai_geometry_e04_20261113/) and [direct-prototype interface](experiments/ai_geometry_e05_20261113/) are retained as development provenance. No new outer-test scoring or manuscript changes.

**Cost-Sensitive Attribute Reduction for Label Distribution Learning Using Granular-Ball Computing with Objectwise Reconstruction Bounds**

This is the submission-materials index for the review-aligned manuscript revised on 13 November 2026 (from the 12 November terminology source). The working branch contains one manuscript version and the corresponding retained code and frozen records.

| Material | Location |
| --- | --- |
| Manuscript PDF, 78 pages | [PDF](review_artifacts/asc_submission_2026_11_12/ASC_LDL_terminology_revised.pdf) |
| Editable LaTeX project | [Source ZIP](review_artifacts/asc_submission_2026_11_12/ASC_LDL_terminology_source.zip) |
| Frozen-record reproduction packet | [Reviewer packet](review_artifacts/asc_submission_2026_11_12/ASC_REVIEW_REPRODUCIBILITY_v2.zip) |
| 25-task results, selected subsets, parameter CVs, splits/protocol manifests | [Results ZIP](review_audit/complete25/LDL_25_results_light.zip) |
| Full official 3,300-image O1 selection | [Code, protocol and completed results](experiments/official_image_selection_20261113/) |
| Additional original-data experiments (27 runs) and image replay | [Protocols, code and results](experiments/additional_20261113/) |
| Official image dataset source and cache policy | [SCUT-FBP5500 cache instructions](datasets/SCUT_FBP5500/) |
| Frozen experiment installer and numerical amendments | [Experiment runner](local_runner/) |
| Scope and limitations | [Reproduction scope](review_artifacts/asc_submission_2026_11_12/REPRODUCTION_SCOPE.md) |
| Submission consistency audit | [Audit report and scope](docs/FINAL_AUDIT_20261113.md) |
| Independent checks performed on 13 November | [Verification](docs/VERIFICATION_20261113.md) |

## Verify the frozen records

Extract the reviewer packet, enter `ASC_REVIEW_REPRODUCIBILITY_v2`, install `requirements.txt`, and run:

```bash
python run_checks.py --figures
```

This checks the internal SHA-256 manifest and runs six checks on frozen inputs: independent-candidate summaries, recorded timing, recorded storage envelopes, object decisions, feature-family statistics, and figure regeneration. It does **not** rerun feature selection from original datasets, independently reconstruct all error vectors, run ResNet image inference, or repeat hardware timing. See the scope document before interpreting PASS.

## Build the manuscript

Extract the source ZIP, enter `ASC_LDL_terminology`, and run:

```bash
latexmk -pdf main_short_flowcharts.tex
```

The current PDF/source incorporate the paragraph-review corrections: node eligibility, an explicit anchored oracle, seed-pair order, parameter/metric disclosure, a margin proposition, complexity and evidence limits. Frozen experiment code, pre-existing numeric table rows and result figures are unchanged. Revision details are inside the source ZIP as `REVISION_NOTES.md`; previous manuscript bytes remain in Git history. The 27-run geometry supplement, nine original-data replays, measured selection resources and official image-to-prediction replay are now integrated into the manuscript; source and detailed records are also indexed in the additional-experiment supplement. No editorial acceptance is implied.

## Rerun experiments

The frozen source and original split/protocol records are embedded in [BOOTSTRAP_25.py](local_runner/BOOTSTRAP_25.py). The installer verifies the embedded package and obtains datasets and third-party sources from recorded upstream versions. Read [runner instructions](local_runner/README.md) and [Movie/GBFRS amendments](local_runner/README_FAST.md). Keep the numerical amendment and exact-score optimization files together; they are part of the recorded implementation provenance. Full runs may take days and require external datasets.

The light results archive contains 936 final result/selection entries and 17,004 parameter-CV records. It is not a replacement for the full original return archive or image-pipeline SQLite caches. Those external records are not claimed to be included here.

## Historical versions

Superseded drafts, the earlier prediction-guard workflow, duplicate reviewer packets and development-only experiment folders were removed from the working branch on 13 November 2026. They remain recoverable from [the pre-cleanup commit](https://github.com/barbarossa-li/ldl-cost-sensitive-reduction/tree/1887e40dba08cef452b130b869f67c8fdfe143c1). No history was rewritten. The older root README described a different development workflow and should not be used for this submission.

No original face images, pretrained weights or unlicensed third-party sources are added by this cleanup. Code licensing does not replace dataset-specific permissions. See [LICENSE](LICENSE).

## Completed full-image reselection

The official 3,300-object / 512-coordinate full-to-fixed-reference O1 search is complete. All 512 single-coordinate deletions were infeasible; Full is locally single-deletion irreducible. The 513 actual error vectors contain 1,692,900 float64 values, retained privately with hashes. Two predeclared candidates were independently checked with the original full pair sorting, exactly matching errors. Test predictions equal Full (Brier 0.1064455253). See the full-image supplement and current manuscript Section 7.7.2. This is a known-split reproducibility run; subject-disjoint evaluation and complete cross-method training-resource benchmarks are not claimed.

## Latest consistency audit

The current 76-page PDF/source include 16 targeted corrections to evidence scope, observed-category domains, proof boundary cases and eager/streaming resource descriptions. All 21 table numeric sequences, displayed equations, algorithms and original plot bytes are preserved. The source includes FINAL_AUDIT_NOTES.md and paragraph_inventory.csv. This content audit does not complete missing author funding, competing-interest or contribution statements, verify live journal requirements, or constitute submission/acceptance.

## Applied Intelligence E0 pilot status

E0 completed 27 training-internal fold cases and 540 descriptive metric rows, with no new outer-test scoring. A fresh path-rebound artifact replay reproduces all 240 aggregate metric values exactly. Current local weights have weak influence under the original mixture scale and U/L neighbor sets largely coincide; the old global-fallback gains require a simple shrinkage control. [Results and code](experiments/ai_geometry_e0_20261113/). Next is a training-only calibration/readout-control development stage, followed by formal protocol freeze. The 76-page baseline remains unchanged.
