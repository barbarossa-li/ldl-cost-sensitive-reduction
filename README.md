# LDL and granular-ball research materials

## Cost-sensitive attribute reduction

Code, protocols and results supporting **Objectwise Reconstruction Bounds for Cost-Sensitive Attribute Reduction in Label Distribution Learning via Granular-Ball Computing**.

The authors' current manuscript and consolidated submission package were last revised on 18 November 2026 and are distributed separately. This repository provides the supporting materials below.

| Material | Entry point |
| --- | --- |
| Frozen-record arithmetic and statistical checks | [Reproduction packet](review_artifacts/asc_submission_2026_11_12/) |
| 25-task results, selected subsets, CV records and split manifests | [Results archive](review_audit/complete25/LDL_25_results_light.zip) |
| Geometry experiments and image-prediction replay | [Code, protocols and results](experiments/additional_20261113/) |
| Complete official-image selection | [Code, protocols and results](experiments/official_image_selection_20261113/) |
| Benchmark runner and numerical amendments | [Runner instructions](local_runner/) |
| Image dataset source and access information | [SCUT-FBP5500](datasets/SCUT_FBP5500/) |

Extract the reproduction packet, install its requirements and run `python run_checks.py`. Add `--figures` to redraw historical figures. These checks recompute saved records; use the experiment-stage instructions for fresh selection or image processing. Dataset access terms and required external inputs are documented with each stage.

The 15 November search controls are included in the manuscript's accompanying reproducibility package; they are not claimed as a new repository release here.

## Related research

Separate development studies are retained under [experiments](experiments/) and [research_plans](research_plans/). Their individual READMEs state their protocols, results and scope. They should not be treated as additional results of the cost-sensitive reduction manuscript.

## Historical material

Superseded manuscript copies and internal editing audits were removed from the default branch during presentation cleanup. They remain available at the [17 November pre-cleanup commit](https://github.com/barbarossa-li/ldl-cost-sensitive-reduction/tree/3a1ad10a7bd79ba3cdde302db56798c9fca1430c). Experimental records and Git history are retained.

See [LICENSE](LICENSE) for code licensing; dataset permissions remain governed by their providers.
