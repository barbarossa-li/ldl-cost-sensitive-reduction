# ASC submission materials — 12 November 2026

Manuscript: **Cost-Sensitive Attribute Reduction for Label Distribution Learning Using Granular-Ball Computing with Objectwise Reconstruction Bounds**.

## Files
- [Revised manuscript PDF](ASC_LDL_terminology_revised.pdf) — 68 pages.
- [Editable LaTeX project](ASC_LDL_terminology_source.zip) — entry: `ASC_LDL_terminology/main_short_flowcharts.tex`.
- [Reviewer reproducibility packet](ASC_REVIEW_REPRODUCIBILITY_v2.zip) — frozen records, analysis scripts, requirements and internal SHA-256 manifest.
- [Checksums](SHA256SUMS).
- [Verification record](VERIFICATION.md).
- [Reproduction scope](REPRODUCTION_SCOPE.md).

## Quick start
Download the reproducibility ZIP from this directory and extract it. Open a terminal inside `ASC_REVIEW_REPRODUCIBILITY_v2`, then run:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_checks.py --figures
```

The command verifies the internal manifest, uses an isolated temporary input copy, writes logs and regenerated figures into `verification_outputs/`, and exits with status 0 when all six check groups pass. Dependencies: Python 3 with NumPy, SciPy and Matplotlib. Dependency versions are not pinned in this inherited v2 packet; see the tested versions in VERIFICATION.md.

For the manuscript, extract the separate source ZIP and run `latexmk -pdf main_short_flowcharts.tex` in its project directory (pdfLaTeX/BibTeX; TikZ and the other packages specified in the preamble are needed).

## Scope
This packet supports recomputation from frozen experimental records. It does **not** constitute complete end-to-end reproduction from original datasets. Read REPRODUCTION_SCOPE.md before interpreting a successful run. The v2 experimental packet is unchanged; this dated submission directory adds the terminology-revised manuscript and source. Earlier versions are retained for provenance.

No new experiment results were produced by the terminology revision. Mathematical expressions and table numeric rows were checked against the supplied colleague-editing source and were unchanged.
