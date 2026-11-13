# Current ASC manuscript and reproduction materials

**Current manuscript revision: 13 November 2026**, based on the 12 November terminology source. This directory path is retained to keep existing material links stable; the current files replace the previous PDF and source in place. Previous bytes remain in Git history.

## Files
- [Current manuscript PDF](ASC_LDL_terminology_revised.pdf) — 72 pages, review-aligned revision.
- [Editable source](ASC_LDL_terminology_source.zip) — entry `ASC_LDL_terminology/main_short_flowcharts.tex`.
- [Frozen-record reviewer packet](ASC_REVIEW_REPRODUCIBILITY_v2.zip) — experiment records and analysis unchanged.
- [Current SHA-256 checksums](SHA256SUMS).
- [Verification and revision scope](VERIFICATION.md).
- [Reproduction limits](REPRODUCTION_SCOPE.md).

The source ZIP includes `REVISION_NOTES.md`, the restored eight-object figure data and a check script. Text, notation and pseudocode were corrected to match the frozen code; original experimental numeric rows and the six existing plotted files are unchanged. Missing raw inputs or new experiments are not represented as completed work.

## Build

Extract the source, enter `ASC_LDL_terminology`, and run `latexmk -pdf main_short_flowcharts.tex` (pdfLaTeX and BibTeX). The figure-data check is `python check_synthetic_example.py /path/to/extracted/LDL_25_Expanded_Run` using the frozen installer payload indexed at the repository root.

## Frozen-record checks

Extract the separate reviewer ZIP, enter `ASC_REVIEW_REPRODUCIBILITY_v2`, install its `requirements.txt`, then run `python run_checks.py --figures`. Its six groups were rerun on 13 November and passed. The packet is unchanged by the manuscript revision. The command verifies saved records and does not rerun original-data selection, image inference, all reconstruction-vector hashes, query-pool disjointness or hardware timing. See the scope document.
