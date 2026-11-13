# Current ASC manuscript

Updated 13 November 2026: 76 pages, including the full official-image O1 reselection and earlier paragraph-review corrections / 27-run supplement. Existing PDF/source links are preserved; old bytes remain in Git history.

- [Current PDF](ASC_LDL_terminology_revised.pdf)
- [Editable LaTeX source](ASC_LDL_terminology_source.zip), entry ASC_LDL_terminology/main_short_flowcharts.tex
- [Full official-image selection: code/protocol/results](../../experiments/official_image_selection_20261113/)
- [Earlier 27-run geometry and image-prediction supplement](../../experiments/additional_20261113/)
- [Checksums](SHA256SUMS)
- [Verification](VERIFICATION.md)
- [Frozen-record reviewer packet](ASC_REVIEW_REPRODUCIBILITY_v2.zip), unchanged

The full image search retains 512/512 coordinates: all 512 single deletions violate the fixed objectwise bound. It establishes a local endpoint, not a global minimum. The test predictions equal Full. Source REVISION_NOTES.md distinguishes the successive stages. Subject-disjoint evaluation is still unavailable without a verified identity map. Raw inputs and individual vectors remain in the private Drive cache. Build with latexmk -pdf main_short_flowcharts.tex.
