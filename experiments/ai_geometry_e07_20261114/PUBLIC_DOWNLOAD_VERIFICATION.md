# Public E0.7 download verification — PASS

Date: 2026-11-14 (Asia/Shanghai).

Protocol/source freeze commit: `4917253807a4318aa437b1b0d151e0234847e6b8` (before benchmark execution).

Result package commit: `01a509d265f4e00488cd5071c32b8896847aaf68`.

The unauthenticated, commit-pinned public raw download of AI_LDL_E07_nested_budget_code_results.zip was 317,085 bytes. SHA-256 `c7dd8c390a1c77f8115c9a9dd690c4a1fdebb9403a2eaee30b9ab47c529b3a09` matched the local tested archive and SHA256SUMS. ZIP CRC verification passed.

After fresh extraction, results/verify_public.py passed its manifest check, aggregation of 4,656 metric rows into 1,552 family rows, 24 fixed-budget contrasts and 12 selected cap100 policies. source/check.py passed all seven synthetic technical checks, including deliberate replacement of outer-holdout labels without changing inner risks, selected settings or predictions.

These public-copy checks are aggregate replay and synthetic tests; they do not download original data or independently regenerate all benchmark predictions. The original E0.7 run used 32 hash-verified arrays, froze all 48 prediction bundles before scoring, and independently recomputed all 4,656 Brier values from the private predictions (maximum difference 5.551115123125783e-17). The 2,304 fixed-budget settings common with E0.6 matched exactly; they are replay evidence, not new independent results. New computation is nested inner-budget selection and its heldout training-pool evaluation. No original outer-test scoring was performed.

All 16 families have already been consulted during development; no independent-confirmation, novelty, significance or manuscript-readiness claim follows from these checks. Private row indices and predictions are not included in the public package.
