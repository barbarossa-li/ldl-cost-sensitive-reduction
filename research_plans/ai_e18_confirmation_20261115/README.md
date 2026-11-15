# E18 independent confirmation: design freeze

This is a **design/source-preflight freeze**, not a completed experiment or an execution-ready full benchmark runner. No new-task model fits, predictions or risk scores have been run. The next execution source and its technical checks must be frozen before real-data fitting.

Three human-vote tasks: ChaosNLI-SNLI1514, ChaosNLI-MNLI1599 and DICES990 with989 metadata-consistent eligible items. These represent TWO text application families, not three independent domains or new image-subject validation. Read PROTOCOL.json for predeclared endpoints, all20 planned cases, source hashes, grouping/eligibility, model menus, failure and scoring gates.

Primary question: fixed GG50 vs same-capacity UU50, Full feature representation, k5/alpha.5, assessed separately in NLI and conversation safety. Later training-only selected policies and a DICES new-rater cohort analysis are secondary. Five folds are not five independent datasets. Existing E10--E17 negatives and E12 fixed prospective results remain unchanged.

Shared text TFIDF/SVD must be refit within EVERY inner training fold. IDs/votes/raters/metadata are never features. Encoder state must be included separately in deployment accounting. The protocol does not claim semantic-token explanations or foundation-model/NLP SOTA.

`preflight.py` downloads only missing verified data to private/raw; the ChaosNLI mirror hashes come from the pinned official downloader. `eligibility.py` inspects text/IDs/counts and produces private planned partitions with public counts. `design.py` contains deterministic connected grouping, capped complete-group splits and train-only text representation. `check.py` verifies these on synthetic data only. `PREFLIGHT_PUBLIC.json`, `ELIGIBILITY_PUBLIC.json` and `TECHNICAL_CHECKS.json` contain no sample IDs/row allocations or performance outcomes.

Sources: https://github.com/easonnie/ChaosNLI (commit2a999dc4a672df4ef87b9c7134247201352dee81; linked mirror revisioneb7cc18dbfa0f161d93a2b22c7856ca0faafcf2b), and https://github.com/google-research-datasets/dices-dataset (commitb5596d9edd585361967991a591abc7b11fb1f7a3). ChaosNLI is CC BY-NC4.0 per source; DICES CC BY4.0 per source. Raw data remain private regardless. Bounded project history/cache-index checks found no prior project evaluation of these tasks; only metadata/label-validity checks were done here. Publicly published benchmark tables were visible during source verification.
