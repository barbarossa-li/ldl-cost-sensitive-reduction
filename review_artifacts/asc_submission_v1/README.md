# ASC reviewer reproducibility packet v1

Associated manuscript: **Cost-Sensitive Attribute Reduction for Label Distribution Learning with Objectwise Reconstruction Bounds**, 50-page manuscript source version 20.

Download [ASC_REVIEW_REPRODUCIBILITY_v1.zip](ASC_REVIEW_REPRODUCIBILITY_v1.zip), verify it using [SHA256SUMS](SHA256SUMS), and extract it. Read [REVIEW_PACKET_AUDIT.md](REVIEW_PACKET_AUDIT.md) for the preparation audit.

```bash
cd ASC_REVIEW_REPRODUCIBILITY_v1
python -m venv .venv
# Activate .venv for your operating system before the commands below.
python -m pip install -r requirements.txt
python run_checks.py --figures
```

Expected: 13 passed checks (12 numerical/statistical checks plus regeneration of five data figures). New logs and figures appear in `verification_outputs/`. The runner checks the internal manifest and works on a temporary copy of the frozen inputs.

This is offline arithmetic/statistical and figure reconstruction from frozen results. It does not rerun training, feature selection, raw-data predictions, image inference, raw error-vector replay, or new hardware timing. The package README and `REPRODUCTION_MATRIX.csv` specify the exact scope and external inputs needed for deeper reproduction.

Use the commit permalink of this directory for review. A main-branch link can change. The publication packet fixes the preparation runner to create the figure output directory after a clean extraction; frozen scientific records are unchanged. Its internal manifest and example logs have been refreshed. Publication/download verification is recorded separately in `PUBLIC_DOWNLOAD_VERIFICATION.md` after testing the public copy.
