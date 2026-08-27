# Revision-audit provenance

The revised analysis assets in this branch were synchronized from experiment repository
commit `4037f0bf4aaffd6e39a88a92a7be92df6bab917e` after execution and verification on the
study server (`nokogiri`). No study analysis was executed on the workstation while
preparing this replication-package update.

## Integrity anchors

- Frozen detector export SHA-256:
  `8cd6aaf024a487ac372d95edc9e1b5ce91a98fd2d83af6de5394444722f6471d`
- Audited detector derivative SHA-256:
  `6af94e923188ebd57023d67f25a79e0f1aac7c760a8b4220a6b0fe292c438c63`
- Exclusion-manifest SHA-256:
  `1b9b82be9a52a623787946e002aa9c945c1636081c7580f737b0856b50a032d2`
- RQ1 paired figure SHA-256:
  `2b35a4ab09741b43f761dcd0148673d6f60619a9e97cc47f29c721c0372a266d`
- Fixed-corpus BM25 figure SHA-256:
  `313daa9cfd658a76a124d14cb60eddc26b7dcf7ad3bd7189e493e602e45fa6ff`
- Ten-page composite figure SHA-256:
  `f08c80f27a5bcdeeb125cffdc284cdf376d5438a6155b30192129a40318ae902`

`analysis/outputs/deep_audit/verification_report.json` records a `PASS` result and the
expected result-family cardinalities.

## Public-layout adaptations

The notebooks, analysis logic, result tables, datasets, and figures are copied from the
experiment repository. Documentation paths were adapted from the internal `phase3/`
layout to this repository's flat `analysis/`, `collection/`, and `datasets/` layout.
`analysis/deep_audit/server_guard.py` also permits an explicit
`LLM_PRS_ALLOW_REPLICATION_RUN=1` override for external reproduction; the study-team
default remains server-only.

The frozen `datasets/temperature_1/run_3/detection_results.csv` is preserved unchanged.
The revised manuscript uses `run_3_cleaned/detection_results.csv`; the 18 exclusions and
their reasons are recorded in `run_3_cleaned/generation_exclusions.csv`.
