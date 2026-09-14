# Exact corrected-study source layout

This subtree mirrors the verified experiment repository layout so that source paths and runtime
guards remain auditable. The public root `datasets/` directory is exposed here through the
relative `phase3/datasets` symlink.

Authoritative entry points:

- corrected generation: `../run_generation.sh`;
- corrected original-detector rerun: `../run_corrected_detection_pipeline.sh` and
  `detection/corrected_rerun/`;
- corrected Prelim-RQ/RQ1 analysis: `analysis/deep_audit/`;
- post-hoc Binoculars robustness source: `detection/b1_modern/`, with verified record-level scores
  and aggregate outputs under `analysis/outputs/tier_b/b1-modern-detectors-v1/`;
- corrected RQ2/RQ3 analysis: `analysis/corrected_rq23/`;
- genuine-issue audit: `analysis/b4_issue_source_audit/`; and
- generation-session diagnostics: `analysis/generation_session_*`.

The exact heavy-analysis launchers are server-specific and intentionally fail closed outside the
configured environment. Use the precomputed outputs for inspection, and run
`../verify_public_package.py` for a non-analytic public-package integrity check.
