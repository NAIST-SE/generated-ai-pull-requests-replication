# SUPERSEDED: legacy successful-detection sensitivity analysis

> **Do not cite the notebook or outputs in this directory for the revised manuscript.**

This directory is retained only to preserve the history of the detector-failure review.
It used the frozen 5,528-AI-row detector export and an earlier unpaired
Cliff's-delta workflow. Its claims about invariant directions do not hold under the
cleaned paired estimand.

The authoritative replacement is:

- `../outputs/deep_audit/prelim/` for paired Cochran's Q and exact McNemar results;
- `../outputs/deep_audit/rq1/` for the operational and successful-only RQ1 analyses;
- `../outputs/deep_audit/rq1/rq1_operational_vs_successful_only.csv` for the
  one-to-one comparison of all 135 context contrasts.

The revised sensitivity analysis finds ten direction reversals and nine changes in
adjusted-significance status. All ten reversals occur in Llama/RoBERTa panels, where
some successful-only paired comparisons retain only 11--21 PRs. See
`../deep_audit/README.md` for the current estimands and verification procedure.
