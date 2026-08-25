"""
paired_stats.py -- shared statistics helpers for the TSE revision (Tier-A).

Addresses the co-author review:
  * paired human/AI comparisons  -> Wilcoxon signed-rank (not Mann-Whitney U)
  * matched-pairs rank-biserial effect size reported alongside Cliff's delta
  * a single Benjamini-Hochberg (FDR) correction across a test family
  * a real per-cell p-value (replacing the blanket "all Mann-Whitney p < .001")

Why paired: human and AI descriptions are matched by PR (`normalized_pr_id`).
Each PR contributes one human description and, per (model, context), one AI
description grounded in the SAME code change. The pre-revision notebooks fed these
matched arrays to scipy.stats.mannwhitneyu, which assumes *independent* samples and
discards the pairing. Wilcoxon signed-rank on the per-PR differences is the correct
test and is already used elsewhere in the paper (generation-stability analysis).

Sign convention (kept identical to the original notebooks):
  cliffs_delta(human, ai) > 0  ==>  human values tend to exceed AI values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata

try:  # server env (.venv) has statsmodels; keep a tiny fallback for local dry-runs
    from statsmodels.stats.multitest import multipletests
    _HAVE_SM = True
except Exception:  # pragma: no cover
    _HAVE_SM = False


# --------------------------------------------------------------------------- #
# Effect sizes
# --------------------------------------------------------------------------- #
def cliffs_delta(a, b):
    """Cliff's delta (stochastic dominance) of `a` vs `b`.

    Exact, O((m+n) log n) via searchsorted. Matches the O(m*n) reference
    implementation used in the original notebooks:
        delta = ( #(a > b) - #(a < b) ) / (m * n)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return float("nan")
    b_sorted = np.sort(b)
    less = np.searchsorted(b_sorted, a, side="left")          # count of b < x
    greater = n - np.searchsorted(b_sorted, a, side="right")  # count of b > x
    return float((less.sum() - greater.sum()) / (m * n))


def interpret_cliffs_delta(d):
    """Standard Cliff's delta magnitude thresholds (Romano et al.)."""
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return "NA"
    ad = abs(d)
    if ad < 0.147:
        return "Negligible"
    if ad < 0.33:
        return "Small"
    if ad < 0.474:
        return "Medium"
    return "Large"


def matched_pairs_rank_biserial(human, ai):
    """Rank-biserial correlation r_rb for the Wilcoxon signed-rank test.

    r_rb = (R+ - R-) / (R+ + R-), computed on the nonzero paired differences.
    This is the effect size that natively pairs with Wilcoxon. Same sign
    convention as cliffs_delta(human, ai): positive => human > ai.
    Inputs must already be aligned pair-for-pair.
    """
    d = np.asarray(human, float) - np.asarray(ai, float)
    d = d[~np.isnan(d)]
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return float("nan")
    ranks = rankdata(np.abs(d))
    r_pos = ranks[d > 0].sum()
    r_neg = ranks[d < 0].sum()
    total = n * (n + 1) / 2.0
    return float((r_pos - r_neg) / total)


# --------------------------------------------------------------------------- #
# Pairing + test
# --------------------------------------------------------------------------- #
def align_pairs(human_df, ai_df, value_col, key="normalized_pr_id"):
    """Inner-join human and AI rows on `key`, returning aligned numpy arrays.

    If either side has multiple rows per key (should not happen within a single
    model+context slice) the mean is taken. Returns (human_vals, ai_vals) in a
    common, matched order.
    """
    h = human_df.groupby(key)[value_col].mean()
    a = ai_df.groupby(key)[value_col].mean()
    merged = pd.concat([h.rename("h"), a.rename("a")], axis=1, join="inner").dropna()
    return merged["h"].to_numpy(), merged["a"].to_numpy()


def paired_test(human_vals, ai_vals):
    """Wilcoxon signed-rank on aligned pairs, plus effect sizes.

    `human_vals` and `ai_vals` must already be aligned by PR (same order).
    Returns a dict with n, medians, cliffs_delta (+magnitude), rank_biserial,
    the Wilcoxon statistic, and the raw two-sided p-value.
    """
    h = np.asarray(human_vals, float)
    a = np.asarray(ai_vals, float)
    mask = ~(np.isnan(h) | np.isnan(a))
    h, a = h[mask], a[mask]
    n = len(h)

    out = {
        "n": n,
        "median_human": float(np.median(h)) if n else float("nan"),
        "median_ai": float(np.median(a)) if n else float("nan"),
        "cliffs_delta": cliffs_delta(h, a),
        "rank_biserial": matched_pairs_rank_biserial(h, a),
        "wilcoxon_stat": float("nan"),
        "p": 1.0,
    }
    out["magnitude"] = interpret_cliffs_delta(out["cliffs_delta"])

    if n == 0 or np.all(h - a == 0):
        return out
    try:
        stat, p = wilcoxon(h, a)  # paired; zero_method='wilcox' drops zero-diffs
        out["wilcoxon_stat"] = float(stat)
        out["p"] = float(p)
    except ValueError:
        # e.g. all differences zero after dropping -> no signal
        pass
    return out


def paired_test_from_frames(human_df, ai_df, value_col, key="normalized_pr_id"):
    """Convenience: align by `key` then run `paired_test`."""
    h, a = align_pairs(human_df, ai_df, value_col, key=key)
    return paired_test(h, a)


# --------------------------------------------------------------------------- #
# Multiple-comparison correction (apply ONCE per test family)
# --------------------------------------------------------------------------- #
def _bh_fallback(pvals, alpha=0.05):
    """Benjamini-Hochberg without statsmodels (used only if it is unavailable)."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(1, n + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out < alpha, out


def apply_fdr(df, pcol="p", method="fdr_bh", alpha=0.05, out_prefix="adj_"):
    """Add BH-adjusted p-values and a boolean `sig` flag over the whole df.

    The df passed in defines ONE family of tests -- collect every per-cell raw
    p-value into a single frame, then call this once. Rows with NaN p are
    excluded from the correction and flagged not-significant.
    """
    df = df.copy()
    p = df[pcol].to_numpy(dtype=float)
    ok = ~np.isnan(p)
    adj = np.full_like(p, np.nan, dtype=float)
    sig = np.zeros(len(p), dtype=bool)
    if ok.sum() > 0:
        if _HAVE_SM:
            rej, adj_ok, _, _ = multipletests(p[ok], alpha=alpha, method=method)
        else:
            rej, adj_ok = _bh_fallback(p[ok], alpha=alpha)
        adj[ok] = adj_ok
        sig[ok] = rej
    df[out_prefix + pcol] = adj
    df["sig"] = sig
    return df


def apply_fdr_by(df, by, pcol="p", method="fdr_bh", alpha=0.05, out_prefix="adj_"):
    """FDR-correct within each group of `by`, preserving ALL columns.

    Version-robust replacement for ``df.groupby(by).apply(apply_fdr)`` -- pandas
    versions disagree on whether the grouping column survives ``groupby().apply()``
    (newer/py3.12 pandas drops it, which raised ``KeyError: 'Metric'``). Iterating the
    groupby and concatenating keeps every column in all versions.
    """
    parts = []
    for _, g in df.groupby(by, sort=False):
        parts.append(apply_fdr(g, pcol=pcol, method=method, alpha=alpha, out_prefix=out_prefix))
    if not parts:
        return df.copy()
    return pd.concat(parts, ignore_index=True)


def sig_mark(adj_p, alpha=0.05):
    """Compact significance marker for table cells."""
    if adj_p is None or (isinstance(adj_p, float) and np.isnan(adj_p)):
        return "n.s."
    if adj_p < 1e-3:
        return "***"
    if adj_p < 1e-2:
        return "**"
    if adj_p < alpha:
        return "*"
    return "n.s."
