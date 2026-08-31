# Six-context generation-session bridge report

This no-API diagnostic replaces only P1--P4 with the newer session and holds corrected P5--P6 fixed.
It is a robustness bridge, not a unified-session replication of the full 308-PR study.

## Decision

**P6 remains top-ranked in every current bridge cell; this diagnostic does not trigger a complete rerun.**

The predeclared trigger is whether P6 loses top aggregate rank in any current model-by-metric cell.

## Complete-case coverage

| Model | PRs | Rows per bridge version and metric |
|---|---:|---:|
| gpt | 50 | 300 |
| qwen | 39 | 234 |

## Headline stability

### gpt — bertscore_f1

P6 top status: **retained top rank** (historical bridge: True; current bridge: True).
Current top context(s): Without File Contents. P6 gap from best non-P6 context: 0.01190899.
Within-PR P6 tied-top share: 0.320; strict-top share: 0.320 (n=50).

### gpt — bm25_reciprocal_rank

P6 top status: **retained top rank** (historical bridge: True; current bridge: True).
Current top context(s): Without File Contents. P6 gap from best non-P6 context: 0.03702947.
Within-PR P6 tied-top share: 0.580; strict-top share: 0.100 (n=50).

### gpt — rougeL_fmeasure

P6 top status: **retained top rank** (historical bridge: True; current bridge: True).
Current top context(s): Without File Contents. P6 gap from best non-P6 context: 0.01181914.
Within-PR P6 tied-top share: 0.280; strict-top share: 0.260 (n=50).

### qwen — bertscore_f1

P6 top status: **retained top rank** (historical bridge: True; current bridge: True).
Current top context(s): Without File Contents. P6 gap from best non-P6 context: 0.01126394.
Within-PR P6 tied-top share: 0.231; strict-top share: 0.231 (n=39).

### qwen — bm25_reciprocal_rank

P6 top status: **became top-ranked** (historical bridge: False; current bridge: True).
Current top context(s): Without File Contents. P6 gap from best non-P6 context: 0.07327004.
Within-PR P6 tied-top share: 0.487; strict-top share: 0.026 (n=39).

### qwen — rougeL_fmeasure

P6 top status: **retained top rank** (historical bridge: True; current bridge: True).
Current top context(s): Without File Contents. P6 gap from best non-P6 context: 0.01039259.
Within-PR P6 tied-top share: 0.462; strict-top share: 0.436 (n=39).

## Current bridge rankings

### gpt — bertscore_f1

| Rank | Context | Code | Aggregate |
|---:|---|---|---:|
| 1 | Without File Contents | P-6 | 0.00112743 |
| 2 | File Diff | P-3 | -0.01078155 |
| 3 | File Contents | P-4 | -0.02469844 |
| 4 | Repo Path | P-1 | -0.02924430 |
| 5 | Linked Issue | P-5 | -0.03262780 |
| 6 | PR Title | P-2 | -0.03308865 |

### gpt — bm25_reciprocal_rank

| Rank | Context | Code | Aggregate |
|---:|---|---|---:|
| 1 | Without File Contents | P-6 | 0.53700114 |
| 2 | File Diff | P-3 | 0.49997167 |
| 3 | PR Title | P-2 | 0.47618028 |
| 4 | Linked Issue | P-5 | 0.43475746 |
| 5 | File Contents | P-4 | 0.31086616 |
| 6 | Repo Path | P-1 | 0.28230037 |

### gpt — rougeL_fmeasure

| Rank | Context | Code | Aggregate |
|---:|---|---|---:|
| 1 | Without File Contents | P-6 | 0.11673830 |
| 2 | File Diff | P-3 | 0.10491916 |
| 3 | PR Title | P-2 | 0.10472411 |
| 4 | Linked Issue | P-5 | 0.09776981 |
| 5 | Repo Path | P-1 | 0.09573167 |
| 6 | File Contents | P-4 | 0.09048969 |

### qwen — bertscore_f1

| Rank | Context | Code | Aggregate |
|---:|---|---|---:|
| 1 | Without File Contents | P-6 | -0.02305795 |
| 2 | File Diff | P-3 | -0.03432190 |
| 3 | Repo Path | P-1 | -0.04292946 |
| 4 | Linked Issue | P-5 | -0.06366685 |
| 5 | File Contents | P-4 | -0.06380112 |
| 6 | PR Title | P-2 | -0.08003340 |

### qwen — bm25_reciprocal_rank

| Rank | Context | Code | Aggregate |
|---:|---|---|---:|
| 1 | Without File Contents | P-6 | 0.48509054 |
| 2 | File Diff | P-3 | 0.41182051 |
| 3 | PR Title | P-2 | 0.40637790 |
| 4 | Linked Issue | P-5 | 0.22645390 |
| 5 | File Contents | P-4 | 0.15961690 |
| 6 | Repo Path | P-1 | 0.11018152 |

### qwen — rougeL_fmeasure

| Rank | Context | Code | Aggregate |
|---:|---|---|---:|
| 1 | Without File Contents | P-6 | 0.14693878 |
| 2 | File Diff | P-3 | 0.13654618 |
| 3 | File Contents | P-4 | 0.10714286 |
| 4 | PR Title | P-2 | 0.10638298 |
| 5 | Repo Path | P-1 | 0.10588235 |
| 6 | Linked Issue | P-5 | 0.10526316 |

## Interpretation boundary

P5 and P6 are identical in both bridge versions by design. The diagnostic therefore answers whether the newer P1--P4 results displace P6 in the frozen sample; it does not estimate all six contexts from one newly generated session and does not establish statistical equivalence.
