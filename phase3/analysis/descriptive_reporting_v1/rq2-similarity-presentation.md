# Show the verified reference-similarity medians

`format_rq2_similarity.py` reads the hash-pinned ROUGE and BERTScore summary CSVs
from `phase3/analysis/outputs/corrected_rq23_v1/verification/` in the replication
package. It formats ROUGE-L F1 and baseline-rescaled BERTScore F1 medians as a
TeX fragment for panel (b) of the main paper's existing RQ2 table.

```bash
python3 format_rq2_similarity.py \
  --source-dir /path/to/replication/phase3/analysis/outputs/corrected_rq23_v1/verification \
  --output-dir /path/to/separate/presentation-output
```

Copy the reviewed `rq2_similarity_medians.tex` to the manuscript's `tables/`
directory. Load it with `\input{tables/rq2_similarity_medians}` before the table,
then expand `\rqTwoSimilarityMedians` within its tabular environment. Keep the
accompanying provenance JSON with the review records.
The script checks all 18 model/context populations for both metrics. It rounds
medians to three decimal places and bolds the highest unrounded median in each
model/metric row. Bold does not indicate statistical significance.

These medians use each configuration's available outputs: 308 for Repo Path,
PR Title, and Patch; 302 for Qwen/GPT Full Code and 308 for Llama Full Code;
273 for Linked Issue and Multi-source. They are descriptive reference-similarity
scores, distinct from the paired human--AI effects in panel (a). BERTScore is
baseline-rescaled and may be negative; it is not a percentage of human similarity.
No metrics or inferential tests are recomputed, and frozen outputs are unchanged.
