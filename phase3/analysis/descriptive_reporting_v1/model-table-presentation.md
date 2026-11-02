# Keep model names and order consistent

Publication tables use Qwen2.5-72B, GPT-5-mini, and Llama-3.1-70B, in that
order. Full model identifiers remain in the model setup. Frozen statistical
outputs and their historical manifests remain unchanged.

The baseline detector-table producer now follows its existing `MODELS` list,
which already gives this order. Do not rerun that producer to update layout.
Use the presentation formatter with the current publication TeX instead:

```bash
python3 format_model_tables.py \
  --manuscript-dir /path/to/current/manuscript \
  --output-dir /path/to/separate/presentation-output
```

The formatter writes `main.tex` and `tables/prelim_detection_intervals.tex`
under the output directory. It standardizes the RQ1 headers and RQ2 row names,
reorders the RQ3 model rows within each encoder, and orders the baseline model
groups. It computes no metrics, intervals, or tests. Review the resulting diff
before copying either file into the manuscript; compile after copying.

The script checks the expected table labels and model groups and leaves other
manuscript text untouched. Supplementary tables already follow this model order
and naming convention.
