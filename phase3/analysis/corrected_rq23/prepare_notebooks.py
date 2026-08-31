#!/usr/bin/env python3
"""Prepare isolated corrected-corpus execution copies of the RQ2/RQ3 notebooks."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


NOTEBOOKS = [
    ("analysis_rq2_word_sentence_count.ipynb", "iwadi-main"),
    ("analysis_rq2_reading_ease.ipynb", "iwadi-main"),
    ("analysis_rq2_ttr.ipynb", "iwadi-main"),
    ("analysis_rq2_mtld.ipynb", "iwadi-py312"),
    ("analysis_rq2_lexical_density.ipynb", "iwadi-py312"),
    ("analysis_rq2_rouge.ipynb", "iwadi-main"),
    ("analysis_rq2_bertscore.ipynb", "iwadi-main"),
    ("analysis_rq3_codebert.ipynb", "iwadi-main"),
    ("analysis_rq3_graphcodebert.ipynb", "iwadi-main"),
]

CORPUS_SHA256 = "07136384785d6d02d17762535ab51a2007a7a47b8c41b77fbb311ce9c8780306"
DIFFS_SHA256 = "45c71d0170fc9343d999998ad6515e9b5b8298f63b66215f4110b3911f681d0c"

DEBERTA_REVISION = "5b07a9086c1dbb79981ff7b05b4d1ad83b3af51c"
CODEBERT_REVISION = "3b0952feddeffad0063f274080e3c23d75e7eb39"
GRAPHCODEBERT_REVISION = "2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_lines(source: str) -> list[str]:
    return source.splitlines(keepends=True)


def code_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": hashlib.sha256(source.encode("utf-8")).hexdigest()[:8],
        "metadata": {},
        "outputs": [],
        "source": source_lines(source),
    }


def replace_input_paths(text: str, corpus: Path, diffs: Path) -> str:
    corpus_literal = repr(str(corpus.resolve()))
    diffs_literal = repr(str(diffs.resolve()))
    replacements = {
        "csv_path = '../datasets/temperature_1/run_3_cleaned/detection_results.csv'":
            f"csv_path = {corpus_literal}",
        "csv_path = '../datasets/temperature_1/run_3/detection_results.csv'":
            f"csv_path = {corpus_literal}",
        "csv_path = Path('..') / 'datasets' / 'temperature_1' / 'run_3_cleaned' / 'detection_results.csv'":
            f"csv_path = Path({corpus_literal})",
        "diffs_path = Path('..') / 'datasets' / 'selected_prs_with_diffs_and_file_contents.csv'":
            f"diffs_path = Path({diffs_literal})",
        "N≈5,526": "N=5,322",
        "5834 descriptions": "5630 descriptions",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


BERTSCORE_SETUP = """import os
from pathlib import Path

# The rerun is offline and pinned to the exact cached DeBERTa snapshot recorded by the launcher.
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f"Using device: {device}")

BATCH_SIZE = 8
SAFE_MAX_LENGTH = 512
MODEL_NAME = 'microsoft/deberta-xlarge-mnli'
MODEL_REVISION = os.environ['RQ23_DEBERTA_REVISION']
MODEL_SNAPSHOT = Path(os.environ['RQ23_DEBERTA_SNAPSHOT']).resolve()
BASELINE_PATH = Path(os.environ['RQ23_BERTSCORE_BASELINE']).resolve()

if MODEL_REVISION != '5b07a9086c1dbb79981ff7b05b4d1ad83b3af51c':
    raise RuntimeError(f'Unexpected DeBERTa revision: {MODEL_REVISION}')
if not (MODEL_SNAPSHOT / 'config.json').is_file():
    raise RuntimeError(f'Missing pinned DeBERTa snapshot: {MODEL_SNAPSHOT}')
if not BASELINE_PATH.is_file():
    raise RuntimeError(f'Missing BERTScore rescaling baseline: {BASELINE_PATH}')

scorer = BERTScorer(
    model_type=str(MODEL_SNAPSHOT),
    num_layers=40,
    rescale_with_baseline=True,
    baseline_path=str(BASELINE_PATH),
    lang='en',
    device=device,
    batch_size=BATCH_SIZE,
)
tokenizer_max_len = getattr(scorer._tokenizer, 'model_max_length', SAFE_MAX_LENGTH)
if tokenizer_max_len is None or int(tokenizer_max_len) > 1_000_000:
    scorer._tokenizer.model_max_length = SAFE_MAX_LENGTH
else:
    scorer._tokenizer.model_max_length = min(int(tokenizer_max_len), SAFE_MAX_LENGTH)

print(
    f"BERTScorer initialized (model={MODEL_NAME}, revision={MODEL_REVISION}, "
    f"device={device}, batch_size={BATCH_SIZE}, "
    f"tokenizer_max_length={scorer._tokenizer.model_max_length})."
)
"""


ROUGE_SUMMARY = """# Corrected generated-to-reference summary; human self-comparisons are not a baseline.
from pathlib import Path

row_dir = Path('outputs/row_level')
row_dir.mkdir(parents=True, exist_ok=True)
row_cols = [
    'normalized_pr_id', 'entry_type', 'model', 'variation', 'Context_Name',
    'rouge1_precision', 'rouge1_recall', 'rouge1_fmeasure',
    'rouge2_precision', 'rouge2_recall', 'rouge2_fmeasure',
    'rougeL_precision', 'rougeL_recall', 'rougeL_fmeasure',
]
row_scores = ai_df[row_cols].sort_values(
    ['model', 'variation', 'normalized_pr_id'], kind='stable'
)
row_scores.to_csv(row_dir / 'rouge_metrics.csv', index=False)

summary = (
    row_scores.groupby(['model', 'variation', 'Context_Name'], as_index=False)
    .agg(
        n=('normalized_pr_id', 'size'),
        rouge1_mean=('rouge1_fmeasure', 'mean'),
        rouge1_median=('rouge1_fmeasure', 'median'),
        rouge2_mean=('rouge2_fmeasure', 'mean'),
        rouge2_median=('rouge2_fmeasure', 'median'),
        rougeL_mean=('rougeL_fmeasure', 'mean'),
        rougeL_median=('rougeL_fmeasure', 'median'),
    )
    .rename(columns={'Context_Name': 'context'})
)
summary = summary.sort_values(['model', 'variation'], kind='stable')
best = (
    summary.sort_values(
        ['model', 'rougeL_median', 'rougeL_mean'],
        ascending=[True, False, False],
        kind='stable',
    )
    .groupby('model', as_index=False)
    .first()
)
summary.to_csv('outputs/rq2_rouge_generated_to_reference_summary.csv', index=False)
best.to_csv('outputs/rq2_rouge_best_context.csv', index=False)
display(summary)
display(best)
print('Saved generated-to-reference ROUGE summaries without a human-self baseline.')
"""


BERTSCORE_SUMMARY = """# Corrected generated-to-reference summary; human self-comparisons are not a baseline.
import hashlib
import json
from pathlib import Path
import bert_score

row_dir = Path('outputs/row_level')
row_dir.mkdir(parents=True, exist_ok=True)
row_cols = ['normalized_pr_id', 'entry_type', 'model', 'variation', 'Context_Name', 'bertscore_f1']
row_scores = ai_df[row_cols].sort_values(
    ['model', 'variation', 'normalized_pr_id'], kind='stable'
)
row_scores.to_csv(row_dir / 'bertscore_metrics.csv', index=False)

summary = (
    row_scores.groupby(['model', 'variation', 'Context_Name'], as_index=False)
    .agg(n=('normalized_pr_id', 'size'), mean=('bertscore_f1', 'mean'), median=('bertscore_f1', 'median'))
    .rename(columns={'Context_Name': 'context'})
    .sort_values(['model', 'variation'], kind='stable')
)
best = (
    summary.sort_values(['model', 'median', 'mean'], ascending=[True, False, False], kind='stable')
    .groupby('model', as_index=False)
    .first()
)
summary.to_csv('outputs/rq2_bertscore_generated_to_reference_summary.csv', index=False)
best.to_csv('outputs/rq2_bertscore_best_context.csv', index=False)

baseline_sha = hashlib.sha256(BASELINE_PATH.read_bytes()).hexdigest()
metadata_dir = Path('outputs/model_metadata')
metadata_dir.mkdir(parents=True, exist_ok=True)
(metadata_dir / 'bertscore.json').write_text(
    json.dumps({
        'bert_score_version': bert_score.__version__,
        'model_name': MODEL_NAME,
        'model_revision': MODEL_REVISION,
        'model_snapshot': str(MODEL_SNAPSHOT),
        'num_layers': 40,
        'rescale_with_baseline': True,
        'baseline_path': str(BASELINE_PATH),
        'baseline_sha256': baseline_sha,
        'max_length': SAFE_MAX_LENGTH,
        'batch_size': BATCH_SIZE,
    }, indent=2, sort_keys=True) + '\\n',
    encoding='utf-8',
)
display(summary)
display(best)
print('Saved generated-to-reference BERTScore summaries without a human-self baseline.')
"""


EXPORT_CELLS = {
    "analysis_rq2_word_sentence_count.ipynb": """from pathlib import Path as _ExportPath
_export_dir = _ExportPath('outputs/row_level')
_export_dir.mkdir(parents=True, exist_ok=True)
_export = pd.concat([human_df, ai_df], ignore_index=True)[
    ['normalized_pr_id', 'entry_type', 'model', 'variation', 'Context_Name', 'word_count', 'sentence_count']
].sort_values(['entry_type', 'model', 'variation', 'normalized_pr_id'], kind='stable')
_export.to_csv(_export_dir / 'word_sentence_metrics.csv', index=False)
print(f'Saved {len(_export)} row-level word/sentence records.')
""",
    "analysis_rq2_reading_ease.ipynb": """from pathlib import Path as _ExportPath
_export_dir = _ExportPath('outputs/row_level')
_export_dir.mkdir(parents=True, exist_ok=True)
_export = pd.concat([human_df, ai_df], ignore_index=True)[
    ['normalized_pr_id', 'entry_type', 'model', 'variation', 'Context_Name', 'flesch_kincaid_grade']
].sort_values(['entry_type', 'model', 'variation', 'normalized_pr_id'], kind='stable')
_export.to_csv(_export_dir / 'fkgl_metrics.csv', index=False)
print(f'Saved {len(_export)} row-level FKGL records.')
""",
    "analysis_rq2_ttr.ipynb": """from pathlib import Path as _ExportPath
_export_dir = _ExportPath('outputs/row_level')
_export_dir.mkdir(parents=True, exist_ok=True)
_export = pd.concat([human_df, ai_df], ignore_index=True)[
    ['normalized_pr_id', 'entry_type', 'model', 'variation', 'Context_Name', 'ttr']
].sort_values(['entry_type', 'model', 'variation', 'normalized_pr_id'], kind='stable')
_export.to_csv(_export_dir / 'ttr_metrics.csv', index=False)
print(f'Saved {len(_export)} row-level TTR records.')
""",
    "analysis_rq2_mtld.ipynb": """from pathlib import Path as _ExportPath
_export_dir = _ExportPath('outputs/row_level')
_export_dir.mkdir(parents=True, exist_ok=True)
_export = pd.concat([human_df, ai_df], ignore_index=True)[
    ['normalized_pr_id', 'entry_type', 'model', 'variation', 'Context_Name', 'mtld_original', 'mtld_bidir']
].sort_values(['entry_type', 'model', 'variation', 'normalized_pr_id'], kind='stable')
_export.to_csv(_export_dir / 'mtld_metrics.csv', index=False)
print(f'Saved {len(_export)} row-level MTLD records.')
""",
    "analysis_rq2_lexical_density.ipynb": """from pathlib import Path as _ExportPath
_export_dir = _ExportPath('outputs/row_level')
_export_dir.mkdir(parents=True, exist_ok=True)
_export = pd.concat([human_df, ai_df], ignore_index=True)[
    ['normalized_pr_id', 'entry_type', 'model', 'variation', 'Context_Name', 'lexical_density']
].sort_values(['entry_type', 'model', 'variation', 'normalized_pr_id'], kind='stable')
_export.to_csv(_export_dir / 'lexical_density_metrics.csv', index=False)
print(f'Saved {len(_export)} row-level lexical-density records.')
""",
    "analysis_rq3_codebert.ipynb": """import json as _export_json
from pathlib import Path as _ExportPath
_export_dir = _ExportPath('outputs/row_level')
_export_dir.mkdir(parents=True, exist_ok=True)
_export = combined_df[
    ['normalized_pr_id', 'entry_type', 'model', 'variation', 'Context_Name', 'codebert_similarity']
].sort_values(['entry_type', 'model', 'variation', 'normalized_pr_id'], kind='stable')
_export.to_csv(_export_dir / 'codebert_metrics.csv', index=False)
_meta_dir = _ExportPath('outputs/model_metadata')
_meta_dir.mkdir(parents=True, exist_ok=True)
(_meta_dir / 'codebert.json').write_text(_export_json.dumps({
    'model_name': MODEL_NAME, 'model_revision': MODEL_REVISION,
    'max_length': MAX_LENGTH, 'representation': 'first token of last hidden state',
}, indent=2, sort_keys=True) + '\\n', encoding='utf-8')
print(f'Saved {len(_export)} row-level CodeBERT records.')
""",
    "analysis_rq3_graphcodebert.ipynb": """import json as _export_json
from pathlib import Path as _ExportPath
_export_dir = _ExportPath('outputs/row_level')
_export_dir.mkdir(parents=True, exist_ok=True)
_export = combined_df[
    ['normalized_pr_id', 'entry_type', 'model', 'variation', 'Context_Name', 'graphcodebert_similarity']
].sort_values(['entry_type', 'model', 'variation', 'normalized_pr_id'], kind='stable')
_export.to_csv(_export_dir / 'graphcodebert_metrics.csv', index=False)
_meta_dir = _ExportPath('outputs/model_metadata')
_meta_dir.mkdir(parents=True, exist_ok=True)
(_meta_dir / 'graphcodebert.json').write_text(_export_json.dumps({
    'model_name': MODEL_NAME, 'model_revision': MODEL_REVISION,
    'max_length': MAX_LENGTH, 'representation': 'first token of last hidden state',
    'data_flow_graph_supplied': False,
}, indent=2, sort_keys=True) + '\\n', encoding='utf-8')
print(f'Saved {len(_export)} row-level GraphCodeBERT records.')
""",
}


def pin_code_model(source: str, model_name: str, revision: str) -> str:
    old = (
        f"MODEL_NAME = '{model_name}'\n"
        "MAX_LENGTH = 512\n\n"
        "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n"
        "model = AutoModel.from_pretrained(MODEL_NAME).to(device)\n"
        "model.eval()"
    )
    new = (
        f"MODEL_NAME = '{model_name}'\n"
        f"MODEL_REVISION = '{revision}'\n"
        "MAX_LENGTH = 512\n\n"
        "tokenizer = AutoTokenizer.from_pretrained(\n"
        "    MODEL_NAME, revision=MODEL_REVISION, local_files_only=True\n"
        ")\n"
        "model = AutoModel.from_pretrained(\n"
        "    MODEL_NAME, revision=MODEL_REVISION, local_files_only=True\n"
        ").to(device)\n"
        "model.eval()"
    )
    if old not in source:
        raise RuntimeError(f"Could not find unpinned {model_name} loader cell")
    return source.replace(old, new)


def prepare_notebook(source_path: Path, destination: Path, corpus: Path, diffs: Path) -> dict[str, Any]:
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    name = source_path.name
    replaced_self_baseline = False
    pinned_model = False

    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        source = replace_input_paths(source, corpus, diffs)

        if name == "analysis_rq2_bertscore.ipynb" and "# Initialize BERTScorer" in source:
            source = BERTSCORE_SETUP
            pinned_model = True
        elif name == "analysis_rq3_codebert.ipynb" and "# Load CodeBERT model and tokenizer" in source:
            source = pin_code_model(source, "microsoft/codebert-base", CODEBERT_REVISION)
            pinned_model = True
        elif name == "analysis_rq3_graphcodebert.ipynb" and "# Load GraphCodeBERT model and tokenizer" in source:
            source = pin_code_model(source, "microsoft/graphcodebert-base", GRAPHCODEBERT_REVISION)
            pinned_model = True

        if name == "analysis_rq2_rouge.ipynb" and "# Human baseline (self-vs-self ROUGE)" in source:
            source = ROUGE_SUMMARY
            replaced_self_baseline = True
        elif name == "analysis_rq2_bertscore.ipynb" and "# Compute Human Ground Truth baseline distribution" in source:
            source = BERTSCORE_SUMMARY
            replaced_self_baseline = True

        cell["source"] = source_lines(source)
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    if name in EXPORT_CELLS:
        notebook["cells"].append(code_cell(EXPORT_CELLS[name]))

    rendered = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
    if "run_3_cleaned" in rendered or "run_3/detection_results.csv" in rendered:
        raise RuntimeError(f"Prepared notebook still contains a historical input path: {name}")
    if "self-vs-self" in rendered or "Compute Human Ground Truth baseline distribution" in rendered:
        raise RuntimeError(f"Prepared notebook still contains a human-self baseline: {name}")
    if name in {"analysis_rq2_rouge.ipynb", "analysis_rq2_bertscore.ipynb"} and not replaced_self_baseline:
        raise RuntimeError(f"Did not replace the self-baseline cell in {name}")
    if name in {
        "analysis_rq2_bertscore.ipynb",
        "analysis_rq3_codebert.ipynb",
        "analysis_rq3_graphcodebert.ipynb",
    } and not pinned_model:
        raise RuntimeError(f"Did not pin the transformer model in {name}")

    destination.write_text(rendered, encoding="utf-8")
    return {
        "name": name,
        "source_sha256": sha256(source_path),
        "prepared_sha256": sha256(destination),
        "self_baseline_removed": replaced_self_baseline,
        "transformer_pinned": pinned_model,
        "row_level_export_added": name in EXPORT_CELLS or name in {
            "analysis_rq2_rouge.ipynb", "analysis_rq2_bertscore.ipynb"
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--diffs", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    output_dir = args.output_dir.resolve()
    corpus = args.corpus.resolve()
    diffs = args.diffs.resolve()

    if sha256(corpus) != CORPUS_SHA256:
        raise RuntimeError("Corrected corpus SHA-256 mismatch")
    if sha256(diffs) != DIFFS_SHA256:
        raise RuntimeError("PR diff source SHA-256 mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for name, kernel in NOTEBOOKS:
        source = analysis_dir / name
        destination = output_dir / name
        record = prepare_notebook(source, destination, corpus, diffs)
        record["kernel"] = kernel
        records.append(record)

    for helper in ("paired_stats.py", "assemble_textual_features_pdf.py"):
        shutil.copy2(analysis_dir / helper, output_dir / helper)
    shutil.copy2(analysis_dir / "deep_audit" / "server_guard.py", output_dir / "server_guard.py")

    manifest = {
        "protocol": "corrected-rq23-v1",
        "corpus": str(corpus),
        "corpus_sha256": CORPUS_SHA256,
        "diffs": str(diffs),
        "diffs_sha256": DIFFS_SHA256,
        "transformers": {
            "microsoft/deberta-xlarge-mnli": DEBERTA_REVISION,
            "microsoft/codebert-base": CODEBERT_REVISION,
            "microsoft/graphcodebert-base": GRAPHCODEBERT_REVISION,
        },
        "notebooks": records,
    }
    (output_dir / "preparation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
