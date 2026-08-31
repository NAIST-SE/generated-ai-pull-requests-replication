#!/usr/bin/env python3
"""B4 prompt-exact linked-issue reuse analysis.

SERVER ONLY. Study inputs are immutable. Outputs are written atomically to a new
versioned directory. Run synthetic tests before this program.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from rouge_score import rouge_scorer
from scipy.stats import rankdata


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
CODE_REPO_ROOT = Path(__file__).resolve().parents[3]
for import_root in [CODE_REPO_ROOT, ANALYSIS_DIR]:
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import paired_stats as PS  # noqa: E402
from deep_audit.server_guard import require_nokogiri  # noqa: E402
from phase3.detection.corrected_rerun.build_corpus import (  # noqa: E402
    select_generation_rows,
)
from phase3.detection.corrected_rerun.spec import (  # noqa: E402
    SOURCE_FILES as CORRECTED_SOURCE_FILES,
    normalise_pr_id,
    verify_frozen_sources,
)
from b4_metrics import (  # noqa: E402
    ISSUE_OPEN,
    extract_issue_block,
    lexical_metrics,
    parse_issue_components,
    represent,
    sha256_text,
    tokenize,
)


PROTOCOL_VERSION = "b4-linked-issue-reuse-v2"
GENERATION_PROTOCOL_VERSION = "b4-corrected-generation-v3.2"
EXPECTED_PRS = 273
EXPECTED_AI = 4902
EXPECTED_HUMAN = 273
EXPECTED_DESCRIPTIONS = EXPECTED_AI + EXPECTED_HUMAN
EXPECTED_EXPOSED_PROMPTS = 1638
CONTEXTS = ["P-1", "P-2", "P-3", "P-4", "P-5", "P-6"]
UNEXPOSED = ["P-1", "P-2", "P-3", "P-4"]
EXPOSED = ["P-5", "P-6"]
MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "gpt-5-mini-2025-08-07",
    "meta-llama/llama-3.1-70b-instruct",
]
REPRESENTATIONS = ["raw_case_sensitive", "conservative_normalized"]
ISSUE_MANIFEST_SHA256 = "bdeb1d70b03c46d9f54e8775bb816d71183d486a5e55d9f76f992ab7938b49dc"
CORRECTED_CORPUS_SHA256 = "07136384785d6d02d17762535ab51a2007a7a47b8c41b77fbb311ce9c8780306"
CORRECTED_CORPUS_MANIFEST_SHA256 = "546ab556c1b7e5030f72d8e86a29d009e31e069fbb98dad613ddd209ebe9c9a9"
EXPECTED_MODEL_CONTEXT_COUNTS = {
    (model, context): (
        267
        if context == "P-4" and model in {MODELS[0], MODELS[1]}
        else EXPECTED_PRS
    )
    for model in MODELS
    for context in CONTEXTS
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_output(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *args], text=True
    ).strip()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def parse_args() -> argparse.Namespace:
    repo_default = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_default)
    parser.add_argument(
        "--issue-manifest",
        type=Path,
        default=Path(
            "phase3/analysis/outputs/tier_b/b4-issue-source-audit-v1/"
            "regeneration_sources.jsonl"
        ),
    )
    parser.add_argument(
        "--corrected-corpus",
        type=Path,
        default=Path(
            "phase3/datasets/temperature_1/run_3_corrected_detection_v1/corpus.csv"
        ),
    )
    parser.add_argument(
        "--corrected-corpus-manifest",
        type=Path,
        default=Path(
            "phase3/datasets/temperature_1/run_3_corrected_detection_v1/"
            "corpus_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("phase3/analysis/outputs/tier_b/b4-linked-issue-reuse-v2"),
    )
    parser.add_argument(
        "--with-bertscore",
        action="store_true",
        help="Run the frozen DeBERTa-xlarge BERTScore supporting analysis on CUDA.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate all frozen inputs, prompt sources, corpus joins, and controls without scoring.",
    )
    parser.add_argument("--bertscore-batch-size", type=int, default=8)
    parser.add_argument("--bertscore-chunk-size", type=int, default=128)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    return parser.parse_args()


def resolve_under(repo_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def verify_auxiliary_inputs(paths: dict[str, tuple[Path, str]]) -> dict[str, str]:
    observed = {}
    for key, (path, expected) in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[key] = sha256_file(path)
        if observed[key] != expected:
            raise RuntimeError(
                f"Frozen input hash mismatch for {key}: expected {expected}, "
                f"observed {observed[key]}"
            )
    return observed


def load_issue_manifest(path: Path) -> tuple[pd.DataFrame, dict]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(records) != EXPECTED_PRS:
        raise RuntimeError(
            f"Expected {EXPECTED_PRS} issue-source records; observed {len(records)}"
        )
    source_rows = []
    artifact_keys = []
    for record in records:
        pr_id = normalise_pr_id(record["pr_id"])
        issue_text = record["prompt_issue_block"]
        if sha256_text(issue_text) != record["prompt_issue_block_sha256"]:
            raise RuntimeError(f"Issue-block hash mismatch for PR {pr_id}")
        issues = record.get("issues", [])
        if not issues:
            raise RuntimeError(f"Issue-source record has no issues for PR {pr_id}")
        reference_orders = [int(issue["reference_order"]) for issue in issues]
        if reference_orders != sorted(reference_orders) or len(set(reference_orders)) != len(reference_orders):
            raise RuntimeError(f"Issue reference order is unstable for PR {pr_id}")
        for issue in issues:
            if "/issues/" not in issue["url"] or "/pull/" in issue["url"]:
                raise RuntimeError(f"Non-issue artifact in source record for PR {pr_id}")
            if sha256_text(issue["title"]) != issue["title_sha256"]:
                raise RuntimeError(f"Issue-title hash mismatch for PR {pr_id}")
            if sha256_text(issue["body"]) != issue["body_sha256"]:
                raise RuntimeError(f"Issue-body hash mismatch for PR {pr_id}")
            artifact_keys.append(issue["url"])
        components = parse_issue_components(issue_text)
        if components.document_count != len(issues):
            raise RuntimeError(f"Rendered issue count mismatch for PR {pr_id}")
        source_rows.append(
            {
                "repository": record["repository"],
                "pr_id": pr_id,
                "issue_document_id": f"prompt-{sha256_text(issue_text)[:16]}",
                "issue_text": issue_text,
                "issue_sha256": sha256_text(issue_text),
                "issue_title": components.title,
                "issue_title_sha256": sha256_text(components.title),
                "issue_body": components.body,
                "issue_body_sha256": sha256_text(components.body),
                "rendered_issue_count": components.document_count,
                "component_parse_mode": components.parse_mode,
                "prompt_rows_verified": 0,
            }
        )
    sources = pd.DataFrame(source_rows)
    if sources["pr_id"].duplicated().any() or sources["pr_id"].nunique() != EXPECTED_PRS:
        raise RuntimeError("Issue-source manifest does not contain 273 unique PRs")
    block_counts = sources["issue_sha256"].value_counts()
    return sources, {
        "issue_source_records": len(sources),
        "genuine_issue_artifacts": len(artifact_keys),
        "unique_issue_artifacts": len(set(artifact_keys)),
        "unique_rendered_issue_blocks": int(sources["issue_sha256"].nunique()),
        "duplicate_rendered_block_groups": int((block_counts > 1).sum()),
        "maximum_rendered_block_multiplicity": int(block_counts.max()),
        "rendered_issue_count_distribution": {
            str(key): int(value)
            for key, value in sources["rendered_issue_count"].value_counts().sort_index().items()
        },
    }


def load_corrected_generations(
    repo_root: Path,
    sources: pd.DataFrame,
) -> tuple[pd.DataFrame, dict, dict[str, str]]:
    frozen_generation_hashes = verify_frozen_sources(repo_root)
    completion_path = repo_root / CORRECTED_SOURCE_FILES["corrected_completion"]["path"]
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "complete_and_verified":
        raise RuntimeError("Corrected generation completion status is not complete_and_verified")
    if completion.get("protocol_version") != GENERATION_PROTOCOL_VERSION:
        raise RuntimeError("Unexpected corrected generation protocol version")
    selected = select_generation_rows(repo_root)
    eligible_ids = set(sources["pr_id"])
    raw = selected[selected["pr_id"].isin(eligible_ids)].copy()
    if len(raw) != EXPECTED_AI:
        raise RuntimeError(f"Expected {EXPECTED_AI} eligible AI rows; observed {len(raw)}")
    observed_counts = raw.groupby(["model", "variation"]).size().to_dict()
    if observed_counts != EXPECTED_MODEL_CONTEXT_COUNTS:
        raise RuntimeError(
            f"Eligible model/context counts differ from the frozen specification: {observed_counts}"
        )
    if raw.duplicated(["pr_id", "model", "variation"]).any():
        raise RuntimeError("Corrected generations contain duplicate PR/model/context keys")
    if not raw["finish_reason"].eq("stop").all():
        raise RuntimeError("An eligible generation has a non-stop finish reason")
    if set(raw["model"]) != set(MODELS) or set(raw["variation"]) != set(CONTEXTS):
        raise RuntimeError("Corrected generations have an unexpected model/context set")
    unexposed = raw[raw["variation"].isin(UNEXPOSED)]
    if unexposed["full_prompt"].str.contains(ISSUE_OPEN, regex=False).any():
        raise RuntimeError("An issue-unexposed prompt contains an <issues> block")

    exposed = raw[raw["variation"].isin(EXPOSED)].copy()
    if len(exposed) != EXPECTED_EXPOSED_PROMPTS:
        raise RuntimeError(
            f"Expected {EXPECTED_EXPOSED_PROMPTS} P-5/P-6 prompts; observed {len(exposed)}"
        )
    exposed["issue_text"] = exposed["full_prompt"].map(extract_issue_block)
    exposed["issue_sha256"] = exposed["issue_text"].map(sha256_text)
    per_model_context = exposed.groupby(["pr_id", "model"])["issue_sha256"].nunique()
    if not per_model_context.eq(1).all():
        raise RuntimeError("P-5 and P-6 issue blocks differ within at least one PR/model")
    per_pr = exposed.groupby("pr_id")["issue_sha256"].nunique()
    if len(per_pr) != EXPECTED_PRS or not per_pr.eq(1).all():
        raise RuntimeError("Prompt-exact issue blocks do not agree across models for every PR")
    observed_source = exposed.groupby("pr_id")["issue_sha256"].first()
    expected_source = sources.set_index("pr_id")["issue_sha256"]
    aligned_sources = pd.concat(
        [expected_source.rename("expected"), observed_source.rename("observed")],
        axis=1,
        join="outer",
    )
    if aligned_sources.isna().any().any() or not aligned_sources["expected"].eq(
        aligned_sources["observed"]
    ).all():
        raise RuntimeError("Corrected prompt issue blocks differ from the frozen issue manifest")
    sources["prompt_rows_verified"] = sources["pr_id"].map(
        exposed.groupby("pr_id").size()
    )
    if not sources["prompt_rows_verified"].eq(6).all():
        raise RuntimeError("Prompt-exact issue source coverage is incomplete")
    diagnostics = {
        "exposed_prompt_rows": len(exposed),
        "eligible_ai_rows": len(raw),
        "unique_prs": raw["pr_id"].nunique(),
        "prompt_hash_mismatches": 0,
        "within_pr_model_source_mismatches": 0,
        "across_model_source_mismatches": 0,
        "manifest_source_mismatches": 0,
        "model_context_counts": {
            f"{model}|{context}": int(count)
            for (model, context), count in sorted(observed_counts.items())
        },
    }
    return raw, diagnostics, frozen_generation_hashes


def load_descriptions(
    corrected_corpus_path: Path,
    raw: pd.DataFrame,
    sources: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    corpus = pd.read_csv(corrected_corpus_path, dtype=str, keep_default_na=False)
    required = {"pr_id", "entry_type", "variation", "model", "text", "text_sha256"}
    missing = required.difference(corpus.columns)
    if missing:
        raise RuntimeError(f"Corrected corpus is missing columns: {sorted(missing)}")
    for column in ["pr_id", "variation", "model", "entry_type"]:
        corpus[column] = corpus[column].str.strip()
    corpus["pr_id"] = corpus["pr_id"].map(normalise_pr_id)
    eligible = corpus[corpus["pr_id"].isin(set(sources["pr_id"]))].copy()
    ai = eligible[eligible["entry_type"].str.lower().eq("ai")].copy()
    human = eligible[eligible["entry_type"].str.lower().eq("human")].copy()
    if len(ai) != EXPECTED_AI or len(human) != EXPECTED_HUMAN:
        raise RuntimeError(
            f"Expected {EXPECTED_AI} AI and {EXPECTED_HUMAN} human rows; "
            f"observed {len(ai)} and {len(human)}"
        )
    if ai.duplicated(["pr_id", "model", "variation"]).any():
        raise RuntimeError("Clean AI data contain duplicate PR/model/context keys")
    if human["pr_id"].duplicated().any() or human["pr_id"].nunique() != EXPECTED_PRS:
        raise RuntimeError("Human corpus must contain one row for each of 308 PRs")

    provenance = ai.merge(
        raw[["pr_id", "model", "variation", "generated_text", "finish_reason"]],
        on=["pr_id", "model", "variation"],
        how="left",
        validate="one_to_one",
    )
    if provenance["generated_text"].isna().any() or provenance["generated_text"].eq("").any():
        raise RuntimeError("A corrected-corpus AI row has no matching raw generation")
    mismatches = int((provenance["text"] != provenance["generated_text"]).sum())
    if mismatches:
        raise RuntimeError(f"Corrected-corpus/raw generated text mismatch for {mismatches} rows")
    if not provenance["finish_reason"].eq("stop").all():
        raise RuntimeError("A retained corrected generation has finish_reason other than stop")
    if not eligible["text_sha256"].eq(eligible["text"].map(sha256_text)).all():
        raise RuntimeError("Corrected corpus contains a description SHA-256 mismatch")

    ai["source_type"] = "AI"
    ai["exposure"] = ai["variation"].map(
        {**{code: "unexposed" for code in UNEXPOSED}, **{code: "issue_exposed" for code in EXPOSED}}
    )
    if ai["exposure"].isna().any() or set(ai["model"]) != set(MODELS):
        raise RuntimeError("Clean AI rows contain an unexpected context or model")
    human["source_type"] = "Human"
    human["model"] = "Human"
    human["variation"] = "HUMAN"
    human["exposure"] = "human_baseline"
    descriptions = pd.concat([ai, human], ignore_index=True)
    descriptions = descriptions.rename(columns={"text": "description_text"})
    descriptions["description_sha256"] = descriptions["description_text"].map(sha256_text)
    descriptions["description_id"] = descriptions.apply(
        lambda row: sha256_text(
            "\x1f".join(
                [row["pr_id"], row["model"], row["variation"], row["description_sha256"]]
            )
        )[:24],
        axis=1,
    )
    if descriptions["description_id"].duplicated().any():
        raise RuntimeError("Deterministic description IDs are not unique")
    descriptions = descriptions.merge(sources, on="pr_id", how="left", validate="many_to_one")
    if descriptions["issue_text"].isna().any():
        raise RuntimeError("At least one description has no prompt-exact issue source")
    return descriptions, {
        "clean_ai_rows": len(ai),
        "human_rows": len(human),
        "description_rows": len(descriptions),
        "generated_text_mismatches": 0,
        "non_stop_clean_rows": 0,
        "corrected_corpus_text_hash_mismatches": 0,
    }


def assign_unrelated_sources(sources: pd.DataFrame) -> pd.DataFrame:
    work = sources[["pr_id", "issue_text", "issue_sha256"]].copy()
    work["normalized_tokens"] = work["issue_text"].map(
        lambda text: len(tokenize(text, "conservative_normalized"))
    )
    matches = []
    for row in work.itertuples(index=False):
        other_prs = work[work["pr_id"] != row.pr_id].copy()
        same_hash_excluded = int(other_prs["issue_sha256"].eq(row.issue_sha256).sum())
        others = other_prs[other_prs["issue_sha256"] != row.issue_sha256].copy()
        if others.empty:
            raise RuntimeError(
                f"No distinct-hash unrelated issue candidate remains for PR {row.pr_id}"
            )
        others["distance"] = (others["normalized_tokens"] - row.normalized_tokens).abs()
        chosen = others.sort_values(["distance", "issue_sha256", "pr_id"]).iloc[0]
        matches.append(
            {
                "pr_id": row.pr_id,
                "unrelated_pr_id": chosen["pr_id"],
                "unrelated_issue_text": chosen["issue_text"],
                "unrelated_issue_sha256": chosen["issue_sha256"],
                "source_token_length": int(row.normalized_tokens),
                "unrelated_token_length": int(chosen["normalized_tokens"]),
                "token_length_distance": int(chosen["distance"]),
                "same_hash_candidates_excluded": same_hash_excluded,
            }
        )
    result = pd.DataFrame(matches)
    if (result["pr_id"] == result["unrelated_pr_id"]).any():
        raise RuntimeError("Unrelated-issue matching selected the true PR")
    source_hashes = sources.set_index("pr_id")["issue_sha256"]
    selected_hashes = result.set_index("pr_id")["unrelated_issue_sha256"]
    if source_hashes.eq(selected_hashes).any():
        raise RuntimeError("Unrelated-issue matching selected identical issue text")
    return result


def score_lexical(
    descriptions: pd.DataFrame,
    unrelated: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    true_rows = []
    component_rows = []
    span_rows = []
    unrelated_rows = []
    unrelated_map = unrelated.set_index("pr_id").to_dict("index")

    for ordinal, row in enumerate(descriptions.itertuples(index=False), start=1):
        if ordinal % 500 == 0:
            print(f"Lexical scoring: {ordinal}/{len(descriptions)}", flush=True)
        base = {
            "repository": row.repository,
            "pr_id": row.pr_id,
            "model": row.model,
            "context_code": row.variation,
            "exposure": row.exposure,
            "source_type": row.source_type,
            "description_id": row.description_id,
            "description_sha256": row.description_sha256,
            "issue_document_id": row.issue_document_id,
            "issue_sha256": row.issue_sha256,
        }
        for representation in REPRESENTATIONS:
            metrics = lexical_metrics(row.issue_text, row.description_text, representation)
            rouge_l = rouge.score(
                represent(row.issue_text, representation),
                represent(row.description_text, representation),
            )["rougeL"]
            true_rows.append(
                {
                    **base,
                    "control_type": "true_issue",
                    "issue_component": "combined",
                    "representation": representation,
                    **{key: value for key, value in metrics.items() if not key.startswith("longest_span_") or key in {"longest_span_tokens", "longest_span_generated_fraction"}},
                    "rouge_l_precision": rouge_l.precision,
                    "rouge_l_recall": rouge_l.recall,
                    "rouge_l_f1": rouge_l.fmeasure,
                }
            )
            span_rows.append(
                {
                    **base,
                    "representation": representation,
                    "source_start": metrics["longest_span_source_start"],
                    "generated_start": metrics["longest_span_generated_start"],
                    "span_tokens": metrics["longest_span_tokens"],
                    "span_text": metrics["longest_span_text"],
                }
            )
            for component_name, component_text in [
                ("title", row.issue_title),
                ("body", row.issue_body),
            ]:
                component_metrics = lexical_metrics(
                    component_text, row.description_text, representation
                )
                component_rows.append(
                    {
                        **base,
                        "control_type": "true_issue",
                        "issue_component": component_name,
                        "representation": representation,
                        "source_sha256": sha256_text(component_text),
                        **{key: value for key, value in component_metrics.items() if not key.startswith("longest_span_") or key in {"longest_span_tokens", "longest_span_generated_fraction"}},
                    }
                )

        unrelated_source = unrelated_map[row.pr_id]
        unrelated_metrics = lexical_metrics(
            unrelated_source["unrelated_issue_text"],
            row.description_text,
            "conservative_normalized",
        )
        unrelated_rows.append(
            {
                **base,
                "control_type": "length_matched_unrelated_issue",
                "issue_component": "combined",
                "representation": "conservative_normalized",
                "control_pr_id": unrelated_source["unrelated_pr_id"],
                "control_issue_sha256": unrelated_source["unrelated_issue_sha256"],
                "token_length_distance": unrelated_source["token_length_distance"],
                **{key: value for key, value in unrelated_metrics.items() if not key.startswith("longest_span_") or key in {"longest_span_tokens", "longest_span_generated_fraction"}},
            }
        )
    return (
        pd.DataFrame(true_rows),
        pd.DataFrame(component_rows),
        pd.DataFrame(span_rows),
        pd.DataFrame(unrelated_rows),
    )


def score_bm25(
    descriptions: pd.DataFrame,
    sources: pd.DataFrame,
    k1: float,
    b: float,
) -> pd.DataFrame:
    corpus = sources.sort_values("pr_id").reset_index(drop=True)
    corpus_ids = corpus["pr_id"].tolist()
    corpus_tokens = corpus["issue_text"].map(
        lambda text: tokenize(text, "conservative_normalized")
    ).tolist()
    if any(not tokens for tokens in corpus_tokens):
        raise RuntimeError("BM25 issue corpus contains an empty token sequence")
    relevant = {pr_id: index for index, pr_id in enumerate(corpus_ids)}
    bm25 = BM25Okapi(corpus_tokens, k1=k1, b=b)
    rows = []
    for row in descriptions.itertuples(index=False):
        query = tokenize(row.description_text, "conservative_normalized")
        if not query:
            raise RuntimeError(f"Empty BM25 query for description {row.description_id}")
        scores = np.asarray(bm25.get_scores(query), dtype=float)
        ranks = rankdata(-scores, method="average")
        target = relevant[row.pr_id]
        rank = float(ranks[target])
        rows.append(
            {
                "repository": row.repository,
                "pr_id": row.pr_id,
                "model": row.model,
                "context_code": row.variation,
                "exposure": row.exposure,
                "source_type": row.source_type,
                "description_id": row.description_id,
                "description_sha256": row.description_sha256,
                "issue_document_id": row.issue_document_id,
                "issue_sha256": row.issue_sha256,
                "query_tokens": len(query),
                "true_issue_score": float(scores[target]),
                "true_issue_rank": rank,
                "reciprocal_rank": 1.0 / rank,
                "recall_at_1": int(rank <= 1),
                "recall_at_5": int(rank <= 5),
                "recall_at_10": int(rank <= 10),
            }
        )
    return pd.DataFrame(rows)


def score_bertscore(
    descriptions: pd.DataFrame,
    batch_size: int,
    chunk_size: int,
) -> tuple[pd.DataFrame, dict]:
    import torch
    from bert_score import BERTScorer

    if not torch.cuda.is_available():
        raise RuntimeError("The frozen B4 BERTScore run requires CUDA")
    model_name = "microsoft/deberta-xlarge-mnli"
    scorer = BERTScorer(
        model_type=model_name,
        rescale_with_baseline=True,
        lang="en",
        device="cuda",
        batch_size=batch_size,
    )
    safe_max_length = 512
    configured_max_length = getattr(
        scorer._tokenizer, "model_max_length", safe_max_length
    )
    if configured_max_length is None or int(configured_max_length) <= 0:
        configured_max_length = safe_max_length
    scorer._tokenizer.model_max_length = min(
        int(configured_max_length), safe_max_length
    )
    if scorer._tokenizer.model_max_length <= 0:
        scorer._tokenizer.model_max_length = safe_max_length

    candidates = descriptions["description_text"].tolist()
    references = descriptions["issue_text"].tolist()
    precision_parts = []
    recall_parts = []
    f1_parts = []
    with torch.no_grad():
        for start in range(0, len(descriptions), chunk_size):
            end = min(start + chunk_size, len(descriptions))
            precision, recall, f1 = scorer.score(
                candidates[start:end], references[start:end]
            )
            precision_parts.extend(precision.detach().cpu().numpy().tolist())
            recall_parts.extend(recall.detach().cpu().numpy().tolist())
            f1_parts.extend(f1.detach().cpu().numpy().tolist())
            print(f"BERTScore: {end}/{len(descriptions)}", flush=True)

    tokenizer = scorer._tokenizer
    candidate_lengths = [
        len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
        for text in candidates
    ]
    source_lengths = [
        len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
        for text in references
    ]
    result = descriptions[
        [
            "repository",
            "pr_id",
            "model",
            "variation",
            "exposure",
            "source_type",
            "description_id",
            "description_sha256",
            "issue_document_id",
            "issue_sha256",
        ]
    ].rename(columns={"variation": "context_code"}).copy()
    result["bertscore_precision"] = precision_parts
    result["bertscore_recall"] = recall_parts
    result["bertscore_f1"] = f1_parts
    result["generated_model_tokens"] = candidate_lengths
    result["source_model_tokens"] = source_lengths
    result["generated_truncated"] = result["generated_model_tokens"] > safe_max_length
    result["source_truncated"] = result["source_model_tokens"] > safe_max_length
    result["pair_truncated"] = result["generated_truncated"] | result["source_truncated"]
    metadata = {
        "model_type": model_name,
        "lang": "en",
        "rescale_with_baseline": True,
        "idf": False,
        "device": "cuda",
        "batch_size": batch_size,
        "chunk_size": chunk_size,
        "tokenizer_max_length": safe_max_length,
        "pair_truncated_count": int(result["pair_truncated"].sum()),
        "generated_truncated_count": int(result["generated_truncated"].sum()),
        "source_truncated_count": int(result["source_truncated"].sum()),
    }
    return result, metadata


def paired_contrast_family(
    frame: pd.DataFrame,
    value_col: str,
    exposed_code: str,
    controls: list[str],
    family: str,
) -> pd.DataFrame:
    ai = frame[frame["source_type"].eq("AI")].copy()
    rows = []
    for model in MODELS:
        model_data = ai[ai["model"].eq(model)]
        exposed = model_data[model_data["context_code"].eq(exposed_code)][
            ["pr_id", value_col]
        ].set_index("pr_id")[value_col]
        for control_code in controls:
            control = model_data[model_data["context_code"].eq(control_code)][
                ["pr_id", value_col]
            ].set_index("pr_id")[value_col]
            aligned = pd.concat(
                [exposed.rename("exposed"), control.rename("control")],
                axis=1,
                join="inner",
            ).dropna()
            result = PS.paired_test(
                aligned["exposed"].to_numpy(), aligned["control"].to_numpy()
            )
            rows.append(
                {
                    "family": family,
                    "metric": value_col,
                    "model": model,
                    "exposed_context": exposed_code,
                    "control_context": control_code,
                    "sign": "positive effect sizes favour exposed_context",
                    "n": result["n"],
                    "median_exposed": result["median_human"],
                    "median_control": result["median_ai"],
                    "cliffs_delta": result["cliffs_delta"],
                    "cliffs_magnitude": result["magnitude"],
                    "rank_biserial": result["rank_biserial"],
                    "wilcoxon_stat": result["wilcoxon_stat"],
                    "p": result["p"],
                }
            )
    return PS.apply_fdr(pd.DataFrame(rows), pcol="p")


def exposed_pair_family(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return paired_contrast_family(
        frame,
        value_col,
        "P-5",
        ["P-6"],
        "p5_vs_p6_exposed_contexts_3_tests",
    )


def summarize(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    groups = ["source_type", "model", "context_code", "exposure"]
    parts = []
    for metric in metrics:
        part = (
            frame.groupby(groups, dropna=False)[metric]
            .agg(count="count", mean="mean", median="median", std="std", min="min", max="max")
            .reset_index()
        )
        part.insert(len(groups), "metric", metric)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def metrics_long(lexical: pd.DataFrame) -> pd.DataFrame:
    id_columns = [
        "repository",
        "pr_id",
        "model",
        "context_code",
        "description_id",
        "issue_document_id",
        "issue_component",
        "representation",
        "source_tokens",
        "generated_tokens",
        "issue_sha256",
        "description_sha256",
        "control_type",
        "source_type",
        "exposure",
    ]
    value_columns = [
        "ngram_3_containment",
        "ngram_5_containment",
        "ngram_8_containment",
        "matched_5gram_token_coverage",
        "longest_span_tokens",
        "longest_span_generated_fraction",
        "rouge_l_precision",
        "rouge_l_recall",
        "rouge_l_f1",
    ]
    long = lexical.melt(
        id_vars=id_columns,
        value_vars=value_columns,
        var_name="metric",
        value_name="score",
    )
    long["metric_version"] = PROTOCOL_VERSION
    long["truncated"] = False
    long["success"] = True
    long["error_type"] = ""
    return long


def write_outputs(
    temp_dir: Path,
    sources: pd.DataFrame,
    unrelated: pd.DataFrame,
    descriptions: pd.DataFrame,
    lexical: pd.DataFrame,
    components: pd.DataFrame,
    spans: pd.DataFrame,
    unrelated_scores: pd.DataFrame,
    bm25: pd.DataFrame,
    bertscore: pd.DataFrame | None,
    statistical_tables: dict[str, pd.DataFrame],
    metadata: dict,
) -> None:
    tables = {
        "prompt_exact_issue_sources.csv": sources,
        "unrelated_issue_matches.csv": unrelated,
        "description_manifest.csv": descriptions.drop(
            columns=["description_text", "issue_text", "issue_title", "issue_body"]
        ),
        "lexical_scores.csv": lexical,
        "metrics_long.csv": metrics_long(lexical),
        "component_scores.csv": components,
        "longest_shared_spans.csv": spans,
        "unrelated_issue_scores.csv": unrelated_scores,
        "bm25_retrieval_scores.csv": bm25,
        "lexical_summary.csv": summarize(
            lexical[lexical["representation"].eq("conservative_normalized")],
            [
                "ngram_3_containment",
                "ngram_5_containment",
                "ngram_8_containment",
                "matched_5gram_token_coverage",
                "longest_span_generated_fraction",
                "rouge_l_f1",
            ],
        ),
        "bm25_summary.csv": summarize(
            bm25, ["reciprocal_rank", "recall_at_1", "recall_at_5", "recall_at_10"]
        ),
        **{f"stats_{name}.csv": frame for name, frame in statistical_tables.items()},
    }
    if bertscore is not None:
        tables["bertscore_scores.csv"] = bertscore
        tables["bertscore_summary.csv"] = summarize(
            bertscore,
            ["bertscore_precision", "bertscore_recall", "bertscore_f1", "pair_truncated"],
        )

    normalized = lexical[lexical["representation"].eq("conservative_normalized")]
    top_keys = (
        normalized.sort_values(
            ["model", "context_code", "ngram_5_containment", "longest_span_tokens"],
            ascending=[True, True, False, False],
        )
        .groupby(["model", "context_code"], sort=False)
        .head(10)[["description_id", "ngram_5_containment", "longest_span_tokens"]]
    )
    inspection = top_keys.merge(
        descriptions[
            [
                "description_id",
                "pr_id",
                "model",
                "variation",
                "description_text",
                "issue_text",
            ]
        ],
        on="description_id",
        how="left",
        validate="one_to_one",
    ).merge(
        spans[spans["representation"].eq("conservative_normalized")][
            ["description_id", "span_text"]
        ],
        on="description_id",
        how="left",
        validate="one_to_one",
    )
    tables["high_overlap_inspection_sample.csv"] = inspection

    for filename, frame in tables.items():
        frame.to_csv(temp_dir / filename, index=False)
    (temp_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    require_nokogiri()
    args = parse_args()
    repo_root = args.repo_root.resolve()
    issue_manifest_path = resolve_under(repo_root, args.issue_manifest)
    corrected_corpus_path = resolve_under(repo_root, args.corrected_corpus)
    corrected_corpus_manifest_path = resolve_under(
        repo_root, args.corrected_corpus_manifest
    )
    output_dir = resolve_under(CODE_REPO_ROOT, args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing B4 output: {output_dir}")
    if args.bertscore_batch_size < 1 or args.bertscore_chunk_size < 1:
        raise ValueError("BERTScore batch/chunk sizes must be positive")

    auxiliary_paths = {
        "issue_manifest": (issue_manifest_path, ISSUE_MANIFEST_SHA256),
        "corrected_corpus": (corrected_corpus_path, CORRECTED_CORPUS_SHA256),
        "corrected_corpus_manifest": (
            corrected_corpus_manifest_path,
            CORRECTED_CORPUS_MANIFEST_SHA256,
        ),
    }
    source_hashes_before = verify_auxiliary_inputs(auxiliary_paths)
    corpus_manifest = json.loads(
        corrected_corpus_manifest_path.read_text(encoding="utf-8")
    )
    if corpus_manifest.get("status") != "corpus_complete_and_verified":
        raise RuntimeError("Corrected corpus manifest is not complete_and_verified")
    if corpus_manifest.get("corpus_sha256") != CORRECTED_CORPUS_SHA256:
        raise RuntimeError("Corrected corpus manifest records an unexpected corpus hash")

    sources, issue_source_diagnostics = load_issue_manifest(issue_manifest_path)
    raw, prompt_diagnostics, generation_hashes_before = load_corrected_generations(
        repo_root, sources
    )
    descriptions, description_diagnostics = load_descriptions(
        corrected_corpus_path, raw, sources
    )
    unrelated = assign_unrelated_sources(sources)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "status": "preflight_passed_no_scores_computed",
                    "input_sha256": {
                        **source_hashes_before,
                        **{
                            f"generation_{key}": value
                            for key, value in generation_hashes_before.items()
                        },
                    },
                    "issue_source_diagnostics": issue_source_diagnostics,
                    "prompt_diagnostics": prompt_diagnostics,
                    "description_diagnostics": description_diagnostics,
                    "unrelated_issue_control": {
                        "same_issue_sha256_excluded": True,
                        "source_prs_with_same_hash_candidate_excluded": int(
                            unrelated["same_hash_candidates_excluded"].gt(0).sum()
                        ),
                        "same_hash_candidate_rows_excluded": int(
                            unrelated["same_hash_candidates_excluded"].sum()
                        ),
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    lexical, components, spans, unrelated_scores = score_lexical(
        descriptions, unrelated
    )
    bm25 = score_bm25(descriptions, sources, args.bm25_k1, args.bm25_b)

    normalized_ai = lexical[
        lexical["representation"].eq("conservative_normalized")
        & lexical["source_type"].eq("AI")
    ].copy()
    stats = {
        "primary_p5_5gram": paired_contrast_family(
            normalized_ai,
            "ngram_5_containment",
            "P-5",
            UNEXPOSED,
            "primary_p5_vs_unexposed_12_tests",
        ),
        "secondary_p6_5gram": paired_contrast_family(
            normalized_ai,
            "ngram_5_containment",
            "P-6",
            UNEXPOSED,
            "secondary_p6_vs_unexposed_12_tests",
        ),
        "p5_vs_p6_5gram": exposed_pair_family(
            normalized_ai, "ngram_5_containment"
        ),
        "secondary_bm25_p5": paired_contrast_family(
            bm25,
            "reciprocal_rank",
            "P-5",
            UNEXPOSED,
            "secondary_bm25_p5_vs_unexposed_12_tests",
        ),
    }

    bertscore = None
    bertscore_metadata = {"executed": False}
    if args.with_bertscore:
        bertscore, bertscore_metadata = score_bertscore(
            descriptions,
            args.bertscore_batch_size,
            args.bertscore_chunk_size,
        )
        bertscore_metadata["executed"] = True
        stats["secondary_bertscore_p5"] = paired_contrast_family(
            bertscore,
            "bertscore_f1",
            "P-5",
            UNEXPOSED,
            "secondary_bertscore_p5_vs_unexposed_12_tests",
        )

    source_hashes_after = verify_auxiliary_inputs(auxiliary_paths)
    generation_hashes_after = verify_frozen_sources(repo_root)
    if (
        source_hashes_before != source_hashes_after
        or generation_hashes_before != generation_hashes_after
    ):
        raise RuntimeError("A frozen input changed during B4 execution")
    input_hashes = {
        **source_hashes_after,
        **{
            f"generation_{key}": value
            for key, value in generation_hashes_after.items()
        },
    }

    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "generation_protocol_version": GENERATION_PROTOCOL_VERSION,
        "started_or_completed_at_utc": utc_now(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "input_repo_root": str(repo_root),
        "input_git_commit": git_output(repo_root, "rev-parse", "HEAD"),
        "input_git_branch": git_output(repo_root, "branch", "--show-current"),
        "analysis_repo_root": str(CODE_REPO_ROOT),
        "analysis_git_commit": git_output(CODE_REPO_ROOT, "rev-parse", "HEAD"),
        "analysis_git_branch": git_output(CODE_REPO_ROOT, "branch", "--show-current"),
        "analysis_git_status_porcelain": git_output(
            CODE_REPO_ROOT, "status", "--porcelain"
        ),
        "input_sha256": input_hashes,
        "context_exposure": {
            "issue_exposed": EXPOSED,
            "issue_unexposed": UNEXPOSED,
        },
        "generation_session_boundary": {
            "qwen_gpt": "unchanged run-3 P-1--P-4 controls versus corrected-v3.2 P-5/P-6",
            "llama": "corrected-v3.2 P-1--P-6 under the pinned CoreWeave route",
            "interpretation": "information-transfer robustness analysis, not an isolated causal context effect",
        },
        "primary_outcome": "conservative_normalized occurrence-level generated 5-gram containment",
        "issue_source_diagnostics": issue_source_diagnostics,
        "prompt_diagnostics": prompt_diagnostics,
        "description_diagnostics": description_diagnostics,
        "unrelated_issue_control": {
            "selection": "closest normalized-token length from another PR, deterministic SHA-256/PR-ID tie break",
            "same_issue_sha256_excluded": True,
            "source_prs_with_same_hash_candidate_excluded": int(
                unrelated["same_hash_candidates_excluded"].gt(0).sum()
            ),
            "same_hash_candidate_rows_excluded": int(
                unrelated["same_hash_candidates_excluded"].sum()
            ),
        },
        "bm25": {
            "implementation": "rank_bm25.BM25Okapi",
            "k1": args.bm25_k1,
            "b": args.bm25_b,
            "corpus_documents": len(sources),
            "tie_policy": "average rank",
        },
        "rouge": {"metric": "rougeL", "use_stemmer": False},
        "bertscore": bertscore_metadata,
        "packages": {
            name: package_version(name)
            for name in [
                "pandas",
                "numpy",
                "scipy",
                "statsmodels",
                "rank-bm25",
                "rouge-score",
                "bert-score",
                "torch",
                "transformers",
            ]
        },
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f"{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        write_outputs(
            temp_dir,
            sources,
            unrelated,
            descriptions,
            lexical,
            components,
            spans,
            unrelated_scores,
            bm25,
            bertscore,
            stats,
            metadata,
        )
        output_hashes = {
            path.name: sha256_file(path)
            for path in sorted(temp_dir.iterdir())
            if path.is_file()
        }
        completion = {
            "protocol_version": PROTOCOL_VERSION,
            "status": "complete" if args.with_bertscore else "lexical_bm25_complete_bertscore_pending",
            "completed_at_utc": utc_now(),
            "counts": {
                "prompt_exact_issue_documents": len(sources),
                "description_rows": len(descriptions),
                "lexical_score_rows": len(lexical),
                "component_score_rows": len(components),
                "bm25_score_rows": len(bm25),
                "bertscore_rows": 0 if bertscore is None else len(bertscore),
            },
            "integrity_gates": {
                "input_hashes_match": True,
                "prompt_md5_match": True,
                "p5_p6_source_match_within_pr_model": True,
                "source_match_across_models": True,
                "source_match_frozen_issue_manifest": True,
                "corrected_corpus_raw_text_match": True,
                "unrelated_control_excludes_identical_issue_hash": True,
                "frozen_inputs_unchanged": True,
            },
            "output_sha256": output_hashes,
        }
        (temp_dir / "completion.json").write_text(
            json.dumps(completion, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_dir, output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print(json.dumps(completion, indent=2, sort_keys=True))
    print(stats["primary_p5_5gram"].to_string(index=False))
    print(f"B4 outputs written to {output_dir}")


if __name__ == "__main__":
    main()
