from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from .analyse import (
    _agreement_rows,
    _cluster_bootstrap_ci,
    _paired_cluster_difference_ci,
    _paired_context,
    _paired_detector_difference_rows,
    _performance_rows,
    _read_score_table,
    _rq1_rows,
    _threshold_metrics,
)
from . import import_external_binoculars as external_import
from . import verify_analysis as independent_verifier
from .common import (
    B1ValidationError,
    gaussian_class_probability,
    manifest_row,
    read_jsonl,
    select_smoke_rows,
    sha256_text,
)
from .config import (
    ANALYSIS_DETECTOR_NAMES,
    CONTINGENCY_DETECTOR_NAMES,
    DETECTOR_CONFIGS,
    GENERATOR_MODELS,
    SMOKE_COUNT,
)


def synthetic_row(
    index: int, *, entry_type: str, model: str = "", markup: bool = False
):
    text = (
        f"## Change {index}\n\n- Update `value_{index}` and add a test."
        if markup
        else ("word " * (index + 2)).strip()
    )
    return {
        "corpus_row": str(index),
        "pr_id": str(1000 + index),
        "variation": "original" if entry_type == "human" else f"P-{(index % 6) + 1}",
        "model": "human" if entry_type == "human" else model,
        "entry_type": entry_type,
        "entry_key": f"case-{index}",
        "text": text,
        "text_sha256": sha256_text(text),
        "source_experiment_id": "" if entry_type == "human" else "synthetic-run",
        "scoring_order": str(index),
    }


def test_manifest_hides_human_model_and_normalizes_fields():
    row = manifest_row(synthetic_row(1, entry_type="human"))
    assert row["model"] == ""
    assert row["context_code"] == ""
    assert row["run"] == "human_baseline"
    assert row["text_sha256"] == sha256_text("word word word")


def test_smoke_selection_has_ten_unique_cases_and_all_generators():
    rows = [synthetic_row(i, entry_type="human", markup=i == 3) for i in range(1, 8)]
    next_index = 20
    for model in GENERATOR_MODELS:
        rows.extend(
            synthetic_row(
                next_index + offset, entry_type="ai", model=model, markup=offset == 2
            )
            for offset in range(5)
        )
        next_index += 10
    selected = select_smoke_rows(rows)
    assert len(selected) == SMOKE_COUNT
    assert len({row["description_id"] for row in selected}) == SMOKE_COUNT
    assert {row["model"] for row in selected if row["entry_type"] == "ai"} == set(
        GENERATOR_MODELS
    )
    assert any(row["contains_markdown_or_code"] for row in selected)


def test_fast_detectgpt_stable_probability_matches_balanced_normal_intent():
    params = DETECTOR_CONFIGS["fast-detectgpt"]["calibration"]
    at_human_mean = gaussian_class_probability(params["mu0"], params)
    at_ai_mean = gaussian_class_probability(params["mu1"], params)
    assert 0.0 < at_human_mean < 0.5
    assert 0.5 < at_ai_mean < 1.0
    assert math.isfinite(gaussian_class_probability(100.0, params))


def test_jsonl_reader_rejects_malformed_record(tmp_path):
    path = tmp_path / "attempts.jsonl"
    path.write_text(json.dumps({"success": True}) + "\nnot-json\n", encoding="utf-8")
    with pytest.raises(B1ValidationError, match="Invalid JSONL"):
        read_jsonl(path)


def test_frozen_score_directions_and_thresholds_are_explicit():
    assert (
        DETECTOR_CONFIGS["binoculars"]["score_direction"]
        == "lower_raw_score_is_more_ai"
    )
    assert DETECTOR_CONFIGS["binoculars"]["threshold"] == pytest.approx(
        0.8536432310785527
    )
    assert (
        DETECTOR_CONFIGS["fast-detectgpt"]["score_direction"]
        == "higher_raw_score_is_more_ai"
    )
    assert DETECTOR_CONFIGS["fast-detectgpt"]["threshold"] == 0.5


def test_active_scope_is_binoculars_only_and_fast_detectgpt_is_contingent():
    assert ANALYSIS_DETECTOR_NAMES == ("binoculars",)
    assert CONTINGENCY_DETECTOR_NAMES == ("fast-detectgpt",)


def test_score_reader_preserves_blank_human_metadata(tmp_path):
    path = tmp_path / "scores.csv"
    path.write_text(
        "description_id,pr_id,model,context_code\nhuman-1,1,,\n",
        encoding="utf-8",
    )
    frame = _read_score_table(path)
    assert frame.loc[0, "model"] == ""
    assert frame.loc[0, "context_code"] == ""


def test_threshold_metrics_use_ai_as_positive_class():
    frame = pd.DataFrame(
        {
            "entry_type": ["ai", "ai", "human", "human"],
            "predicted_label": ["AI", "Not AI", "Not AI", "AI"],
        }
    )
    metrics = _threshold_metrics(frame)
    assert metrics["tp"] == metrics["fn"] == metrics["tn"] == metrics["fp"] == 1
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)


def test_threshold_metrics_return_zero_f1_when_no_ai_is_predicted():
    frame = pd.DataFrame(
        {
            "entry_type": ["ai", "ai", "human", "human"],
            "predicted_label": ["Not AI", "Not AI", "Not AI", "Not AI"],
        }
    )
    metrics = _threshold_metrics(frame)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_paired_context_effect_sign_is_context_a_minus_context_b():
    context_a = pd.DataFrame({"pr_id": ["1", "2", "3"], "ai_score": [3, 4, 5]})
    context_b = pd.DataFrame({"pr_id": ["1", "2", "3"], "ai_score": [1, 2, 3]})
    result = _paired_context(context_a, context_b)
    assert result["n"] == 3
    assert result["rank_biserial_a_minus_b"] == pytest.approx(1.0)
    assert "cliffs_delta_a_vs_b" not in result


def test_rq1_builds_one_friedman_and_all_fifteen_pairwise_contrasts():
    rows = []
    for pr_id in range(1, 7):
        for context_number in range(1, 7):
            rows.append(
                {
                    "description_id": f"{pr_id}-{context_number}",
                    "pr_id": str(pr_id),
                    "entry_type": "ai",
                    "model": GENERATOR_MODELS[0],
                    "context_code": f"P-{context_number}",
                    "detector_key": "binoculars",
                    "ai_score": float(context_number + pr_id / 10),
                }
            )
    friedman, pooled, descriptive = _rq1_rows(
        pd.DataFrame(rows), subset="synthetic"
    )
    assert len(friedman) == 1
    assert friedman[0]["n_complete_prs"] == 6
    assert len(pooled) == 15
    assert len(descriptive) == 6
    assert {row["context_code"] for row in descriptive} == {
        f"P-{number}" for number in range(1, 7)
    }


def test_rq1_post_hoc_tests_use_the_same_six_context_complete_prs():
    rows = []
    for pr_id in range(1, 8):
        for context_number in range(1, 7):
            if pr_id == 7 and context_number == 6:
                continue
            rows.append(
                {
                    "description_id": f"{pr_id}-{context_number}",
                    "pr_id": str(pr_id),
                    "entry_type": "ai",
                    "model": GENERATOR_MODELS[0],
                    "context_code": f"P-{context_number}",
                    "detector_key": "binoculars",
                    "ai_score": float(context_number + pr_id / 10),
                }
            )
    friedman, pooled, _ = _rq1_rows(pd.DataFrame(rows), subset="synthetic")
    assert friedman[0]["n_complete_prs"] == 6
    assert {row["n"] for row in pooled} == {6}


def test_cluster_bootstrap_covers_every_primary_threshold_metric():
    frame = pd.DataFrame(
        {
            "pr_id": ["1", "1", "2", "2", "3", "3", "4", "4"],
            "entry_type": ["human", "ai"] * 4,
            "predicted_label": [
                "Not AI",
                "AI",
                "Not AI",
                "Not AI",
                "AI",
                "AI",
                "Not AI",
                "AI",
            ],
            "ai_score": [0.1, 0.9, 0.2, 0.4, 0.8, 0.7, 0.3, 0.85],
        }
    )
    intervals = _cluster_bootstrap_ci(
        frame, replicates=1000, seed=42, include_ranking=True
    )
    reconstructed = independent_verifier._cluster_intervals(
        frame, replicates=1000, seed=42, include_ranking=True
    )
    assert intervals == reconstructed
    for metric in (
        "recall",
        "specificity",
        "precision",
        "f1",
        "accuracy",
        "balanced_accuracy",
        "auroc",
        "auprc",
    ):
        assert 0.0 <= intervals[f"{metric}_ci_low"] <= 1.0
        assert 0.0 <= intervals[f"{metric}_ci_high"] <= 1.0
        assert intervals[f"{metric}_ci_low"] <= intervals[f"{metric}_ci_high"]


def test_paired_modern_existing_comparison_is_pr_clustered(monkeypatch):
    from . import analyse as analysis

    monkeypatch.setattr(analysis, "ANALYSIS_DETECTOR_NAMES", ("modern",))
    monkeypatch.setattr(analysis, "ORIGINAL_DETECTORS", {"existing": {}})
    rows = []
    for detector, human_ai, ai_predictions in (
        ("modern", False, (True, True)),
        ("existing", False, (True, False)),
    ):
        for pr_id, ai_prediction in zip(("1", "2"), ai_predictions):
            rows.append(
                {
                    "description_id": f"human-{pr_id}",
                    "pr_id": pr_id,
                    "entry_type": "human",
                    "model": "",
                    "context_code": "",
                    "detector_key": detector,
                    "predicted_label": "AI" if human_ai else "Not AI",
                }
            )
            rows.append(
                {
                    "description_id": f"ai-{pr_id}",
                    "pr_id": pr_id,
                    "entry_type": "ai",
                    "model": GENERATOR_MODELS[0],
                    "context_code": "P-3",
                    "detector_key": detector,
                    "predicted_label": "AI" if ai_prediction else "Not AI",
                }
            )
    scores = pd.DataFrame(rows)
    recall, specificity = _paired_detector_difference_rows(
        scores, replicates=1000
    )
    assert len(recall) == 3
    assert len(specificity) == 1
    assert recall[0]["recall_difference_modern_minus_existing"] == pytest.approx(
        0.5
    )
    assert specificity[0][
        "specificity_difference_modern_minus_existing"
    ] == pytest.approx(0.0)
    agreement = _agreement_rows(scores)
    assert len(agreement) == 6
    assert all(
        row["both_ai"]
        + row["a_only_ai"]
        + row["b_only_ai"]
        + row["both_not_ai"]
        == row["n_descriptions"]
        for row in agreement
    )
    paired = independent_verifier._pair(scores, "modern", "existing", "ai")
    production_frame = paired.rename(columns={"a": "prediction_a", "b": "prediction_b"})
    assert _paired_cluster_difference_ci(
        production_frame,
        outcome="recall",
        replicates=1000,
        seed=42,
    ) == independent_verifier._paired_interval(
        paired,
        specificity=False,
        replicates=1000,
        seed=42,
    )


def test_performance_rows_include_generator_context_and_threshold_intervals():
    rows = []
    for pr_id in ("1", "2", "3", "4"):
        rows.append(
            {
                "description_id": f"human-{pr_id}",
                "pr_id": pr_id,
                "entry_type": "human",
                "model": "",
                "context_code": "",
                "detector_key": "detector",
                "predicted_label": "Not AI",
                "ai_score": 0.1,
            }
        )
        rows.append(
            {
                "description_id": f"ai-{pr_id}",
                "pr_id": pr_id,
                "entry_type": "ai",
                "model": GENERATOR_MODELS[0],
                "context_code": "P-3",
                "detector_key": "detector",
                "predicted_label": "AI" if pr_id != "4" else "Not AI",
                "ai_score": 0.9 if pr_id != "4" else 0.4,
            }
        )
    performance = _performance_rows(
        pd.DataFrame(rows), subset="synthetic", replicates=1000
    )
    assert len(performance) == 3
    by_level = {row["view_level"]: row for row in performance}
    assert by_level["generator_context"]["context_code"] == "P-3"
    assert by_level["generator_context"]["recall"] == pytest.approx(0.75)
    assert "recall_ci_low" in by_level["generator_context"]
    assert "auroc_ci_low" not in by_level["generator_context"]
    assert "auroc_ci_low" in by_level["pooled"]


def _synthetic_external_row():
    return {
        "record_id": "opaque-1",
        "detector_name": "binoculars",
        "detector_version": "macos-mps",
        "observer_model": "tiiuae/falcon-7b",
        "observer_revision": "cached",
        "performer_model": "tiiuae/falcon-7b-instruct",
        "performer_revision": "cached",
        "precision": "bfloat16",
        "max_length": "512",
        "mode": "low-fpr",
        "threshold": "0.8536432310785527",
        "threshold_source": "official-low-fpr",
        "input_sha256": "a" * 64,
        "data_manifest_sha256": external_import.EXTERNAL_BINOCULARS[
            "handoff_manifest_sha256"
        ],
        "detector_config_sha256": external_import.EXTERNAL_BINOCULARS[
            "external_detector_config_sha256"
        ],
        "attempt_number": "1",
        "started_at_utc": "2026-09-08T00:00:00Z",
        "raw_score": "0.8",
        "ai_score": "-0.8",
        "predicted_label": "AI",
        "input_tokens": "10",
        "used_input_tokens": "10",
        "truncated": "False",
        "success": "True",
        "error_type": "",
        "error_fingerprint": "",
        "completed_at_utc": "2026-09-08T00:00:01Z",
        "elapsed_ms": "1000",
    }


def test_external_provenance_accepts_versioned_inventory_and_frozen_models(tmp_path):
    inventory = tmp_path / "packages.txt"
    inventory.write_text(
        "torch==2.14.0\ntransformers==4.31.0\nnumpy==2.2.6\n",
        encoding="utf-8",
    )
    from .common import sha256_file

    provenance = {
        "status": "accepted_for_analysis",
        "archive_sha256": external_import.EXTERNAL_BINOCULARS["archive_sha256"],
        "handoff_manifest_sha256": external_import.EXTERNAL_BINOCULARS[
            "handoff_manifest_sha256"
        ],
        "smoke_archive_sha256": external_import.EXTERNAL_BINOCULARS[
            "smoke_archive_sha256"
        ],
        "fork_repository": external_import.EXTERNAL_BINOCULARS["fork_repository"],
        "fork_commit": external_import.EXTERNAL_BINOCULARS["fork_commit"],
        "upstream_commit": external_import.EXTERNAL_BINOCULARS["upstream_commit"],
        "runner_sha256": external_import.EXTERNAL_BINOCULARS["runner_sha256"],
        "detector_sha256": external_import.EXTERNAL_BINOCULARS["detector_sha256"],
        "package_inventory_sha256": sha256_file(inventory),
        "observer_revision": external_import.MODEL_SPECS["falcon-observer"][
            "revision"
        ],
        "performer_revision": external_import.MODEL_SPECS["falcon-performer"][
            "revision"
        ],
        "smoke_max_abs_diff": 0.0,
        "hardware": {
            "model": "Mac16,11",
            "chip": "Apple M4 Pro",
            "memory_gb": 64,
            "macos": "26.5.2",
            "build": "25F84",
            "device": "mps",
        },
        "approved_by": "test reviewer",
        "approved_at_utc": "2026-09-14T00:00:00Z",
    }
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps(provenance), encoding="utf-8")
    assert external_import._load_provenance(path, inventory) == provenance


def test_external_binoculars_normalization_preserves_no_raw_text(monkeypatch):
    monkeypatch.setattr(external_import, "EXPECTED_TOTAL", 1)
    monkeypatch.setattr(external_import, "EXPECTED_AI", 1)
    monkeypatch.setattr(external_import, "EXPECTED_HUMAN", 0)
    row = _synthetic_external_row()
    attempt = dict(row)
    attempt.update(
        {
            "attempt_number": 1,
            "threshold": float(row["threshold"]),
            "input_tokens": 10,
            "used_input_tokens": 10,
            "truncated": False,
            "success": True,
            "error_type": None,
            "error_fingerprint": None,
            "elapsed_ms": 1000,
        }
    )
    mapping = {
        "opaque-1": {
            "record_id": "opaque-1",
            "entry_key": "description-1",
            "pr_id": "1",
            "entry_type": "ai",
            "variation": "P-3",
            "model": GENERATOR_MODELS[0],
            "source_experiment_id": "synthetic",
            "text_sha256": "a" * 64,
            "source_corpus_row": "1",
            "scoring_order": "1",
        }
    }
    provenance = {
        "observer_revision": external_import.MODEL_SPECS["falcon-observer"][
            "revision"
        ],
        "performer_revision": external_import.MODEL_SPECS["falcon-performer"][
            "revision"
        ],
    }
    accepted = external_import._validate_external_rows(
        [row], [attempt], mapping, provenance
    )
    assert len(accepted) == 1
    assert accepted[0]["description_id"] == "description-1"
    assert accepted[0]["predicted_label"] == "AI"
    assert "text" not in accepted[0]


def test_external_binoculars_rejects_threshold_label_mismatch(monkeypatch):
    monkeypatch.setattr(external_import, "EXPECTED_TOTAL", 1)
    monkeypatch.setattr(external_import, "EXPECTED_AI", 1)
    monkeypatch.setattr(external_import, "EXPECTED_HUMAN", 0)
    row = _synthetic_external_row()
    row["predicted_label"] = "human"
    attempt = dict(row)
    attempt.update(
        {
            "attempt_number": 1,
            "threshold": float(row["threshold"]),
            "input_tokens": 10,
            "used_input_tokens": 10,
            "truncated": False,
            "success": True,
            "error_type": None,
            "error_fingerprint": None,
            "elapsed_ms": 1000,
        }
    )
    mapping = {
        "opaque-1": {
            "record_id": "opaque-1",
            "entry_key": "description-1",
            "pr_id": "1",
            "entry_type": "ai",
            "variation": "P-3",
            "model": GENERATOR_MODELS[0],
            "source_experiment_id": "synthetic",
            "text_sha256": "a" * 64,
            "source_corpus_row": "1",
            "scoring_order": "1",
        }
    }
    provenance = {
        "observer_revision": external_import.MODEL_SPECS["falcon-observer"][
            "revision"
        ],
        "performer_revision": external_import.MODEL_SPECS["falcon-performer"][
            "revision"
        ],
    }
    with pytest.raises(B1ValidationError, match="threshold-label mismatch"):
        external_import._validate_external_rows([row], [attempt], mapping, provenance)
