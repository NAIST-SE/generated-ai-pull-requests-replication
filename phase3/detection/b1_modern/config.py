"""Frozen candidate configuration for B1 additional-detector scoring.

Binoculars is the active B1 detector. Fast-DetectGPT remains implemented as a
contingency and is not part of the active analysis unless a TSE revision asks
for it. Changing the active detector set requires a documented amendment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


RUN_ID = "b1-modern-detectors-v1"
RUN_SEED = 42
CORPUS_RELATIVE_PATH = Path(
    "phase3/datasets/temperature_1/run_3_corrected_detection_v1/corpus.csv"
)
CORPUS_MANIFEST_RELATIVE_PATH = Path(
    "phase3/datasets/temperature_1/run_3_corrected_detection_v1/corpus_manifest.json"
)
CORPUS_SHA256 = "07136384785d6d02d17762535ab51a2007a7a47b8c41b77fbb311ce9c8780306"
EXPECTED_TOTAL = 5630
EXPECTED_HUMAN = 308
EXPECTED_AI = 5322
SMOKE_COUNT = 10
SMOKE_ABS_TOLERANCE = 1e-6
SMOKE_REL_TOLERANCE = 1e-5
BOOTSTRAP_REPLICATES = 2000

# Immutable receipt values for the collaborator-produced Apple MPS Binoculars run.
# Changing these values requires a protocol amendment and a new accepted import.
EXTERNAL_BINOCULARS = {
    "archive_sha256": "4e46e5bc3e8d5d0577182c94ddc9426d665522346433425039fe57dd2c3cd03a",
    "handoff_manifest_sha256": "95be972b5bf3218643d23dcc9d942942f08847802af7ca5f2d5c2dd50969aee8",
    "handoff_corpus_sha256": "fa3ef2c72bc8f1885f52e1a7073b75e3d643345f41e725e21228015fe9f1d2ab",
    "reidentification_map_sha256": "9497aa6a17fe85c63454e856baeb31c9351e39797f424c9e3bdbcaaf5a6b676b",
    "smoke_archive_sha256": "5f33d1f294e02407cef785de66a90670ce3252738278ef9bb4ee3e74519cb6ac",
    "external_detector_config_sha256": "fa447950652a4615af20e120223bdef25ebe3eb0950d358171cfdd3caab5b601",
    "fork_repository": "https://github.com/raux/Binoculars",
    "fork_commit": "ced95d40b6b819cc6f46d9ddf81baf21741fe618",
    "upstream_commit": "c8ae2f90d50ee696418bc71d8d9e5020e5f9d7b8",
    "runner_sha256": "2ffa5164e8098a197a73653c8c29c5b7390e8a6b82501463e9ed8093acce2591",
    "detector_sha256": "2949f1fe1fefd7230aacd244ebbb19d77d12d92b0147ec4cfeaaf603be3be121",
}

CONTEXT_CODES = ("P-1", "P-2", "P-3", "P-4", "P-5", "P-6")
CONTEXT_LABELS = {
    "P-1": "Repo Path",
    "P-2": "PR Title",
    "P-3": "File Diff",
    "P-4": "File Contents",
    "P-5": "Linked Issue",
    "P-6": "Without File Contents",
}
GENERATOR_MODELS = (
    "Qwen/Qwen2.5-72B-Instruct",
    "gpt-5-mini-2025-08-07",
    "meta-llama/llama-3.1-70b-instruct",
)

ORIGINAL_DETECTION_RESULTS_RELATIVE_PATH = Path(
    "phase3/datasets/temperature_1/run_3_corrected_detection_v1/detection_results.csv"
)
ORIGINAL_DETECTION_RESULTS_SHA256 = (
    "b995b02848b4e7487682379afa22c6b7d6a8a06a404f35dac3ebca60daf9aa91"
)
ORIGINAL_DETECTION_VERIFICATION_RELATIVE_PATH = Path(
    "phase3/analysis/outputs/corrected_detection_v1/verification_report.json"
)
ORIGINAL_DETECTION_VERIFICATION_SHA256 = (
    "6b8bb6cbc056de2f7408352d80730b311bf0eab4fd4e74fa1ce2575e8a4ac074"
)
ORIGINAL_DETECTORS = {
    "zerogpt": {
        "display_name": "ZeroGPT",
        "prefix": "zerogpt",
        "model": "ZeroGPT proprietary service",
        "revision": "not exposed by response",
        "input_mode": "single",
    },
    "roberta": {
        "display_name": "ChatGPT Detector RoBERTa",
        "prefix": "roberta",
        "model": "Hello-SimpleAI/chatgpt-detector-roberta",
        "revision": "d2b342c61775d5dd0221808a79983ed3b86ffd86",
        "input_mode": "single",
    },
    "roberta-qa": {
        "display_name": "ChatGPT-QA Detector RoBERTa",
        "prefix": "roberta-qa",
        "model": "Hello-SimpleAI/chatgpt-qa-detector-roberta",
        "revision": "25aad95e922d92631f86164642a3db104deee087",
        "input_mode": "pair",
    },
}

OFFICIAL_REPOSITORIES = {
    "binoculars": {
        "url": "https://github.com/ahans30/Binoculars.git",
        "commit": "c8ae2f90d50ee696418bc71d8d9e5020e5f9d7b8",
        "directory": "Binoculars",
    },
    "fast-detectgpt": {
        "url": "https://github.com/baoguangsheng/fast-detect-gpt.git",
        "commit": "971b05202bac2bb504d60c0ac0812fea7a8f7c82",
        "directory": "fast-detect-gpt",
    },
}

MODEL_SPECS = {
    "falcon-observer": {
        "model_id": "tiiuae/falcon-7b",
        "revision": "ec89142b67d748a1865ea4451372db8313ada0d8",
    },
    "falcon-performer": {
        "model_id": "tiiuae/falcon-7b-instruct",
        "revision": "8782b5c5d8c9290412416618f36a133653e85285",
    },
    "llama3-sampling": {
        "model_id": "meta-llama/Meta-Llama-3-8B",
        "revision": "8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920",
    },
    "llama3-scoring": {
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "revision": "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
    },
}

DETECTOR_CONFIGS: dict[str, dict[str, Any]] = {
    "binoculars": {
        "display_name": "Binoculars",
        "official_repository": "binoculars",
        "observer": "falcon-observer",
        "performer": "falcon-performer",
        "precision": "bfloat16",
        "max_length": 512,
        "mode": "low-fpr",
        "threshold": 0.8536432310785527,
        "threshold_source": (
            "official Binoculars low-FPR threshold calibrated with the frozen "
            "Falcon-7B/Falcon-7B-Instruct pair"
        ),
        "raw_score": "official Binoculars perplexity/cross-perplexity ratio",
        "analysis_score": "negative raw score; higher values are more AI-like",
        "score_direction": "lower_raw_score_is_more_ai",
        "prediction_rule": "raw_score < threshold",
        "source_model_required": False,
        "faithful_implementation": True,
    },
    "fast-detectgpt": {
        "display_name": "Fast-DetectGPT",
        "official_repository": "fast-detectgpt",
        "sampling_model": "llama3-sampling",
        "scoring_model": "llama3-scoring",
        "precision": "bfloat16",
        "max_length": 8192,
        "calibration": {
            "mu0": 0.1603,
            "sigma0": 1.0791,
            "mu1": 2.4686,
            "sigma1": 2.1582,
        },
        "threshold": 0.5,
        "threshold_source": (
            "official local_infer.py balanced-prior Gaussian calibration for "
            "llama3-8b/llama3-8b-instruct"
        ),
        "raw_score": "official analytic conditional-probability-curvature criterion",
        "analysis_score": "raw criterion; higher values are more AI-like",
        "score_direction": "higher_raw_score_is_more_ai",
        "prediction_rule": "calibrated_score >= 0.5",
        "source_model_required": False,
        "faithful_implementation": True,
    },
}

DETECTOR_NAMES = tuple(DETECTOR_CONFIGS)
ANALYSIS_DETECTOR_NAMES = ("binoculars",)
CONTINGENCY_DETECTOR_NAMES = ("fast-detectgpt",)

MANIFEST_COLUMNS = (
    "description_id",
    "pr_id",
    "entry_type",
    "model",
    "context_code",
    "run",
    "text_sha256",
    "char_count",
    "word_count",
    "source_corpus_row",
    "scoring_order",
)

RESULT_COLUMNS = (
    "description_id",
    "pr_id",
    "entry_type",
    "model",
    "context_code",
    "run",
    "detector_name",
    "detector_version",
    "sampling_model",
    "sampling_revision",
    "observer_model",
    "observer_revision",
    "scoring_model",
    "scoring_revision",
    "source_model_required",
    "faithful_implementation",
    "raw_score",
    "calibrated_score",
    "ai_score",
    "predicted_label",
    "threshold_source",
    "input_tokens",
    "used_input_tokens",
    "truncated",
    "success",
    "error_type",
    "error_message",
    "started_at_utc",
    "completed_at_utc",
    "elapsed_ms",
    "input_sha256",
    "corpus_sha256",
    "detector_config_sha256",
    "attempt_number",
)
