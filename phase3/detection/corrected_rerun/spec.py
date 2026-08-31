"""Frozen inputs and detector configurations for corrected detection run v1."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict


RUN_ID = "run_3_corrected_detection_v1"
RUN_SEED = 42
EXPECTED_HUMAN_ROWS = 308
EXPECTED_AI_ROWS = 5322
EXPECTED_TOTAL_ROWS = EXPECTED_HUMAN_ROWS + EXPECTED_AI_ROWS

DEFAULT_OUTPUT_DIR = Path(
    "phase3/datasets/temperature_1/run_3_corrected_detection_v1"
)

SOURCE_FILES: Dict[str, Dict[str, str]] = {
    "selected_prs": {
        "path": "phase3/datasets/selected_prs_sampled.csv",
        "sha256": "d00daddbf1ed5d054bc3d126248673e9487f381f46c3ae64e68af80e1d139a8e",
    },
    "old_qwen": {
        "path": "phase3/datasets/temperature_1/run_3/Qwen_Qwen2.5-72B-Instruct_1771584467.csv",
        "sha256": "64bb1bdce1052aa8315e91cf1aaf7d5415f3d1c59096d443980bb2748e8a1dc0",
    },
    "old_gpt": {
        "path": "phase3/datasets/temperature_1/run_3/chatgpt_results_exp_chatgpt_1771584462.csv",
        "sha256": "3d8a3adf66ce650aef5c938163915eef39f5f867ff6e07e0027c742c38b21234",
    },
    "corrected_completion": {
        "path": "phase3/datasets/temperature_1/b4-corrected-v3/completion.json",
        "sha256": "7d8226f5b9b5c97afe687b82cae66316652546162a21446daec9caade094f6b1",
    },
    "corrected_llama": {
        "path": "phase3/datasets/temperature_1/b4-corrected-v3/meta-llama_llama-3.1-70b-instruct_b4-corrected-v3.csv",
        "sha256": "e9f97ab7e3b3306e701127fb19dc97c94e7c6767d91568a21febe8b629e38783",
    },
    "corrected_qwen": {
        "path": "phase3/datasets/temperature_1/b4-corrected-v3/Qwen_Qwen2.5-72B-Instruct_b4-corrected-v3.csv",
        "sha256": "2213022098b797b6b0b224e2724c50cea8459f5621ba082c444ff3c332df1b6b",
    },
    "corrected_gpt": {
        "path": "phase3/datasets/temperature_1/b4-corrected-v3/chatgpt_results_b4-corrected-v3.csv",
        "sha256": "5edb3e539d44141c72bacf76fa5c4d28fed8ef256dcc909595fe373a7458934d",
    },
}

MODELS = {
    "llama": "meta-llama/llama-3.1-70b-instruct",
    "qwen": "Qwen/Qwen2.5-72B-Instruct",
    "gpt": "gpt-5-mini-2025-08-07",
}

EXPECTED_AI_COUNTS = {
    (MODELS["llama"], "P-1"): 308,
    (MODELS["llama"], "P-2"): 308,
    (MODELS["llama"], "P-3"): 308,
    (MODELS["llama"], "P-4"): 308,
    (MODELS["llama"], "P-5"): 273,
    (MODELS["llama"], "P-6"): 273,
    (MODELS["qwen"], "P-1"): 308,
    (MODELS["qwen"], "P-2"): 308,
    (MODELS["qwen"], "P-3"): 308,
    (MODELS["qwen"], "P-4"): 302,
    (MODELS["qwen"], "P-5"): 273,
    (MODELS["qwen"], "P-6"): 273,
    (MODELS["gpt"], "P-1"): 308,
    (MODELS["gpt"], "P-2"): 308,
    (MODELS["gpt"], "P-3"): 308,
    (MODELS["gpt"], "P-4"): 302,
    (MODELS["gpt"], "P-5"): 273,
    (MODELS["gpt"], "P-6"): 273,
}

DETECTOR_CONFIGS: Dict[str, Dict[str, Any]] = {
    "zerogpt": {
        "kind": "zerogpt",
        "endpoint": "https://api.zerogpt.com/api/detect/detectText",
        "request_schema": "input_text",
        "text_policy": "complete text; no pipeline truncation",
        "probability_rule": "fakePercentage / 100",
        "binary_label_rule": "numeric isHuman < 50; mixed (50) is not flagged",
    },
    "roberta": {
        "kind": "roberta",
        "model": "Hello-SimpleAI/chatgpt-detector-roberta",
        "revision": "d2b342c61775d5dd0221808a79983ed3b86ffd86",
        "input_mode": "single",
        "max_length": 512,
        "truncation": True,
        "label_0": "human",
        "label_1": "chatgpt",
    },
    "roberta-qa": {
        "kind": "roberta",
        "model": "Hello-SimpleAI/chatgpt-qa-detector-roberta",
        "revision": "25aad95e922d92631f86164642a3db104deee087",
        "input_mode": "pair",
        "question_field": "qa_question",
        "answer_field": "text",
        "question_policy": "PR title for human and AI descriptions",
        "max_length": 512,
        "truncation": True,
        "label_0": "human",
        "label_1": "chatgpt",
    },
    "roberta-qa-legacy-text-only": {
        "kind": "roberta",
        "model": "Hello-SimpleAI/chatgpt-qa-detector-roberta",
        "revision": "25aad95e922d92631f86164642a3db104deee087",
        "input_mode": "single",
        "status": "optional audit-only compatibility diagnostic; not confirmatory",
        "max_length": 512,
        "truncation": True,
        "label_0": "human",
        "label_1": "chatgpt",
    },
}

CONFIRMATORY_DETECTORS = ("zerogpt", "roberta", "roberta-qa")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_text(payload)


def normalise_pr_id(value: Any) -> str:
    text = str(value).strip()
    return re.sub(r"\.0$", "", text)


def normalise_variation(value: Any) -> str:
    match = re.fullmatch(r"P-?([1-6])", str(value).strip(), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid generation variation: {value!r}")
    return f"P-{match.group(1)}"


def source_path(repo_root: Path, source_name: str) -> Path:
    return repo_root / SOURCE_FILES[source_name]["path"]


def verify_frozen_sources(repo_root: Path) -> Dict[str, str]:
    verified: Dict[str, str] = {}
    for name, item in SOURCE_FILES.items():
        path = repo_root / item["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Frozen source is missing: {path}")
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise ValueError(
                f"Frozen source hash mismatch for {name}: expected {item['sha256']}, got {actual}"
            )
        verified[name] = actual
    return verified
