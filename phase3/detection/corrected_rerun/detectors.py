"""Strict detector adapters for the corrected original-detector rerun."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from .spec import DETECTOR_CONFIGS, sha256_text, stable_json_sha256

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:  # local corpus/unit checks do not need the model dependencies
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


class DetectionError(RuntimeError):
    """Base class for a detector call that did not produce a valid score."""


class RetryableDetectionError(DetectionError):
    """A transient call failure that should be retried."""


class PermanentDetectionError(DetectionError):
    """A malformed input or non-transient response that should fail closed."""


def require_nonempty_entry(entry: Dict[str, Any]) -> None:
    text = entry.get("text")
    if not isinstance(text, str) or not text.strip():
        raise PermanentDetectionError("Detector input text is empty")
    expected = entry.get("text_sha256")
    actual = sha256_text(text)
    if expected != actual:
        raise PermanentDetectionError(
            f"Detector input hash mismatch: expected {expected}, got {actual}"
        )


def parse_zerogpt_payload(payload: Dict[str, Any]) -> Tuple[float, bool]:
    if not isinstance(payload, dict):
        raise RetryableDetectionError("ZeroGPT response is not a JSON object")
    if payload.get("code") != 200:
        raise RetryableDetectionError(f"ZeroGPT response code is {payload.get('code')!r}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RetryableDetectionError("ZeroGPT response has no data object")
    percentage = data.get("fakePercentage")
    is_human_score = data.get("isHuman")
    if isinstance(percentage, bool) or not isinstance(percentage, (int, float)):
        raise RetryableDetectionError("ZeroGPT fakePercentage is missing or non-numeric")
    if not 0 <= float(percentage) <= 100:
        raise RetryableDetectionError("ZeroGPT fakePercentage is outside [0, 100]")
    if isinstance(is_human_score, bool) or not isinstance(
        is_human_score, (int, float)
    ):
        raise RetryableDetectionError("ZeroGPT isHuman is missing or non-numeric")
    if not 0 <= float(is_human_score) <= 100:
        raise RetryableDetectionError("ZeroGPT isHuman is outside [0, 100]")
    # The service's ordinal verdict scale places AI-leaning categories below 50,
    # 50 at "mixed", and human-leaning categories above 50. Treat mixed as not
    # flagged to preserve a conservative binary operationalisation.
    return float(percentage) / 100.0, float(is_human_score) < 50.0


class ZeroGPTDetector:
    name = "zerogpt"

    def __init__(
        self,
        tokens_file: Path,
        *,
        session: Optional[requests.Session] = None,
        timeout: Tuple[float, float] = (10.0, 90.0),
    ) -> None:
        tokens = [line.strip() for line in tokens_file.read_text().splitlines() if line.strip()]
        if not tokens:
            raise ValueError(f"No ZeroGPT tokens found in {tokens_file}")
        self.tokens = tokens
        self.token_index = 0
        self.session = session or requests.Session()
        self.timeout = timeout
        self.config = DETECTOR_CONFIGS[self.name]
        self.config_sha256 = stable_json_sha256(self.config)

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    def _next_token(self) -> str:
        token = self.tokens[self.token_index % len(self.tokens)]
        self.token_index += 1
        return token

    def score(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        require_nonempty_entry(entry)
        started = time.perf_counter()
        try:
            response = self.session.post(
                self.config["endpoint"],
                headers={
                    "ApiKey": self._next_token(),
                    "Content-Type": "application/json",
                },
                json={"input_text": entry["text"]},
                timeout=self.timeout,
            )
        except (requests.Timeout, requests.ConnectionError) as error:
            raise RetryableDetectionError(str(error)) from error
        except requests.RequestException as error:
            raise RetryableDetectionError(str(error)) from error

        if response.status_code in {408, 425, 429} or response.status_code >= 500:
            raise RetryableDetectionError(f"ZeroGPT HTTP {response.status_code}")
        if response.status_code >= 400:
            raise PermanentDetectionError(f"ZeroGPT HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError as error:
            raise RetryableDetectionError("ZeroGPT returned invalid JSON") from error
        probability, is_ai = parse_zerogpt_payload(payload)
        raw_response = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return {
            "ai_probability": probability,
            "is_ai_generated": is_ai,
            "raw_response": raw_response,
            "raw_response_sha256": sha256_text(raw_response),
            "detector_model": "ZeroGPT proprietary service",
            "detector_revision": str(
                payload.get("version")
                or payload.get("modelVersion")
                or payload.get("model_version")
                or "not exposed by response"
            ),
            "input_mode": "single",
            "input_char_count": len(entry["text"]),
            "input_token_count": None,
            "used_token_count": None,
            "input_truncated": False,
            "question_token_count": None,
            "answer_token_count": None,
            "http_status": response.status_code,
            "response_date_header": response.headers.get("Date", ""),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }


class RobertaDetector:
    """Pinned single-text or question-answer RoBERTa sequence classifier."""

    def __init__(
        self,
        detector_name: str,
        *,
        device: str = "cuda",
        local_files_only: bool = True,
        tokenizer: Any = None,
        model: Any = None,
    ) -> None:
        if detector_name not in DETECTOR_CONFIGS:
            raise ValueError(f"Unknown detector: {detector_name}")
        config = DETECTOR_CONFIGS[detector_name]
        if config.get("kind") != "roberta":
            raise ValueError(f"Not a RoBERTa detector: {detector_name}")
        self.name = detector_name
        self.config = config
        self.config_sha256 = stable_json_sha256(config)
        self.device = device
        if tokenizer is None or model is None:
            if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
                raise ImportError("torch and transformers are required for RoBERTa scoring")
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
            tokenizer = AutoTokenizer.from_pretrained(
                config["model"],
                revision=config["revision"],
                local_files_only=local_files_only,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                config["model"],
                revision=config["revision"],
                local_files_only=local_files_only,
            )
        self.tokenizer = tokenizer
        self.model = model
        self.model.to(device)
        self.model.eval()

    def tokenizer_arguments(self, entry: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        require_nonempty_entry(entry)
        if self.config["input_mode"] == "pair":
            question = entry.get("qa_question")
            if not isinstance(question, str) or not question.strip():
                raise PermanentDetectionError("QA detector question is empty")
            expected = entry.get("qa_question_sha256")
            if expected != sha256_text(question):
                raise PermanentDetectionError("QA detector question hash mismatch")
            return question, entry["text"]
        return entry["text"], None

    def _token_count(self, text: str) -> int:
        encoded = self.tokenizer(text, add_special_tokens=False, truncation=False)
        return len(encoded["input_ids"])

    def _untruncated_count(self, first: str, second: Optional[str]) -> int:
        if second is None:
            encoded = self.tokenizer(first, add_special_tokens=True, truncation=False)
        else:
            encoded = self.tokenizer(
                first, text_pair=second, add_special_tokens=True, truncation=False
            )
        return len(encoded["input_ids"])

    def score_batch(self, entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not entries:
            return []
        pairs = [self.tokenizer_arguments(entry) for entry in entries]
        first = [item[0] for item in pairs]
        second_values = [item[1] for item in pairs]
        started = time.perf_counter()
        if self.config["input_mode"] == "pair":
            encoded = self.tokenizer(
                first,
                text_pair=second_values,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config["max_length"],
            )
        else:
            encoded = self.tokenizer(
                first,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config["max_length"],
            )
        used_counts = encoded["attention_mask"].sum(dim=1).tolist()
        model_inputs = {key: value.to(self.device) for key, value in encoded.items()}
        if torch is None:
            raise RuntimeError("torch is unavailable")
        with torch.inference_mode():
            output = self.model(**model_inputs)
            probabilities = torch.softmax(output.logits, dim=-1).detach().cpu().tolist()
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        results: List[Dict[str, Any]] = []
        for entry, pair, used_count, probs in zip(entries, pairs, used_counts, probabilities):
            if len(probs) != 2:
                raise PermanentDetectionError(
                    f"Expected two classifier logits, received {len(probs)}"
                )
            first_text, second_text = pair
            untruncated = self._untruncated_count(first_text, second_text)
            if second_text is None:
                question_count = None
                answer_count = self._token_count(first_text)
            else:
                question_count = self._token_count(first_text)
                answer_count = self._token_count(second_text)
            probability = float(probs[1])
            raw = {
                "human_probability": float(probs[0]),
                "ai_probability": probability,
                "model": self.config["model"],
                "revision": self.config["revision"],
                "input_mode": self.config["input_mode"],
                "device": self.device,
            }
            raw_response = json.dumps(raw, sort_keys=True)
            results.append(
                {
                    "ai_probability": probability,
                    "is_ai_generated": probability >= 0.5,
                    "raw_response": raw_response,
                    "raw_response_sha256": sha256_text(raw_response),
                    "detector_model": self.config["model"],
                    "detector_revision": self.config["revision"],
                    "input_mode": self.config["input_mode"],
                    "input_char_count": len(entry["text"]),
                    "input_token_count": untruncated,
                    "used_token_count": int(used_count),
                    "input_truncated": untruncated > int(used_count),
                    "question_token_count": question_count,
                    "answer_token_count": answer_count,
                    "http_status": None,
                    "response_date_header": "",
                    "elapsed_ms": round(elapsed_ms / len(entries), 3),
                }
            )
        return results
