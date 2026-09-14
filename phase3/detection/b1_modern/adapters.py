"""Pinned adapters that call the official B1 detector scoring implementations."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from .common import (
    B1PermanentInputError,
    B1ValidationError,
    gaussian_class_probability,
    model_snapshot_paths,
    sha256_text,
    utc_now,
    verify_official_repositories,
)
from .config import DETECTOR_CONFIGS, MODEL_SPECS, OFFICIAL_REPOSITORIES


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise B1ValidationError(f"Could not import official module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_official_interfaces(runtime_root: Path) -> dict[str, str]:
    """Import the exact official entry points without loading model weights."""
    verify_official_repositories(runtime_root)
    binoculars_root = (
        runtime_root / "external" / OFFICIAL_REPOSITORIES["binoculars"]["directory"]
    )
    sys.path.insert(0, str(binoculars_root))
    try:
        from binoculars import Binoculars
    finally:
        sys.path.pop(0)
    if not callable(Binoculars):
        raise B1ValidationError("Pinned Binoculars class is not callable")

    scripts_root = (
        runtime_root
        / "external"
        / OFFICIAL_REPOSITORIES["fast-detectgpt"]["directory"]
        / "scripts"
    )
    sys.path.insert(0, str(scripts_root))
    try:
        official = _load_module(
            "b1_preflight_fast_detect_gpt", scripts_root / "fast_detect_gpt.py"
        )
    finally:
        sys.path.pop(0)
    if not callable(getattr(official, "get_sampling_discrepancy_analytic", None)):
        raise B1ValidationError(
            "Pinned Fast-DetectGPT analytic criterion is unavailable"
        )
    return {
        "binoculars": "binoculars.Binoculars",
        "fast-detectgpt": "fast_detect_gpt.get_sampling_discrepancy_analytic",
    }


class BinocularsAdapter:
    name = "binoculars"

    def __init__(self, runtime_root: Path) -> None:
        try:
            import torch
        except ImportError as error:
            raise B1ValidationError("PyTorch is required for Binoculars") from error
        verify_official_repositories(runtime_root)
        snapshots = model_snapshot_paths(runtime_root)
        official_root = (
            runtime_root / "external" / OFFICIAL_REPOSITORIES["binoculars"]["directory"]
        )
        sys.path.insert(0, str(official_root))
        try:
            from binoculars import Binoculars
        finally:
            sys.path.pop(0)

        config = DETECTOR_CONFIGS[self.name]
        if config["precision"] != "bfloat16":
            raise B1ValidationError("Unexpected Binoculars precision")
        self.config = config
        self.torch = torch
        self.detector = Binoculars(
            observer_name_or_path=str(snapshots[config["observer"]]),
            performer_name_or_path=str(snapshots[config["performer"]]),
            use_bfloat16=True,
            max_token_observed=config["max_length"],
            mode=config["mode"],
        )
        if not math.isclose(
            self.detector.threshold, config["threshold"], rel_tol=0, abs_tol=1e-15
        ):
            raise B1ValidationError(
                "Official Binoculars threshold differs from frozen configuration"
            )

    def score(self, text: str) -> dict[str, Any]:
        started_at = utc_now()
        started = time.perf_counter()
        untruncated = self.detector.tokenizer(
            text, add_special_tokens=True, truncation=False, return_token_type_ids=False
        )["input_ids"]
        used = self.detector.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.config["max_length"],
            return_token_type_ids=False,
        )["input_ids"]
        if len(used) < 2:
            raise B1PermanentInputError("Binoculars input has fewer than two tokens")
        raw_score = float(self.detector.compute_score(text))
        if not math.isfinite(raw_score):
            raise B1PermanentInputError("Binoculars returned a non-finite score")
        predicted_ai = raw_score < self.config["threshold"]
        payload = {
            "raw_score": raw_score,
            "threshold": self.config["threshold"],
            "prediction": "AI" if predicted_ai else "Not AI",
        }
        return {
            "detector_name": self.config["display_name"],
            "detector_version": OFFICIAL_REPOSITORIES["binoculars"]["commit"],
            "sampling_model": "",
            "sampling_revision": "",
            "observer_model": MODEL_SPECS[self.config["observer"]]["model_id"],
            "observer_revision": MODEL_SPECS[self.config["observer"]]["revision"],
            "scoring_model": MODEL_SPECS[self.config["performer"]]["model_id"],
            "scoring_revision": MODEL_SPECS[self.config["performer"]]["revision"],
            "source_model_required": False,
            "faithful_implementation": True,
            "raw_score": raw_score,
            "calibrated_score": "",
            "ai_score": -raw_score,
            "predicted_label": "AI" if predicted_ai else "Not AI",
            "threshold_source": self.config["threshold_source"],
            "input_tokens": len(untruncated),
            "used_input_tokens": len(used),
            "truncated": len(untruncated) > len(used),
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "raw_response_sha256": sha256_text(json.dumps(payload, sort_keys=True)),
        }


class FastDetectGPTAdapter:
    name = "fast-detectgpt"

    def __init__(self, runtime_root: Path, *, device: str = "cuda:0") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise B1ValidationError(
                "PyTorch and transformers are required for Fast-DetectGPT"
            ) from error
        verify_official_repositories(runtime_root)
        snapshots = model_snapshot_paths(runtime_root)
        config = DETECTOR_CONFIGS[self.name]
        scripts_root = (
            runtime_root
            / "external"
            / OFFICIAL_REPOSITORIES["fast-detectgpt"]["directory"]
            / "scripts"
        )
        sys.path.insert(0, str(scripts_root))
        try:
            official = _load_module(
                "b1_official_fast_detect_gpt", scripts_root / "fast_detect_gpt.py"
            )
        finally:
            sys.path.pop(0)
        self.criterion = official.get_sampling_discrepancy_analytic
        self.config = config
        self.device = device
        self.torch = torch
        sampling_path = snapshots[config["sampling_model"]]
        scoring_path = snapshots[config["scoring_model"]]
        self.sampling_tokenizer = AutoTokenizer.from_pretrained(
            sampling_path, local_files_only=True
        )
        self.scoring_tokenizer = AutoTokenizer.from_pretrained(
            scoring_path, local_files_only=True
        )
        if self.sampling_tokenizer.get_vocab() != self.scoring_tokenizer.get_vocab():
            raise B1ValidationError(
                "Fast-DetectGPT sampling/scoring vocabularies differ"
            )
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "local_files_only": True,
        }
        print("Loading frozen Fast-DetectGPT scoring model", flush=True)
        self.scoring_model = AutoModelForCausalLM.from_pretrained(
            scoring_path, **model_kwargs
        )
        self.scoring_model.to(device)
        self.scoring_model.eval()
        print("Loading frozen Fast-DetectGPT sampling model", flush=True)
        self.sampling_model = AutoModelForCausalLM.from_pretrained(
            sampling_path, **model_kwargs
        )
        self.sampling_model.to(device)
        self.sampling_model.eval()

    def _tokenize(self, tokenizer: Any, text: str, *, truncate: bool) -> Any:
        kwargs = {
            "return_tensors": "pt",
            "padding": False,
            "return_token_type_ids": False,
        }
        if truncate:
            kwargs.update(truncation=True, max_length=self.config["max_length"])
        else:
            kwargs.update(truncation=False)
        return tokenizer(text, **kwargs)

    def score(self, text: str) -> dict[str, Any]:
        started_at = utc_now()
        started = time.perf_counter()
        untruncated = self._tokenize(self.scoring_tokenizer, text, truncate=False)
        scoring_tokens = self._tokenize(self.scoring_tokenizer, text, truncate=True).to(
            self.device
        )
        sampling_tokens = self._tokenize(
            self.sampling_tokenizer, text, truncate=True
        ).to(self.device)
        if not self.torch.equal(scoring_tokens.input_ids, sampling_tokens.input_ids):
            raise B1PermanentInputError(
                "Fast-DetectGPT tokenizers produced different input IDs"
            )
        labels = scoring_tokens.input_ids[:, 1:]
        if labels.shape[1] < 1:
            raise B1PermanentInputError(
                "Fast-DetectGPT input has fewer than two tokens"
            )
        with self.torch.inference_mode():
            logits_score = self.scoring_model(**scoring_tokens).logits[:, :-1]
            logits_reference = self.sampling_model(**sampling_tokens).logits[:, :-1]
            raw_score = float(self.criterion(logits_reference, logits_score, labels))
        del logits_score, logits_reference
        if not math.isfinite(raw_score):
            raise B1PermanentInputError(
                "Fast-DetectGPT returned a non-finite criterion"
            )
        probability = gaussian_class_probability(raw_score, self.config["calibration"])
        predicted_ai = probability >= self.config["threshold"]
        payload = {
            "criterion": raw_score,
            "calibrated_probability": probability,
            "prediction": "AI" if predicted_ai else "Not AI",
        }
        input_tokens = int(untruncated.input_ids.shape[1])
        used_tokens = int(labels.shape[1] + 1)
        return {
            "detector_name": self.config["display_name"],
            "detector_version": OFFICIAL_REPOSITORIES["fast-detectgpt"]["commit"],
            "sampling_model": MODEL_SPECS[self.config["sampling_model"]]["model_id"],
            "sampling_revision": MODEL_SPECS[self.config["sampling_model"]]["revision"],
            "observer_model": "",
            "observer_revision": "",
            "scoring_model": MODEL_SPECS[self.config["scoring_model"]]["model_id"],
            "scoring_revision": MODEL_SPECS[self.config["scoring_model"]]["revision"],
            "source_model_required": False,
            "faithful_implementation": True,
            "raw_score": raw_score,
            "calibrated_score": probability,
            "ai_score": raw_score,
            "predicted_label": "AI" if predicted_ai else "Not AI",
            "threshold_source": self.config["threshold_source"],
            "input_tokens": input_tokens,
            "used_input_tokens": used_tokens,
            "truncated": input_tokens > used_tokens,
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "raw_response_sha256": sha256_text(json.dumps(payload, sort_keys=True)),
        }


def create_adapter(detector: str, runtime_root: Path, *, device: str) -> Any:
    if detector == "binoculars":
        return BinocularsAdapter(runtime_root)
    if detector == "fast-detectgpt":
        return FastDetectGPTAdapter(runtime_root, device=device)
    raise B1ValidationError(f"Unknown B1 detector: {detector}")
