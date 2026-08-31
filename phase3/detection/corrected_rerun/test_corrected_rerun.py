from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from .detectors import RetryableDetectionError, RobertaDetector, ZeroGPTDetector, parse_zerogpt_payload
from .finalize import parse_strict_bool
from .run_detector import load_attempts, verify_cached_roberta_assets
from .spec import DETECTOR_CONFIGS, normalise_pr_id, normalise_variation, sha256_text


class FakeResponse:
    status_code = 200
    headers = {"Date": "test-date"}

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return FakeResponse(self.payload)


def test_variation_normalization_is_explicit():
    assert normalise_variation("P1") == "P-1"
    assert normalise_variation("P-6") == "P-6"
    with pytest.raises(ValueError):
        normalise_variation("original")


def test_pr_id_normalization_does_not_change_integer_text():
    assert normalise_pr_id("22") == "22"
    assert normalise_pr_id("22.0") == "22"


def test_zerogpt_parser_requires_score_and_numeric_human_verdict():
    assert parse_zerogpt_payload(
        {"code": 200, "data": {"fakePercentage": 37.5, "isHuman": 62.5}}
    ) == (0.375, False)
    assert parse_zerogpt_payload(
        {"code": 200, "data": {"fakePercentage": 80, "isHuman": 25}}
    ) == (0.8, True)
    assert parse_zerogpt_payload(
        {"code": 200, "data": {"fakePercentage": 50, "isHuman": 50}}
    ) == (0.5, False)
    with pytest.raises(RetryableDetectionError):
        parse_zerogpt_payload({"code": 200, "data": {"isHuman": 100}})
    with pytest.raises(RetryableDetectionError):
        parse_zerogpt_payload(
            {"code": 200, "data": {"fakePercentage": 50, "isHuman": "false"}}
        )
    with pytest.raises(RetryableDetectionError):
        parse_zerogpt_payload(
            {"code": 200, "data": {"fakePercentage": 50, "isHuman": True}}
        )
    with pytest.raises(RetryableDetectionError):
        parse_zerogpt_payload(
            {"code": 200, "data": {"fakePercentage": 50, "isHuman": 101}}
        )


def test_zerogpt_sends_complete_short_text_without_pipeline_truncation(tmp_path):
    token_file = tmp_path / "tokens.txt"
    token_file.write_text("secret\n")
    session = FakeSession(
        {"code": 200, "data": {"fakePercentage": 80, "isHuman": 25}}
    )
    detector = ZeroGPTDetector(token_file, session=session)
    text = "x"
    result = detector.score({"text": text, "text_sha256": sha256_text(text)})
    assert session.calls[0][1]["json"] == {"input_text": text}
    assert result["input_truncated"] is False
    assert result["ai_probability"] == 0.8
    assert result["is_ai_generated"] is True


def test_qa_detector_uses_title_description_pair_even_for_short_text():
    detector = RobertaDetector.__new__(RobertaDetector)
    detector.config = DETECTOR_CONFIGS["roberta-qa"]
    title = "Fix parser regression"
    description = "x"
    entry = {
        "text": description,
        "text_sha256": sha256_text(description),
        "qa_question": title,
        "qa_question_sha256": sha256_text(title),
    }
    assert detector.tokenizer_arguments(entry) == (title, description)


def test_single_detector_accepts_short_nonempty_text():
    detector = RobertaDetector.__new__(RobertaDetector)
    detector.config = DETECTOR_CONFIGS["roberta"]
    text = "x"
    assert detector.tokenizer_arguments(
        {"text": text, "text_sha256": sha256_text(text)}
    ) == (text, None)


def test_resume_rejects_different_corpus_or_configuration(tmp_path):
    attempts = tmp_path / "attempts.jsonl"
    attempts.write_text(
        json.dumps(
            {
                "entry_key": "1_human",
                "success": True,
                "corpus_sha256": "corpus-a",
                "detector_config_sha256": "config-a",
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="different corpus"):
        load_attempts(attempts, corpus_sha256="corpus-b", config_sha256="config-a")
    with pytest.raises(ValueError, match="different detector configuration"):
        load_attempts(attempts, corpus_sha256="corpus-a", config_sha256="config-b")


def test_boolean_parser_does_not_treat_false_string_as_true():
    assert parse_strict_bool("True") is True
    assert parse_strict_bool("False") is False
    with pytest.raises(ValueError):
        parse_strict_bool("not-a-bool")


def test_cache_preflight_requires_every_pinned_asset(monkeypatch, tmp_path):
    existing = tmp_path / "asset"
    existing.write_text("ok")

    def fake_cache(_model, filename, revision):
        assert revision == DETECTOR_CONFIGS["roberta"]["revision"]
        return str(existing) if filename != "pytorch_model.bin" else None

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(try_to_load_from_cache=fake_cache),
    )
    with pytest.raises(FileNotFoundError, match="pytorch_model.bin"):
        verify_cached_roberta_assets(DETECTOR_CONFIGS["roberta"])
