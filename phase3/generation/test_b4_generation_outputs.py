import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from phase3.generation.generation_output_utils import is_successful_generation_record
from phase3.generation.generate_chatgpt import Generator as GPTGenerator
from phase3.generation.generate_openai import (
    apply_p4_truncation,
    partition_tasks,
    process_batch,
)
from phase3.generation.issue_sources import render_issue_prompt_block, sha256_text
from phase3.generation.verify_b4_generation_outputs import (
    GenerationVerificationError,
    LLAMA_P4_FILE_TOKEN_BUDGET,
    LLAMA_TRUNCATED_P4_MAX_TOKENS,
    MODEL_SPECS,
    validate_output,
)


def make_manifest():
    issues = [
        {
            "reference_order": 1,
            "issue_number": 7,
            "url": "https://github.com/owner/repo/issues/7",
            "title": "Issue title",
            "body": "Issue body",
            "title_sha256": sha256_text("Issue title"),
            "body_sha256": sha256_text("Issue body"),
        }
    ]
    block = render_issue_prompt_block(issues)
    return {
        "PR-1": {
            "pr_id": "PR-1",
            "issues": issues,
            "prompt_issue_block": block,
            "prompt_issue_block_sha256": sha256_text(block),
        }
    }


def make_output(tmp_path, label="llama"):
    manifest = make_manifest()
    spec = MODEL_SPECS[label]
    rows = []
    for variation in sorted(spec["variations"]):
        if variation in {"P-5", "P-6"}:
            context = f"<issues>\n{manifest['PR-1']['prompt_issue_block']}\n</issues>"
        else:
            context = f"<control variation='{variation}'>Control context</control>"
        prompt = f"prefix\n{context}\nsuffix"
        rows.append(
            {
                "experiment_id": "exp-test",
                "timestamp": "2026-01-01T00:00:00",
                "model": spec["model"],
                "pr_id": "PR-1",
                "variation": variation,
                "seed": "42",
                "decoding_params": json.dumps(spec["decoding"]),
                "prompt_hash": hashlib.md5(prompt.encode()).hexdigest(),
                "prompt_token_count": "10",
                "generated_token_count": "4",
                "full_prompt": prompt,
                "generated_text": "A valid generated description.",
                "device_map": "api",
                "finish_reason": "stop",
                "provider": spec["provider"],
                "precision": spec["precision"],
                "truncated_token_count": "",
                "system_fingerprint": "fp-test",
            }
        )
    path = tmp_path / f"{label}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path, manifest


def validate(label, path, manifest):
    return validate_output(
        label,
        path,
        manifest,
        {"PR-1"},
        expected_issue_prs=1,
        expected_llama_control_prs=1,
    )


@pytest.mark.parametrize(
    ("label", "expected_rows"),
    [("llama", 6), ("qwen", 2), ("gpt", 2)],
)
def test_valid_output(label, expected_rows, tmp_path):
    path, manifest = make_output(tmp_path, label)
    result = validate(label, path, manifest)
    assert result["rows"] == expected_rows
    assert result["provider"] == MODEL_SPECS[label]["provider"]
    assert result["control_prs"] == (1 if label == "llama" else 0)


def test_llama_is_pinned_to_coreweave():
    assert MODEL_SPECS["llama"]["provider"] == "coreweave"
    assert MODEL_SPECS["llama"]["precision"] == "bf16"


def test_launcher_applies_approved_p4_file_token_budget():
    launcher = (Path(__file__).resolve().parents[2] / "run_generation.sh").read_text()
    assert "b4_llama_p4_file_token_budget=90000" in launcher
    assert "b4_llama_truncated_p4_completion_token_cap=4096" in launcher
    assert launcher.count("--truncate-p4") == 1
    assert '--max-context-tokens "$b4_llama_p4_file_token_budget"' in launcher
    assert (
        '--truncated-max-tokens '
        '"$b4_llama_truncated_p4_completion_token_cap"'
    ) in launcher


def test_p4_truncation_metadata_is_absent_when_content_is_within_budget():
    class WhitespaceTokenizer:
        @staticmethod
        def encode(text):
            return text.split()

    pr_data = {
        "files": [{"filename": "small.py", "file_content": "one two three"}]
    }
    unchanged, truncated_count = apply_p4_truncation(
        pr_data, WhitespaceTokenizer(), max_tokens=10
    )
    assert unchanged == pr_data
    assert truncated_count is None


def test_accepts_approved_p4_truncation_metadata(tmp_path):
    path, manifest = make_output(tmp_path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    p4 = frame["variation"].eq("P-4")
    frame.loc[p4, "truncated_token_count"] = str(LLAMA_P4_FILE_TOKEN_BUDGET)
    decoding = dict(MODEL_SPECS["llama"]["decoding"])
    decoding["max_tokens"] = LLAMA_TRUNCATED_P4_MAX_TOKENS
    frame.loc[p4, "decoding_params"] = json.dumps(decoding)
    frame.to_csv(path, index=False)
    result = validate("llama", path, manifest)
    assert result["p4_truncated_rows"] == 1
    assert (
        result["truncated_p4_completion_token_cap"]
        == LLAMA_TRUNCATED_P4_MAX_TOKENS
    )


def test_rejects_truncated_p4_without_completion_cap(tmp_path):
    path, manifest = make_output(tmp_path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.loc[frame["variation"].eq("P-4"), "truncated_token_count"] = "100"
    frame.to_csv(path, index=False)
    with pytest.raises(GenerationVerificationError, match="decoding parameters differ"):
        validate("llama", path, manifest)


def test_rejects_completion_cap_without_p4_truncation(tmp_path):
    path, manifest = make_output(tmp_path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    decoding = dict(MODEL_SPECS["llama"]["decoding"])
    decoding["max_tokens"] = LLAMA_TRUNCATED_P4_MAX_TOKENS
    frame.loc[frame["variation"].eq("P-4"), "decoding_params"] = json.dumps(decoding)
    frame.to_csv(path, index=False)
    with pytest.raises(GenerationVerificationError, match="decoding parameters differ"):
        validate("llama", path, manifest)


def test_rejects_truncation_metadata_outside_p4(tmp_path):
    path, manifest = make_output(tmp_path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.loc[frame["variation"].eq("P-5"), "truncated_token_count"] = "100"
    frame.to_csv(path, index=False)
    with pytest.raises(GenerationVerificationError, match="outside P4"):
        validate("llama", path, manifest)


def test_rejects_p4_truncation_above_budget(tmp_path):
    path, manifest = make_output(tmp_path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    p4 = frame["variation"].eq("P-4")
    frame.loc[p4, "truncated_token_count"] = str(LLAMA_P4_FILE_TOKEN_BUDGET + 1)
    decoding = dict(MODEL_SPECS["llama"]["decoding"])
    decoding["max_tokens"] = LLAMA_TRUNCATED_P4_MAX_TOKENS
    frame.loc[p4, "decoding_params"] = json.dumps(decoding)
    frame.to_csv(path, index=False)
    with pytest.raises(GenerationVerificationError, match="exceeds"):
        validate("llama", path, manifest)


def test_openrouter_request_disables_fallback_and_requires_coreweave_bf16(tmp_path):
    class Builder:
        @staticmethod
        def construct_context_block(_pr_data, _variation):
            return "control context"

        @staticmethod
        def wrap_context_in_prompt(context):
            return [{"role": "user", "content": context}]

    observed = {}

    def create(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"description": "Valid"}'),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
            system_fingerprint="fp-coreweave-test",
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    args = SimpleNamespace(
        truncate_p4=False,
        max_context_tokens=30_000,
        truncated_max_tokens=LLAMA_TRUNCATED_P4_MAX_TOKENS,
        dry_run=False,
        provider="coreweave",
        quantization=["bf16"],
        model=MODEL_SPECS["llama"]["model"],
        temperature=1,
        max_tokens=None,
        seed=42,
    )
    process_batch(
        [("PR-1", "P-1")],
        {"PR-1": {}},
        client,
        Builder,
        args,
        "exp-test",
        str(tmp_path / "output.csv"),
        "openrouter",
    )
    assert observed["extra_body"] == {
        "provider": {
            "order": ["coreweave"],
            "allow_fallbacks": False,
            "quantizations": ["bf16"],
        }
    }
    assert observed["seed"] == 42
    assert "max_tokens" not in observed
    output = pd.read_csv(tmp_path / "output.csv", dtype=str, keep_default_na=False)
    assert output.loc[0, "system_fingerprint"] == "fp-coreweave-test"
    assert json.loads(output.loc[0, "decoding_params"]) == MODEL_SPECS["llama"][
        "decoding"
    ]


def test_truncated_p4_request_uses_and_records_completion_cap(tmp_path):
    class WhitespaceTokenizer:
        @staticmethod
        def encode(text):
            return text.split()

        @staticmethod
        def decode(tokens):
            return " ".join(tokens)

    class Builder:
        @staticmethod
        def construct_context_block(pr_data, _variation):
            return pr_data["files"][0]["file_content"]

        @staticmethod
        def wrap_context_in_prompt(context):
            return [{"role": "user", "content": context}]

    observed = {}

    def create(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"description": "Valid"}'),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=1),
            system_fingerprint="fp-truncated-test",
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    args = SimpleNamespace(
        truncate_p4=True,
        max_context_tokens=30,
        truncated_max_tokens=LLAMA_TRUNCATED_P4_MAX_TOKENS,
        dry_run=False,
        provider="coreweave",
        quantization=["bf16"],
        model=MODEL_SPECS["llama"]["model"],
        temperature=1,
        max_tokens=None,
        seed=42,
    )
    process_batch(
        [("PR-1", "P-4")],
        {
            "PR-1": {
                "files": [
                    {
                        "filename": "large.py",
                        "file_content": "token " * 100,
                    }
                ]
            }
        },
        client,
        Builder,
        args,
        "exp-test",
        str(tmp_path / "truncated.csv"),
        "openrouter",
        tokenizer=WhitespaceTokenizer(),
    )
    assert observed["max_tokens"] == LLAMA_TRUNCATED_P4_MAX_TOKENS
    output = pd.read_csv(
        tmp_path / "truncated.csv", dtype=str, keep_default_na=False
    )
    assert output.loc[0, "truncated_token_count"]
    assert json.loads(output.loc[0, "decoding_params"])["max_tokens"] == 4096


def test_gpt_request_sends_seed_and_records_fingerprint():
    observed = {}

    def create(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"description": "Valid"}'),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
            system_fingerprint="fp-openai-test",
        )

    generator = GPTGenerator(
        model_name=MODEL_SPECS["gpt"]["model"],
        mock=True,
        seed=42,
    )
    generator.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    result = generator.generate("test prompt")
    assert observed["seed"] == 42
    assert "temperature" not in observed
    assert result["system_fingerprint"] == "fp-openai-test"


def test_rejects_wrong_issue_block(tmp_path):
    path, manifest = make_output(tmp_path)
    frame = pd.read_csv(path)
    issue_row = frame.index[frame["variation"] == "P-5"][0]
    frame.loc[issue_row, "full_prompt"] = frame.loc[issue_row, "full_prompt"].replace(
        "Issue body", "Target PR body"
    )
    frame.loc[issue_row, "prompt_hash"] = hashlib.md5(
        frame.loc[issue_row, "full_prompt"].encode()
    ).hexdigest()
    frame.to_csv(path, index=False)
    with pytest.raises(GenerationVerificationError, match="issue block differs"):
        validate("llama", path, manifest)


def test_rejects_issue_block_in_control_prompt(tmp_path):
    path, manifest = make_output(tmp_path)
    frame = pd.read_csv(path)
    control_row = frame.index[frame["variation"] == "P-1"][0]
    frame.loc[control_row, "full_prompt"] += manifest["PR-1"]["prompt_issue_block"]
    frame.loc[control_row, "prompt_hash"] = hashlib.md5(
        frame.loc[control_row, "full_prompt"].encode()
    ).hexdigest()
    frame.to_csv(path, index=False)
    with pytest.raises(GenerationVerificationError, match="control prompt contains"):
        validate("llama", path, manifest)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("generated_text", "ERROR_GENERATION_FAILED", "failure marker"),
        ("generated_text", "MOCKED DESCRIPTION", "mock output"),
        ("finish_reason", "length", "non-stop finish"),
        ("provider", "fallback-provider", "provider differs"),
        ("generated_token_count", "0", "generated token count invalid"),
    ],
)
def test_rejects_invalid_generation_rows(tmp_path, column, value, message):
    path, manifest = make_output(tmp_path)
    frame = pd.read_csv(path)
    frame[column] = frame[column].astype(str)
    frame.loc[0, column] = value
    frame.to_csv(path, index=False)
    with pytest.raises(GenerationVerificationError, match=message):
        validate("llama", path, manifest)


def test_rejects_missing_context(tmp_path):
    path, manifest = make_output(tmp_path)
    frame = pd.read_csv(path)
    frame = frame[frame["variation"] != "P-6"]
    frame.to_csv(path, index=False)
    with pytest.raises(GenerationVerificationError, match="context set differs"):
        validate("llama", path, manifest)


def test_rejects_partial_control_pr_set_that_differs_from_issue_set(tmp_path):
    path, manifest = make_output(tmp_path)
    frame = pd.read_csv(path)
    frame.loc[frame["variation"].isin({"P-1", "P-2", "P-3", "P-4"}), "pr_id"] = "PR-2"
    frame.to_csv(path, index=False)
    with pytest.raises(GenerationVerificationError, match="partial control and issue PR sets differ"):
        validate_output(
            "llama",
            path,
            manifest,
            {"PR-1", "PR-2"},
            expected_issue_prs=1,
            expected_llama_control_prs=1,
        )


@pytest.mark.parametrize(
    ("generated_text", "finish_reason", "expected"),
    [
        ("Valid description", "stop", True),
        ("", "stop", False),
        ("ERROR_GENERATION_FAILED", "unknown", False),
        ("MOCKED DESCRIPTION - API KEY MISSING", "stop", False),
        ("Truncated description", "length", False),
        ("Unverified description", "unknown", False),
    ],
)
def test_resume_completion_policy(generated_text, finish_reason, expected):
    assert is_successful_generation_record(
        {"generated_text": generated_text, "finish_reason": finish_reason}
    ) is expected


def test_partition_tasks_accepts_empty_resume():
    assert partition_tasks([], 5) == []


def test_partition_tasks_covers_each_task_once():
    tasks = [(f"PR-{index}", "P-5") for index in range(11)]
    chunks = partition_tasks(tasks, 5)
    assert [task for chunk in chunks for task in chunk] == tasks
