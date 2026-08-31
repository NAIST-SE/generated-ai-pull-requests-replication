import json

import pandas as pd
import pytest

from phase3.data_loader import DataLoader
from phase3.generation.issue_sources import (
    IssueSourceError,
    load_issue_source_manifest,
    render_issue_prompt_block,
    render_issue_section,
    sha256_file,
    sha256_text,
    validate_manifest_record,
)


def make_issue(number=7, order=1):
    title = f"Issue {number}"
    body = f"Body for issue {number}"
    return {
        "reference_order": order,
        "issue_number": number,
        "url": f"https://github.com/owner/repo/issues/{number}",
        "title": title,
        "body": body,
        "title_sha256": sha256_text(title),
        "body_sha256": sha256_text(body),
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-02T00:00:00Z",
    }


def make_record(pr_id="PR-1"):
    issues = [make_issue(7, 1), make_issue(9, 3)]
    block = render_issue_prompt_block(issues)
    return {
        "pr_id": pr_id,
        "pull_number": "11",
        "repository": "owner/repo",
        "issues": issues,
        "prompt_issue_block": block,
        "prompt_issue_block_sha256": sha256_text(block),
    }


def test_manifest_record_preserves_noncontiguous_reference_order():
    record = validate_manifest_record(make_record())
    assert [issue["reference_order"] for issue in record["issues"]] == [1, 3]


def test_manifest_record_rejects_pull_url():
    record = make_record()
    record["issues"][0]["url"] = "https://github.com/owner/repo/pull/7"
    with pytest.raises(IssueSourceError, match="non-issue URL"):
        validate_manifest_record(record)


def test_manifest_record_rejects_wrong_repository_url():
    record = make_record()
    record["issues"][0]["url"] = "https://github.com/other/repo/issues/7"
    with pytest.raises(IssueSourceError, match="differs from repository/number"):
        validate_manifest_record(record)


def test_manifest_file_hash_and_counts_are_fail_closed(tmp_path):
    manifest = tmp_path / "sources.jsonl"
    manifest.write_text(json.dumps(make_record(), sort_keys=True) + "\n")
    observed_hash = sha256_file(manifest)
    loaded = load_issue_source_manifest(
        manifest,
        expected_sha256=observed_hash,
        expected_pr_count=1,
        expected_issue_count=2,
    )
    assert list(loaded) == ["PR-1"]
    with pytest.raises(IssueSourceError, match="hash mismatch"):
        load_issue_source_manifest(
            manifest,
            expected_sha256="0" * 64,
            expected_pr_count=1,
            expected_issue_count=2,
        )


def test_render_issue_section_rejects_target_pr_collision():
    issue = make_issue()
    block = render_issue_prompt_block([issue])
    pr_data = {
        "info": {"title": issue["title"], "body": issue["body"]},
        "issues": [issue],
        "issue_source": {
            "prompt_issue_block": block,
            "prompt_issue_block_sha256": sha256_text(block),
        },
    }
    with pytest.raises(IssueSourceError, match="equals the target PR"):
        render_issue_section(pr_data)


def test_data_loader_attaches_only_typed_manifest_issues(tmp_path):
    datasets = tmp_path / "phase3" / "datasets"
    datasets.mkdir(parents=True)
    pd.DataFrame(
        [{"id": "PR-1", "title": "Target title", "body": "Target body"}]
    ).to_csv(datasets / "selected_prs_sampled.csv", index=False)
    pd.DataFrame(
        [{"id": "PR-1", "filename": "src/Main.java", "patch": "+x"}]
    ).to_csv(
        datasets / "selected_prs_with_diffs_and_file_contents.csv", index=False
    )

    manifest = tmp_path / "sources.jsonl"
    record = make_record()
    manifest.write_text(json.dumps(record, sort_keys=True) + "\n")
    loader = DataLoader(
        base_path=str(datasets),
        issue_manifest_path=manifest,
        issue_manifest_sha256=sha256_file(manifest),
        issue_manifest_expected_prs=1,
        issue_manifest_expected_issues=2,
    )
    data = loader.get_comprehensive_data(require_issue_manifest=True)
    assert len(data["PR-1"]["issues"]) == 2
    assert data["PR-1"]["issues"][0]["title"] == "Issue 7"
    assert data["PR-1"]["info"]["title"] == "Target title"
    assert "prompt_issue_block_sha256" in data["PR-1"]["issue_source"]


def test_data_loader_refuses_missing_manifest_for_p5(tmp_path):
    datasets = tmp_path / "phase3" / "datasets"
    datasets.mkdir(parents=True)
    loader = DataLoader(
        base_path=str(datasets),
        issue_manifest_path=tmp_path / "missing.jsonl",
    )
    with pytest.raises(IssueSourceError, match="manifest is absent"):
        loader.load_audited_issue_sources(required=True)


def test_chatgpt_builder_uses_exact_frozen_issue_block():
    from phase3.generation.generate_chatgpt import PromptBuilder, VARIATIONS

    record = validate_manifest_record(make_record())
    pr_data = {
        "info": {
            "id": "PR-1",
            "title": "Target title",
            "body": "Target body",
            "repository_name_with_owner": "owner/repo",
            "issue_titles": "legacy target title",
            "issue_bodies": "legacy target body",
        },
        "files": [],
        "issues": record["issues"],
        "issue_source": {
            "prompt_issue_block": record["prompt_issue_block"],
            "prompt_issue_block_sha256": record["prompt_issue_block_sha256"],
        },
    }
    prompt = PromptBuilder.build_prompt(pr_data, VARIATIONS["P-5"])
    assert (
        f"<issues>\n{record['prompt_issue_block']}\n</issues>" in prompt
    )
    assert "legacy target title" not in prompt
