import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union
import os

from phase3.generation.issue_sources import (
    DEFAULT_MANIFEST_RELATIVE_PATH,
    EXPECTED_ISSUE_COUNT,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_PR_COUNT,
    IssueSourceError,
    attach_issue_sources,
    load_issue_source_manifest,
)

# --- Mappings defined in Instructions ---
CONTEXT_MAPPING = {
    "P-1": "Repo Path",  # Repo name & file pathnames
    "P-2": "PR Title",  # PR Title only
    "P-3": "PR Diffs",  # Diffs (Default Baseline for RQ1)
    "P-4": "File Contents",  # Full file contents
    "P-5": "Related Issues",  # Linked issue descriptions
    "P-6": "Without File Contents",  # Ablation: All context EXCEPT file contents
}

MODEL_MAPPING = {
    "gpt-5-mini-2025-08-07": "Industrial (GPT-5-mini-2025-08-07)",
    "Llama-3.1-8B-Instruct": "Open Source (Llama-3.1-8B-Instruct)",
    "gpt-4o-mini": "Industrial (GPT-4o-mini Fallback)",
}

# --- Blacklist ---
# PRs known to be problematic (e.g. no files, empty content)
BLACKLIST_PRS = ["PR_kwDOAQ0TF86IEX8G"]  # No files fetched


class DataLoader:
    """
    Single source of truth for loading and merging datasets for Phase 3.
    """

    def __init__(
        self,
        base_path: str = "phase3/datasets",
        issue_manifest_path: Optional[Union[str, Path]] = None,
        issue_manifest_sha256: str = EXPECTED_MANIFEST_SHA256,
        issue_manifest_expected_prs: int = EXPECTED_PR_COUNT,
        issue_manifest_expected_issues: int = EXPECTED_ISSUE_COUNT,
    ):
        self.base_path = Path(base_path)
        if not self.base_path.is_absolute():
            # Attempt to resolve relative to project root if running from elsewhere
            # This assumes the script is run from project root or similar
            current_abs = Path(os.getcwd())
            if (current_abs / base_path).exists():
                self.base_path = current_abs / base_path
            elif (current_abs / ".." / base_path).exists():
                self.base_path = current_abs / ".." / base_path
        self.base_path = self.base_path.resolve()
        self.issue_manifest_path = (
            Path(issue_manifest_path).resolve()
            if issue_manifest_path is not None
            else (self.base_path.parent / DEFAULT_MANIFEST_RELATIVE_PATH).resolve()
        )
        self.issue_manifest_sha256 = issue_manifest_sha256
        self.issue_manifest_expected_prs = issue_manifest_expected_prs
        self.issue_manifest_expected_issues = issue_manifest_expected_issues

    def load_selected_prs(self) -> pd.DataFrame:
        """Loads the base list of selected PRs (Human Baseline), filtering out blacklisted PRs."""
        path = self.base_path / "selected_prs_sampled.csv"
        if not path.exists():
            raise FileNotFoundError(f"Selected PRs file not found at {path}")
        df = pd.read_csv(path)

        # Filter blacklist
        if "id" in df.columns:
            initial_count = len(df)
            df = df[~df["id"].isin(BLACKLIST_PRS)]
            filtered_count = len(df)
            if initial_count != filtered_count:
                print(
                    f"Filtered {initial_count - filtered_count} blacklisted PRs from selected_prs_sampled.csv"
                )

        return df

    def load_pr_files(self) -> pd.DataFrame:
        """Loads PR files with diffs and contents, filtering out blacklisted PRs."""
        path = self.base_path / "selected_prs_with_diffs_and_file_contents.csv"
        if not path.exists():
            raise FileNotFoundError(f"PR Files file not found at {path}")
        df = pd.read_csv(path)

        # Filter blacklist
        if "id" in df.columns:
            df = df[~df["id"].isin(BLACKLIST_PRS)]
        if "pr_id" in df.columns:
            df = df[~df["pr_id"].isin(BLACKLIST_PRS)]

        return df

    def load_pr_issues(self) -> pd.DataFrame:
        """Load the legacy aggregate fetch for audit/inspection only.

        Generation must use ``load_audited_issue_sources`` instead: this table contains
        target-PR ``title``/``body`` columns and mixed issue/pull artifacts.
        """
        path = self.base_path / "selected_prs_with_related_issues.csv"
        if not path.exists():
            raise FileNotFoundError(f"PR Issues file not found at {path}")
        df = pd.read_csv(path)

        # Filter blacklist
        if "id" in df.columns:
            df = df[~df["id"].isin(BLACKLIST_PRS)]

        return df

    def load_audited_issue_sources(
        self, *, required: bool = False
    ) -> Dict[str, Dict]:
        """Load the hash-pinned, genuine-issue-only B4 regeneration manifest."""
        if not self.issue_manifest_path.is_file():
            if required:
                raise IssueSourceError(
                    "P5/P6 requested but the audited issue-source manifest is absent: "
                    f"{self.issue_manifest_path}"
                )
            return {}
        return load_issue_source_manifest(
            self.issue_manifest_path,
            expected_sha256=self.issue_manifest_sha256,
            expected_pr_count=self.issue_manifest_expected_prs,
            expected_issue_count=self.issue_manifest_expected_issues,
        )

    def load_generated_descriptions(self) -> pd.DataFrame:
        """
        Loads the aggregated generated descriptions CSV from the datasets folder.
        """
        path = self.base_path / "generated_descriptions.csv"
        if not path.exists():
            print(f"Warning: Generated descriptions file not found at {path}")
            return pd.DataFrame(
                columns=["pr_id", "variation", "model", "generated_description"]
            )

        return pd.read_csv(path)

    def load_generation_token_runs(self) -> pd.DataFrame:
        """
        Loads model-specific generation outputs that include token usage metadata.

        Expected columns include:
        - variation
        - model
        - prompt_token_count
        - context_token_count
        - generated_token_count
        - truncated_token_count
        """
        patterns = [
            "chatgpt_results_exp_chatgpt_*.csv",
            "meta-llama-3.1-8b-instruct_*.csv",
            "meta-llama_llama-3.1-70b-instruct_*.csv",
            "Qwen_Qwen2.5-72B-Instruct_*.csv",
        ]

        files = []
        for pattern in patterns:
            files.extend(sorted(self.base_path.glob(pattern)))

        if not files:
            print(f"Warning: No generation token run files found in {self.base_path}")
            return pd.DataFrame()

        frames = []
        for path in files:
            df = pd.read_csv(path)
            df["source_file"] = path.name
            frames.append(df)

        return pd.concat(frames, ignore_index=True)

    def get_comprehensive_data(
        self, *, require_issue_manifest: bool = False
    ) -> Dict[str, Dict]:
        """
        Returns a dictionary of PRs with all associated data (Metadata, Files, Issues).
        Structure: { pr_id: { 'info': {}, 'files': [], 'issues': [] } }
        """
        df_prs = self.load_selected_prs()
        df_files = self.load_pr_files()

        data = {}

        # Initialize with base PR info
        for _, row in df_prs.iterrows():
            pr_id = str(row["id"])  # Ensure ID is string
            data[pr_id] = {
                "info": row.to_dict(),
                "files": [],
                "issues": [],
                "issue_source": None,
            }

        # Add Files
        for _, row in df_files.iterrows():
            pr_id = str(row["id"])
            if pr_id in data:
                data[pr_id]["files"].append(row.to_dict())

        # Attach only the independently audited, genuine-issue-only source records.
        issue_manifest = self.load_audited_issue_sources(
            required=require_issue_manifest
        )
        if issue_manifest:
            attach_issue_sources(
                data,
                issue_manifest,
                manifest_path=self.issue_manifest_path,
                manifest_sha256=self.issue_manifest_sha256,
            )

        return data
