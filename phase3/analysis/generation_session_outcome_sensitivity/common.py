"""Shared frozen inputs for the generation-session outcome-sensitivity diagnostic."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import socket
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROTOCOL = "generation-session-outcome-sensitivity-v1"
MODELS = {
    "qwen": "Qwen/Qwen2.5-72B-Instruct",
    "gpt": "gpt-5-mini-2025-08-07",
}
VARIATIONS = ("P-1", "P-2", "P-3", "P-4")
CONTEXTS = {
    "P-1": "Repo Path",
    "P-2": "PR Title",
    "P-3": "File Diff",
    "P-4": "File Contents",
}
EXPECTED = {
    (MODELS["gpt"], variation): 50 for variation in VARIATIONS
} | {
    (MODELS["qwen"], variation): (39 if variation == "P-4" else 50)
    for variation in VARIATIONS
}
FAILURE_MARKERS = ("ERROR_GENERATION_FAILED", "MOCKED DESCRIPTION")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def require_nokogiri() -> None:
    hostname = socket.gethostname().split(".")[0].lower()
    if hostname != "nokogiri":
        raise SystemExit(f"Refusing to run on {hostname!r}; expected Nokogiri")


def project_root(script: Path) -> Path:
    return script.resolve().parents[3]


def output_dir(root: Path) -> Path:
    return root / "phase3/analysis/outputs/tier_b" / PROTOCOL


def source_paths(root: Path) -> dict[str, Path]:
    diagnostic = root / "phase3/analysis/outputs/tier_b/generation-session-diagnostic-v1"
    corrected = root / "phase3/datasets/temperature_1/run_3_corrected_detection_v1"
    rq2 = root / "phase3/analysis/outputs/corrected_rq23_v1/notebooks/outputs"
    return {
        "sample_manifest": diagnostic / "sample_manifest.json",
        "diagnostic_verification": diagnostic / "verification.json",
        "qwen_current": diagnostic / "qwen_current.csv",
        "gpt_current": diagnostic / "gpt_current.csv",
        "corrected_detector_corpus": corrected / "detection_results.csv",
        "corrected_detector_completion": corrected / "completion.json",
        "rq2_completion": root / "phase3/analysis/outputs/corrected_rq23_v1/completion.json",
        "rq2_outputs": rq2,
    }


def normalize_pr_id(value: Any) -> str:
    return re.sub(r"\.0$", "", str(value).strip())


def successful_current(frame: pd.DataFrame) -> pd.Series:
    generated = frame["generated_text"].fillna("").astype(str).str.strip()
    finish = frame["finish_reason"].fillna("").astype(str).str.strip().str.lower()
    failed = pd.Series(False, index=frame.index)
    upper = generated.str.upper()
    for marker in FAILURE_MARKERS:
        failed |= upper.str.contains(marker, regex=False)
    return generated.ne("") & ~failed & finish.eq("stop")


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Return paired raw data in memory, the 308-human corpus, and source hashes.

    Raw text is intentionally never written by this diagnostic.
    """
    paths = source_paths(root)
    for label, path in paths.items():
        if label == "rq2_outputs":
            if not path.is_dir():
                raise FileNotFoundError(path)
        elif not path.is_file():
            raise FileNotFoundError(path)

    sample = json.loads(paths["sample_manifest"].read_text(encoding="utf-8"))
    selected = {normalize_pr_id(value) for value in sample["selected_pr_ids"]}
    if len(selected) != 50:
        raise RuntimeError(f"Expected 50 frozen PR IDs, observed {len(selected)}")
    verification = json.loads(paths["diagnostic_verification"].read_text(encoding="utf-8"))
    if verification.get("status") not in {
        "complete_and_verified",
        "partial_generation_with_verified_failures",
    }:
        raise RuntimeError("Generation diagnostic is not structurally verified")

    corpus = pd.read_csv(
        paths["corrected_detector_corpus"], dtype=str, keep_default_na=False
    )
    corpus["pr_id"] = corpus["pr_id"].map(normalize_pr_id)
    humans = corpus[corpus["entry_type"].str.lower().eq("human")].copy()
    ai = corpus[corpus["entry_type"].str.lower().eq("ai")].copy()
    if len(humans) != 308 or humans["pr_id"].nunique() != 308:
        raise RuntimeError("Corrected human corpus is not 308 unique PR descriptions")
    if humans["text"].str.strip().eq("").any() or humans["qa_question"].str.strip().eq("").any():
        raise RuntimeError("Corrected human corpus contains empty text/title")
    if not (humans["text"].map(sha256_text) == humans["text_sha256"]).all():
        raise RuntimeError("Human text hashes do not validate")
    humans = humans.sort_values("pr_id").reset_index(drop=True)
    human_by_id = humans.set_index("pr_id", verify_integrity=True)

    pair_rows: list[dict[str, Any]] = []
    for label, model in MODELS.items():
        current_path = paths[f"{label}_current"]
        current = pd.read_csv(current_path, dtype=str, keep_default_na=False)
        current["pr_id"] = current["pr_id"].map(normalize_pr_id)
        current = current.loc[successful_current(current)].copy()
        current = current[
            current["pr_id"].isin(selected) & current["variation"].isin(VARIATIONS)
        ].copy()
        if current.duplicated(["pr_id", "variation"]).any():
            raise RuntimeError(f"Current {label} data contain duplicate successful pairs")
        old = ai[
            ai["model"].eq(model)
            & ai["pr_id"].isin(selected)
            & ai["variation"].isin(VARIATIONS)
        ].copy()
        old = old.set_index(["pr_id", "variation"], verify_integrity=True)
        observed = current.groupby("variation").size().to_dict()
        for variation in VARIATIONS:
            expected = EXPECTED[(model, variation)]
            if int(observed.get(variation, 0)) != expected:
                raise RuntimeError(
                    f"{model} {variation}: expected {expected} current rows, "
                    f"observed {observed.get(variation, 0)}"
                )
        for row in current.sort_values(["variation", "pr_id"]).itertuples(index=False):
            key = (row.pr_id, row.variation)
            if key not in old.index:
                raise RuntimeError(f"No historical corrected-corpus row for {model} {key}")
            historical = old.loc[key]
            human = human_by_id.loc[row.pr_id]
            if str(historical["qa_question"]) != str(human["qa_question"]):
                raise RuntimeError(f"Title mismatch for {row.pr_id}")
            if sha256_text(str(historical["text"])) != str(historical["text_sha256"]):
                raise RuntimeError(f"Historical text hash mismatch for {model} {key}")
            pair_rows.append(
                {
                    "model_label": label,
                    "model": model,
                    "pr_id": row.pr_id,
                    "variation": row.variation,
                    "context": CONTEXTS[row.variation],
                    "historical_text": str(historical["text"]),
                    "current_text": str(row.generated_text),
                    "human_text": str(human["text"]),
                    "title": str(human["qa_question"]),
                }
            )
    pairs = pd.DataFrame(pair_rows).sort_values(
        ["model", "variation", "pr_id"]
    ).reset_index(drop=True)
    if len(pairs) != sum(EXPECTED.values()):
        raise RuntimeError(f"Expected 389 successful pairs, observed {len(pairs)}")
    if pairs.duplicated(["model", "variation", "pr_id"]).any():
        raise RuntimeError("Paired input contains duplicate keys")
    hashes = {
        label: sha256_file(path)
        for label, path in paths.items()
        if label != "rq2_outputs"
    }
    return pairs, humans, hashes


def public_pair_manifest(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in pairs.itertuples(index=False):
        rows.append(
            {
                "model_label": row.model_label,
                "model": row.model,
                "pr_id": row.pr_id,
                "variation": row.variation,
                "context": row.context,
                "historical_text_sha256": sha256_text(row.historical_text),
                "current_text_sha256": sha256_text(row.current_text),
                "human_text_sha256": sha256_text(row.human_text),
                "title_sha256": sha256_text(row.title),
                "historical_chars": len(row.historical_text),
                "current_chars": len(row.current_text),
                "human_chars": len(row.human_text),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_median_interval(
    differences: np.ndarray, seed: int, iterations: int = 10_000
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    medians = np.empty(iterations, dtype=float)
    for start in range(0, iterations, 500):
        stop = min(start + 500, iterations)
        draws = rng.choice(values, size=(stop - start, len(values)), replace=True)
        medians[start:stop] = np.median(draws, axis=1)
    low, high = np.percentile(medians, [2.5, 97.5])
    return float(low), float(high)


def validate_column(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    observed_col: str,
    expected_col: str,
    keys: list[str],
    label: str,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Fail closed unless recomputed historical values match authoritative rows."""
    observed_values = observed[keys + [observed_col]].rename(
        columns={observed_col: "_observed_value"}
    )
    expected_values = expected[keys + [expected_col]].rename(
        columns={expected_col: "_expected_value"}
    )
    merged = observed_values.merge(
        expected_values,
        on=keys,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        raise RuntimeError(f"{label}: authoritative key set differs")
    a = merged["_observed_value"].to_numpy(float)
    b = merged["_expected_value"].to_numpy(float)
    if not np.allclose(a, b, rtol=tolerance, atol=tolerance, equal_nan=True):
        bad = np.where(
            ~np.isclose(a, b, rtol=tolerance, atol=tolerance, equal_nan=True)
        )[0]
        finite_bad = [index for index in bad if np.isfinite(a[index]) and np.isfinite(b[index])]
        maximum_bad = (
            max(abs(float(a[index]) - float(b[index])) for index in finite_bad)
            if finite_bad
            else float("nan")
        )
        examples = []
        for index in bad[:5]:
            examples.append(
                {
                    **{key: str(merged.iloc[index][key]) for key in keys},
                    "observed": float(a[index]),
                    "expected": float(b[index]),
                    "absolute_difference": abs(float(a[index]) - float(b[index])),
                }
            )
        raise RuntimeError(
            f"{label}: {len(bad)} historical values differ; "
            f"max_abs_difference={maximum_bad}; examples={examples}"
        )
    finite = np.isfinite(a) & np.isfinite(b)
    maximum = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
    return {"label": label, "rows": len(merged), "max_abs_difference": maximum}
