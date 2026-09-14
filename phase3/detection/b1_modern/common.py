"""Shared validation, provenance, and append-only I/O for B1."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import (
    CORPUS_MANIFEST_RELATIVE_PATH,
    CORPUS_RELATIVE_PATH,
    CORPUS_SHA256,
    EXPECTED_AI,
    EXPECTED_HUMAN,
    EXPECTED_TOTAL,
    GENERATOR_MODELS,
    MANIFEST_COLUMNS,
    MODEL_SPECS,
    OFFICIAL_REPOSITORIES,
    RUN_ID,
    SMOKE_COUNT,
)


class B1ValidationError(RuntimeError):
    """A fail-closed B1 input, configuration, or output error."""


class B1PermanentInputError(B1ValidationError):
    """A documented row-level detector failure that does not invalidate later rows."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return sha256_text(payload)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(
    path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(columns), extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise B1ValidationError(
                    f"Invalid JSONL at {path}:{line_number}"
                ) from error
    return records


def load_verified_corpus(
    repo_root: Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    corpus_path = repo_root / CORPUS_RELATIVE_PATH
    corpus_manifest_path = repo_root / CORPUS_MANIFEST_RELATIVE_PATH
    if not corpus_path.is_file() or not corpus_manifest_path.is_file():
        raise B1ValidationError("The corrected corpus or its manifest is missing")
    actual_sha = sha256_file(corpus_path)
    if actual_sha != CORPUS_SHA256:
        raise B1ValidationError(
            f"Corrected corpus SHA-256 mismatch: expected {CORPUS_SHA256}, got {actual_sha}"
        )
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    if corpus_manifest.get("corpus_sha256") != CORPUS_SHA256:
        raise B1ValidationError(
            "Corpus manifest does not name the frozen B1 corpus hash"
        )
    if corpus_manifest.get("status") != "corpus_complete_and_verified":
        raise B1ValidationError(
            "Corpus manifest status is not corpus_complete_and_verified"
        )

    rows = read_csv(corpus_path)
    validate_corpus_rows(rows)
    return rows, corpus_manifest


def validate_corpus_rows(rows: Sequence[dict[str, str]]) -> None:
    if len(rows) != EXPECTED_TOTAL:
        raise B1ValidationError(
            f"Expected {EXPECTED_TOTAL} corpus rows, found {len(rows)}"
        )
    required = {
        "corpus_row",
        "pr_id",
        "variation",
        "model",
        "entry_type",
        "entry_key",
        "text",
        "text_sha256",
        "source_experiment_id",
        "scoring_order",
    }
    if not rows or not required.issubset(rows[0]):
        raise B1ValidationError(
            f"Corpus is missing required columns: {sorted(required)}"
        )
    identifiers: set[str] = set()
    scoring_orders: set[int] = set()
    counts = {"human": 0, "ai": 0}
    for row_number, row in enumerate(rows, start=2):
        identifier = row["entry_key"].strip()
        text = row["text"]
        if not identifier or identifier in identifiers:
            raise B1ValidationError(
                f"Blank or duplicate entry_key at CSV row {row_number}"
            )
        identifiers.add(identifier)
        if not text.strip() or sha256_text(text) != row["text_sha256"]:
            raise B1ValidationError(f"Text/hash mismatch for {identifier}")
        entry_type = row["entry_type"].strip().lower()
        if entry_type not in counts:
            raise B1ValidationError(
                f"Invalid entry_type for {identifier}: {entry_type!r}"
            )
        counts[entry_type] += 1
        try:
            scoring_orders.add(int(row["scoring_order"]))
        except ValueError as error:
            raise B1ValidationError(
                f"Invalid scoring_order for {identifier}"
            ) from error
    if counts != {"human": EXPECTED_HUMAN, "ai": EXPECTED_AI}:
        raise B1ValidationError(f"Unexpected class counts: {counts}")
    if scoring_orders != set(range(1, EXPECTED_TOTAL + 1)):
        raise B1ValidationError("scoring_order is not a permutation of 1..5630")


def manifest_row(corpus_row: dict[str, str]) -> dict[str, Any]:
    entry_type = corpus_row["entry_type"].strip().lower()
    is_human = entry_type == "human"
    model = "" if is_human else corpus_row["model"].strip()
    if not is_human and model not in GENERATOR_MODELS:
        raise B1ValidationError(f"Unexpected generator model: {model}")
    context = "" if is_human else corpus_row["variation"].strip()
    run = corpus_row["source_experiment_id"].strip() or (
        "human_baseline" if is_human else "unrecorded"
    )
    return {
        "description_id": corpus_row["entry_key"].strip(),
        "pr_id": corpus_row["pr_id"].strip(),
        "entry_type": entry_type,
        "model": model,
        "context_code": context,
        "run": run,
        "text_sha256": corpus_row["text_sha256"],
        "char_count": len(corpus_row["text"]),
        "word_count": len(corpus_row["text"].split()),
        "source_corpus_row": int(corpus_row["corpus_row"]),
        "scoring_order": int(corpus_row["scoring_order"]),
    }


MARKUP_PATTERN = re.compile(
    r"```|`[^`]+`|^\s{0,3}#{1,6}\s|^\s*[-*+]\s|\[[^]]+\]\(", re.MULTILINE
)


def select_smoke_rows(corpus_rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    """Select ten deterministic coverage cases without using detector outcomes."""
    enriched = [(row, manifest_row(row)) for row in corpus_rows]
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    used_hashes: set[str] = set()

    def pick(
        candidates: Iterable[tuple[dict[str, str], dict[str, Any]]],
        role: str,
        reverse: bool = False,
    ) -> None:
        available = [
            item for item in candidates if item[1]["description_id"] not in used_ids
        ]
        available.sort(
            key=lambda item: (item[1]["char_count"], item[1]["scoring_order"]),
            reverse=reverse,
        )
        distinct = [
            item for item in available if item[1]["text_sha256"] not in used_hashes
        ]
        pool = distinct or available
        if not pool:
            raise B1ValidationError(f"Could not select smoke role {role}")
        source, record = pool[0]
        output = dict(record)
        output["smoke_role"] = role
        output["contains_markdown_or_code"] = bool(
            MARKUP_PATTERN.search(source["text"])
        )
        selected.append(output)
        used_ids.add(record["description_id"])
        used_hashes.add(record["text_sha256"])

    human = [item for item in enriched if item[1]["entry_type"] == "human"]
    ai = [item for item in enriched if item[1]["entry_type"] == "ai"]
    pick(
        (item for item in human if MARKUP_PATTERN.search(item[0]["text"])),
        "human_markdown_or_code",
    )
    pick(human, "human_short", reverse=False)
    pick(human, "human_long", reverse=True)
    pick(
        (item for item in ai if MARKUP_PATTERN.search(item[0]["text"])),
        "ai_markdown_or_code",
    )
    for model in GENERATOR_MODELS:
        model_rows = [item for item in ai if item[1]["model"] == model]
        safe_name = model.split("/")[-1].replace("-2025-08-07", "")
        pick(model_rows, f"{safe_name}_short", reverse=False)
        pick(model_rows, f"{safe_name}_long", reverse=True)
    if len(selected) != SMOKE_COUNT:
        raise B1ValidationError(
            f"Expected {SMOKE_COUNT} smoke cases, selected {len(selected)}"
        )
    if not any(row["contains_markdown_or_code"] for row in selected):
        raise B1ValidationError("Smoke selection contains no Markdown/code case")
    return selected


def verify_official_repositories(runtime_root: Path) -> dict[str, str]:
    commits: dict[str, str] = {}
    for name, spec in OFFICIAL_REPOSITORIES.items():
        path = runtime_root / "external" / spec["directory"]
        if not (path / ".git").is_dir():
            raise B1ValidationError(f"Missing official repository checkout: {path}")
        actual = git_commit(path)
        if actual != spec["commit"]:
            raise B1ValidationError(
                f"Official {name} commit mismatch: expected {spec['commit']}, got {actual}"
            )
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=path, text=True
        ).strip()
        if status:
            raise B1ValidationError(f"Official {name} checkout has local modifications")
        commits[name] = actual
    return commits


def model_snapshot_paths(
    runtime_root: Path, *, local_files_only: bool = True
) -> dict[str, Path]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise B1ValidationError("huggingface_hub is required") from error
    paths: dict[str, Path] = {}
    for name, spec in MODEL_SPECS.items():
        try:
            snapshot = snapshot_download(
                repo_id=spec["model_id"],
                revision=spec["revision"],
                cache_dir=runtime_root / "hf-cache",
                token=os.environ.get("HF_TOKEN"),
                local_files_only=local_files_only,
                allow_patterns=[
                    "*.json",
                    "*.py",
                    "*.safetensors",
                    "*.model",
                    "*.txt",
                    "tokenizer.*",
                    "special_tokens_map.json",
                ],
            )
        except Exception as error:
            raise B1ValidationError(
                f"Model snapshot unavailable for {spec['model_id']}@{spec['revision']}: {error}"
            ) from error
        paths[name] = Path(snapshot)
    return paths


def source_bundle_hash(package_dir: Path) -> str:
    names = [
        "config.py",
        "common.py",
        "adapters.py",
        "run_detector.py",
        "prepare_manifest.py",
        "verify_reference_parity.py",
        "verify_smoke.py",
        "verify_full.py",
    ]
    values = {name: sha256_file(package_dir / name) for name in names}
    return stable_json_sha256(values)


def runtime_metadata(repo_root: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": sys.platform,
        "python": sys.version,
        "repo_git_commit": git_commit(repo_root),
        "packages": {
            name: package_version(name)
            for name in (
                "torch",
                "transformers",
                "accelerate",
                "huggingface-hub",
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "statsmodels",
            )
        },
    }
    try:
        import torch

        metadata["torch_cuda_version"] = torch.version.cuda
        metadata["cuda_available"] = torch.cuda.is_available()
        metadata["cuda_device_count"] = torch.cuda.device_count()
        metadata["cuda_devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory": torch.cuda.get_device_properties(index).total_memory,
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
            for index in range(torch.cuda.device_count())
        ]
    except ImportError:
        metadata["cuda_available"] = False
    try:
        metadata["nvidia_smi"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        metadata["nvidia_smi"] = f"unavailable: {error}"
    return metadata


def require_approved_host(*, require_cuda: bool) -> None:
    if os.environ.get("B1_APPROVED_HOST") != "1":
        raise B1ValidationError(
            "Set B1_APPROVED_HOST=1 only after confirming this is the approved lab/server host"
        )
    if sys.platform == "darwin":
        raise B1ValidationError("B1 execution is prohibited on the local Mac")
    if require_cuda:
        try:
            import torch
        except ImportError as error:
            raise B1ValidationError("PyTorch is not installed") from error
        if not torch.cuda.is_available():
            raise B1ValidationError("CUDA is required for B1 detector scoring")
        if not torch.cuda.is_bf16_supported():
            raise B1ValidationError(
                "The selected B1 configurations require BF16 support"
            )


def load_manifest(output_root: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    manifest_path = output_root / "input_manifest.csv"
    metadata_path = output_root / "preparation.json"
    if not manifest_path.is_file() or not metadata_path.is_file():
        raise B1ValidationError("Run prepare_manifest before detector scoring")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("run_id") != RUN_ID
        or metadata.get("corpus_sha256") != CORPUS_SHA256
    ):
        raise B1ValidationError("B1 preparation metadata does not match the frozen run")
    if sha256_file(manifest_path) != metadata.get("input_manifest_sha256"):
        raise B1ValidationError("B1 input manifest hash mismatch")
    rows = read_csv(manifest_path)
    if len(rows) != EXPECTED_TOTAL or tuple(rows[0]) != MANIFEST_COLUMNS:
        raise B1ValidationError("B1 input manifest schema or row count mismatch")
    return rows, metadata


def gaussian_class_probability(value: float, params: dict[str, float]) -> float:
    """Stable equivalent of the official two-normal balanced-prior calibration."""

    def log_pdf(mu: float, sigma: float) -> float:
        return (
            -math.log(sigma)
            - 0.5 * math.log(2.0 * math.pi)
            - ((value - mu) ** 2 / (2.0 * sigma**2))
        )

    log0 = log_pdf(params["mu0"], params["sigma0"])
    log1 = log_pdf(params["mu1"], params["sigma1"])
    difference = log0 - log1
    if difference >= 0:
        exponent = math.exp(-difference)
        return exponent / (1.0 + exponent)
    return 1.0 / (1.0 + math.exp(difference))
