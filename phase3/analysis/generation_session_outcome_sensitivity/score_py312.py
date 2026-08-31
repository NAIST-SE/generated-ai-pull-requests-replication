#!/usr/bin/env python3
"""Compute TAALED MTLD and spaCy lexical density in the study's py312 env."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import spacy
import taaled
from nltk.tokenize import word_tokenize

from common import (
    CONTEXTS,
    PROTOCOL,
    load_inputs,
    output_dir,
    project_root,
    require_nokogiri,
    sha256_file,
    source_paths,
    utc_now,
    validate_column,
    write_json,
)


TOLERANCE = 1e-10


CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}


def main() -> None:
    require_nokogiri()
    root = project_root(Path(__file__))
    out = output_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    pairs, _, source_hashes = load_inputs(root)
    nlp = spacy.load("en_core_web_sm")

    unique_texts: dict[str, str] = {}
    for row in pairs.itertuples(index=False):
        unique_texts[row.historical_text] = row.historical_text
        unique_texts[row.current_text] = row.current_text
        unique_texts[row.human_text] = row.human_text
    texts = list(unique_texts)
    lexical_density: dict[str, float] = {}
    for text, doc in zip(texts, nlp.pipe(texts, batch_size=64)):
        tokens = [token for token in doc if not token.is_punct and not token.is_space]
        lexical_density[text] = (
            float(sum(token.pos_ in CONTENT_POS for token in tokens) / len(tokens))
            if tokens
            else float("nan")
        )

    mtld: dict[str, tuple[float, float]] = {}
    for text in texts:
        tokens = word_tokenize(text.lower())
        if len(tokens) < 10:
            mtld[text] = (float("nan"), float("nan"))
        else:
            result = taaled.lexdiv(tokens)
            mtld[text] = (float(result.mtldo), float(result.mtld))

    rows: list[dict[str, Any]] = []
    for pair in pairs.itertuples(index=False):
        human_original, human_bidir = mtld[pair.human_text]
        for version in ("historical", "current"):
            text = getattr(pair, f"{version}_text")
            original, bidir = mtld[text]
            rows.append(
                {
                    "model_label": pair.model_label,
                    "model": pair.model,
                    "pr_id": pair.pr_id,
                    "variation": pair.variation,
                    "context": pair.context,
                    "version": version,
                    "mtld_original": original,
                    "mtld_bidir": bidir,
                    "lexical_density": lexical_density[text],
                    "human_mtld_original": human_original,
                    "human_mtld_bidir": human_bidir,
                    "human_lexical_density": lexical_density[pair.human_text],
                }
            )
    metrics = pd.DataFrame(rows).sort_values(
        ["model", "variation", "pr_id", "version"]
    )
    if len(metrics) != 778:
        raise RuntimeError(f"Expected 778 py312 rows, observed {len(metrics)}")

    historical = metrics[metrics["version"].eq("historical")].rename(
        columns={"pr_id": "normalized_pr_id"}
    )
    keys = ["normalized_pr_id", "model", "variation"]
    rq2 = source_paths(root)["rq2_outputs"]
    validations: list[dict[str, Any]] = []
    for relative, columns in [
        (
            "row_level/mtld_metrics.csv",
            [("mtld_original", "mtld_original"), ("mtld_bidir", "mtld_bidir")],
        ),
        (
            "row_level/lexical_density_metrics.csv",
            [("lexical_density", "lexical_density")],
        ),
    ]:
        expected = pd.read_csv(rq2 / relative, dtype={"normalized_pr_id": str})
        expected["normalized_pr_id"] = expected["normalized_pr_id"].astype(str).str.replace(
            r"\.0$", "", regex=True
        )
        expected = expected[
            expected["entry_type"].str.lower().eq("ai")
            & expected["model"].isin(historical["model"])
            & expected["variation"].isin(CONTEXTS)
        ].merge(
            historical[keys].drop_duplicates(),
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        for observed_col, expected_col in columns:
            validations.append(
                validate_column(
                    historical,
                    expected,
                    observed_col,
                    expected_col,
                    keys,
                    f"historical_{observed_col}",
                    TOLERANCE,
                )
            )

    metrics_path = out / "py312_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": utc_now(),
        "stage": "py312_metrics",
        "rows": len(metrics),
        "unique_texts_processed": len(texts),
        "spacy_version": spacy.__version__,
        "spacy_model": nlp.meta.get("name"),
        "spacy_model_version": nlp.meta.get("version"),
        "content_pos": sorted(CONTENT_POS),
        "mtld_implementation": "TAALED lexdiv: mtldo and mtld; <10 tokens undefined",
        "source_sha256": source_hashes,
        "output_sha256": sha256_file(metrics_path),
        "historical_validation_tolerance": TOLERANCE,
        "historical_validations": validations,
        "raw_text_in_outputs": False,
    }
    write_json(out / "py312_metrics_metadata.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
