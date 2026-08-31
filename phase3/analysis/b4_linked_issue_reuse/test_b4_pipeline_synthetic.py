"""Synthetic integration tests for B4. Never import or read study files."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


try:
    import rank_bm25  # noqa: F401
except ModuleNotFoundError:
    fake_module = types.ModuleType("rank_bm25")

    class SyntheticBM25Okapi:
        def __init__(self, corpus, k1=1.5, b=0.75):
            self.corpus = [set(tokens) for tokens in corpus]

        def get_scores(self, query):
            query_set = set(query)
            return np.asarray([len(query_set.intersection(doc)) for doc in self.corpus])

    fake_module.BM25Okapi = SyntheticBM25Okapi
    sys.modules["rank_bm25"] = fake_module

import analysis_b4_linked_issue_reuse as B4


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class PipelineTests(unittest.TestCase):
    def test_synthetic_end_to_end_without_bertscore(self) -> None:
        original_counts = (
            B4.EXPECTED_PRS,
            B4.EXPECTED_AI,
            B4.EXPECTED_HUMAN,
            B4.EXPECTED_DESCRIPTIONS,
            B4.EXPECTED_EXPOSED_PROMPTS,
            B4.EXPECTED_MODEL_CONTEXT_COUNTS,
        )
        B4.EXPECTED_PRS = 2
        B4.EXPECTED_AI = 36
        B4.EXPECTED_HUMAN = 2
        B4.EXPECTED_DESCRIPTIONS = 38
        B4.EXPECTED_EXPOSED_PROMPTS = 12
        B4.EXPECTED_MODEL_CONTEXT_COUNTS = {
            (model, context): 2 for model in B4.MODELS for context in B4.CONTEXTS
        }
        try:
            with tempfile.TemporaryDirectory(prefix="b4-synthetic-") as directory:
                root = Path(directory)
                completion = root / "completion.json"
                completion.write_text(
                    json.dumps(
                        {
                            "status": "complete_and_verified",
                            "protocol_version": B4.GENERATION_PROTOCOL_VERSION,
                        }
                    ),
                    encoding="utf-8",
                )
                issue_by_pr = {
                    "PR-A": (
                        "Preserve parser state",
                        "Preserve parser state across nested calls.",
                    ),
                    "PR-B": (
                        "Improve serializer",
                        "Improve serializer output for empty nodes.",
                    ),
                }
                records = []
                for ordinal, (pr_id, (title, body)) in enumerate(issue_by_pr.items(), 1):
                    block = f"Title: {title}\nBody: {body}"
                    records.append(
                        {
                            "repository": "example/repo",
                            "pr_id": pr_id,
                            "prompt_issue_block": block,
                            "prompt_issue_block_sha256": hash_text(block),
                            "issues": [
                                {
                                    "url": f"https://github.com/example/repo/issues/{ordinal}",
                                    "issue_number": ordinal,
                                    "reference_order": 0,
                                    "title": title,
                                    "title_sha256": hash_text(title),
                                    "body": body,
                                    "body_sha256": hash_text(body),
                                }
                            ],
                        }
                    )
                issue_path = root / "issues.jsonl"
                issue_path.write_text(
                    "".join(json.dumps(record) + "\n" for record in records),
                    encoding="utf-8",
                )
                sources, source_diag = B4.load_issue_manifest(issue_path)
                self.assertEqual(source_diag["unique_rendered_issue_blocks"], 2)

                rows = []
                for pr_id, (title, body) in issue_by_pr.items():
                    issue_text = f"Title: {title}\nBody: {body}"
                    for model in B4.MODELS:
                        for context in B4.CONTEXTS:
                            context_text = (
                                f"<issues>\n{issue_text}\n</issues>"
                                if context in B4.EXPOSED
                                else f"<context>{context}</context>"
                            )
                            prompt = f"Synthetic prompt\n{context_text}\nEnd"
                            description = (
                                "Preserve parser state across nested calls with tests."
                                if pr_id == "PR-A"
                                else "Improve serializer output for empty nodes with tests."
                            )
                            rows.append(
                                {
                                    "pr_id": pr_id,
                                    "variation": context,
                                    "model": model,
                                    "full_prompt": prompt,
                                    "prompt_hash": hashlib.md5(prompt.encode()).hexdigest(),
                                    "generated_text": description,
                                    "finish_reason": "stop",
                                }
                            )
                raw = pd.DataFrame(rows)
                corrected_sources = {
                    **B4.CORRECTED_SOURCE_FILES,
                    "corrected_completion": {
                        "path": "completion.json",
                        "sha256": "synthetic",
                    },
                }
                with (
                    patch.object(B4, "CORRECTED_SOURCE_FILES", corrected_sources),
                    patch.object(B4, "verify_frozen_sources", return_value={"synthetic": "hash"}),
                    patch.object(B4, "select_generation_rows", return_value=raw),
                ):
                    selected, prompt_diag, _ = B4.load_corrected_generations(root, sources)
                self.assertEqual(prompt_diag["eligible_ai_rows"], 36)
                self.assertTrue(sources["prompt_rows_verified"].eq(6).all())

                corpus_rows = []
                for row in raw.itertuples(index=False):
                    corpus_rows.append(
                        {
                            "pr_id": row.pr_id,
                            "entry_type": "ai",
                            "variation": row.variation,
                            "model": row.model,
                            "text": row.generated_text,
                            "text_sha256": hash_text(row.generated_text),
                        }
                    )
                for pr_id in issue_by_pr:
                    text = "A historical synthetic description."
                    corpus_rows.append(
                        {
                            "pr_id": pr_id,
                            "entry_type": "human",
                            "variation": "original",
                            "model": "human",
                            "text": text,
                            "text_sha256": hash_text(text),
                        }
                    )
                corpus_path = root / "corpus.csv"
                pd.DataFrame(corpus_rows).to_csv(corpus_path, index=False)
                descriptions, description_diag = B4.load_descriptions(
                    corpus_path, selected, sources
                )
                self.assertEqual(description_diag["description_rows"], 38)

                unrelated = B4.assign_unrelated_sources(sources)
                lexical, components, spans, unrelated_scores = B4.score_lexical(
                    descriptions, unrelated
                )
                self.assertEqual(len(lexical), 76)
                self.assertEqual(len(components), 152)
                self.assertEqual(len(spans), 76)
                self.assertEqual(len(unrelated_scores), 38)

                bm25 = B4.score_bm25(descriptions, sources, 1.5, 0.75)
                self.assertEqual(len(bm25), 38)
                self.assertTrue(bm25["reciprocal_rank"].between(0, 1).all())

                normalized_ai = lexical[
                    lexical["representation"].eq("conservative_normalized")
                    & lexical["source_type"].eq("AI")
                ]
                primary = B4.paired_contrast_family(
                    normalized_ai,
                    "ngram_5_containment",
                    "P-5",
                    B4.UNEXPOSED,
                    "synthetic_primary",
                )
                self.assertEqual(len(primary), 12)
                self.assertIn("adj_p", primary.columns)

                output = root / "outputs"
                output.mkdir()
                B4.write_outputs(
                    output,
                    sources,
                    unrelated,
                    descriptions,
                    lexical,
                    components,
                    spans,
                    unrelated_scores,
                    bm25,
                    None,
                    {"primary": primary},
                    {"protocol_version": "synthetic"},
                )
                for expected in [
                    "prompt_exact_issue_sources.csv",
                    "metrics_long.csv",
                    "bm25_retrieval_scores.csv",
                    "stats_primary.csv",
                    "high_overlap_inspection_sample.csv",
                    "metadata.json",
                ]:
                    self.assertTrue((output / expected).is_file(), expected)
        finally:
            (
                B4.EXPECTED_PRS,
                B4.EXPECTED_AI,
                B4.EXPECTED_HUMAN,
                B4.EXPECTED_DESCRIPTIONS,
                B4.EXPECTED_EXPOSED_PROMPTS,
                B4.EXPECTED_MODEL_CONTEXT_COUNTS,
            ) = original_counts

    def test_unrelated_control_excludes_duplicate_issue_hash(self) -> None:
        duplicate = "Title: Same\nBody: Same body"
        distinct = "Title: Other\nBody: Different body"
        sources = pd.DataFrame(
            [
                {"pr_id": "A", "issue_text": duplicate, "issue_sha256": hash_text(duplicate)},
                {"pr_id": "B", "issue_text": duplicate, "issue_sha256": hash_text(duplicate)},
                {"pr_id": "C", "issue_text": distinct, "issue_sha256": hash_text(distinct)},
            ]
        )
        matches = B4.assign_unrelated_sources(sources).set_index("pr_id")
        self.assertEqual(matches.loc["A", "same_hash_candidates_excluded"], 1)
        self.assertEqual(matches.loc["B", "same_hash_candidates_excluded"], 1)
        self.assertNotEqual(
            matches.loc["A", "unrelated_issue_sha256"], sources.iloc[0]["issue_sha256"]
        )
        self.assertNotEqual(
            matches.loc["B", "unrelated_issue_sha256"], sources.iloc[1]["issue_sha256"]
        )

    def test_bertscore_wrapper_records_truncation_metadata(self) -> None:
        class SyntheticTensor:
            def __init__(self, values):
                self.values = np.asarray(values, dtype=float)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.values

        class SyntheticTokenizer:
            model_max_length = 10**30

            @staticmethod
            def encode(text, add_special_tokens=True, truncation=False):
                return list(range(len(text.split()) + (2 if add_special_tokens else 0)))

        class SyntheticBERTScorer:
            def __init__(self, **kwargs):
                self._tokenizer = SyntheticTokenizer()

            @staticmethod
            def score(candidates, references):
                tensor = SyntheticTensor([0.25] * len(candidates))
                return tensor, tensor, tensor

        class SyntheticNoGrad:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
        fake_torch.no_grad = SyntheticNoGrad
        fake_bert_score = types.ModuleType("bert_score")
        fake_bert_score.BERTScorer = SyntheticBERTScorer
        descriptions = pd.DataFrame(
            [
                {
                    "repository": "example/repo",
                    "pr_id": "A",
                    "model": B4.MODELS[0],
                    "variation": "P-5",
                    "exposure": "issue_exposed",
                    "source_type": "AI",
                    "description_id": "description-a",
                    "description_sha256": hash_text("generated text"),
                    "issue_document_id": "issue-a",
                    "issue_sha256": hash_text("issue text"),
                    "description_text": "generated text",
                    "issue_text": "issue text",
                }
            ]
        )
        with patch.dict(sys.modules, {"torch": fake_torch, "bert_score": fake_bert_score}):
            scores, metadata = B4.score_bertscore(descriptions, 2, 10)
        self.assertEqual(len(scores), 1)
        self.assertEqual(metadata["tokenizer_max_length"], 512)
        self.assertFalse(bool(scores.loc[0, "pair_truncated"]))


if __name__ == "__main__":
    unittest.main()
