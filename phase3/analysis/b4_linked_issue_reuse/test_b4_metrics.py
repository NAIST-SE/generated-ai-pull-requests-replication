"""Synthetic-only tests. This file must never load study data."""

from __future__ import annotations

import hashlib
import unittest

from b4_metrics import (
    extract_issue_block,
    lexical_metrics,
    longest_shared_span,
    ngram_containment,
    parse_issue_components,
    represent,
    tokenize,
)


class MetricsTests(unittest.TestCase):
    def test_exact_prompt_extraction(self) -> None:
        block = "Title: Fix parser\nBody: Preserve the exact token stream."
        prompt = f"prefix\n<issues>\n{block}\n</issues>\nsuffix"
        self.assertEqual(extract_issue_block(prompt), block)
        with self.assertRaises(ValueError):
            extract_issue_block("no issue block")
        with self.assertRaises(ValueError):
            extract_issue_block(prompt + f"\n<issues>\n{block}\n</issues>")

    def test_components(self) -> None:
        block = "Title: First\nBody: Alpha beta\n---\nTitle: Second\nBody: Gamma delta"
        parsed = parse_issue_components(block)
        self.assertEqual(parsed.document_count, 2)
        self.assertEqual(parsed.title, "First\n---\nSecond")
        self.assertEqual(parsed.body, "Alpha beta\n---\nGamma delta")
        self.assertEqual(parsed.combined, block)

    def test_normalization_retains_visible_link_and_identifiers(self) -> None:
        text = "See [Parser_ID](https://example.test/a)\n  NOW"
        self.assertEqual(represent(text, "conservative_normalized"), "see parser_id now")
        self.assertEqual(tokenize(text), ["see", "parser_id", "now"])

    def test_ngram_containment_is_occurrence_level(self) -> None:
        source = "one two three four five six seven".split()
        generated = "zero two three four five six eight".split()
        self.assertAlmostEqual(ngram_containment(source, generated, 5), 1 / 3)
        self.assertEqual(ngram_containment(source, ["one", "two"], 5), 0.0)

    def test_longest_span_and_metrics(self) -> None:
        source = "alpha beta gamma delta epsilon zeta".split()
        generated = "other beta gamma delta epsilon end".split()
        span = longest_shared_span(source, generated)
        self.assertEqual(span.length, 4)
        self.assertEqual(span.tokens, ("beta", "gamma", "delta", "epsilon"))
        metrics = lexical_metrics(" ".join(source), " ".join(generated), "conservative_normalized")
        self.assertEqual(metrics["longest_span_tokens"], 4)
        self.assertAlmostEqual(metrics["longest_span_generated_fraction"], 4 / 6)

    def test_hash_fixture_is_stable(self) -> None:
        self.assertEqual(
            hashlib.sha256("Title: X\nBody: Y".encode()).hexdigest(),
            "f41e40699ca3e62e8572014b3533b5a8bc89a240263d75a7d1f11aa496baa85f",
        )


if __name__ == "__main__":
    unittest.main()
