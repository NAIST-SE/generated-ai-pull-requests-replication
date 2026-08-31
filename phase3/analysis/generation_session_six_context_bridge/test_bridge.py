#!/usr/bin/env python3
"""Small model-free tests for bridge helper semantics."""

from __future__ import annotations

import unittest

import pandas as pd

from analyze import aggregate_value, complete_ids


class BridgeHelpersTest(unittest.TestCase):
    def test_headline_aggregations(self) -> None:
        values = pd.Series([0.1, 0.3, 0.9])
        self.assertAlmostEqual(aggregate_value(values, "median"), 0.3)
        self.assertAlmostEqual(aggregate_value(values, "mean"), 1.3 / 3)

    def test_complete_case_selection(self) -> None:
        rows = []
        for model_label, model, count in (
            ("gpt", "gpt-model", 50),
            ("qwen", "qwen-model", 39),
        ):
            for number in range(count):
                for variation in ("P-1", "P-2", "P-3", "P-4"):
                    rows.append(
                        {
                            "model_label": model_label,
                            "model": model,
                            "pr_id": f"{model_label}-{number}",
                            "variation": variation,
                        }
                    )
        selected = complete_ids(pd.DataFrame(rows))
        self.assertEqual(len(selected["gpt"]), 50)
        self.assertEqual(len(selected["qwen"]), 39)

    def test_unknown_aggregation_fails(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_value(pd.Series([1.0]), "maximum")


if __name__ == "__main__":
    unittest.main()
