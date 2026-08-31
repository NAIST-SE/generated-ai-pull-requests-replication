from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import common
from analyze import boolean_values, performance
from score_main import base_metrics, tokenize_bm25


class OutcomeSensitivityTests(unittest.TestCase):
    def test_expected_denominator_total(self) -> None:
        self.assertEqual(sum(common.EXPECTED.values()), 389)

    def test_bootstrap_is_deterministic(self) -> None:
        values = np.array([-2.0, -1.0, 0.0, 4.0])
        self.assertEqual(
            common.bootstrap_median_interval(values, 42, iterations=1000),
            common.bootstrap_median_interval(values, 42, iterations=1000),
        )

    def test_validation_sign_and_mismatch(self) -> None:
        observed = pd.DataFrame({"id": ["a", "b"], "x": [1.0, 2.0]})
        expected = pd.DataFrame({"id": ["a", "b"], "x": [1.0, 2.0]})
        result = common.validate_column(observed, expected, "x", "x", ["id"], "x")
        self.assertEqual(result["max_abs_difference"], 0.0)
        expected.loc[1, "x"] = 3.0
        with self.assertRaises(RuntimeError):
            common.validate_column(observed, expected, "x", "x", ["id"], "x")

    def test_metric_definitions(self) -> None:
        values = base_metrics("Alpha beta. Alpha gamma!")
        self.assertEqual(values["word_count"], 4.0)
        self.assertEqual(values["sentence_count"], 2.0)
        self.assertAlmostEqual(values["ttr"], 5 / 6)
        self.assertEqual(tokenize_bm25("A_b, C-2"), ["a_b", "c", "2"])

    def test_detector_boolean_and_performance(self) -> None:
        self.assertEqual(boolean_values(pd.Series(["True", "false", "1", "0"])).tolist(), [True, False, True, False])
        values = performance(tp=8, fp=2, tn=8, fn=2)
        self.assertAlmostEqual(values["accuracy"], 0.8)
        self.assertAlmostEqual(values["f1"], 0.8)


if __name__ == "__main__":
    unittest.main()
