import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("generation_session_diagnostic.py")
SPEC = importlib.util.spec_from_file_location("generation_session_diagnostic", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_stable_id_hash_ignores_input_order():
    assert MODULE.stable_id_hash(["b", "a"]) == MODULE.stable_id_hash(["a", "b"])


def test_rouge_l_identical_and_disjoint():
    assert MODULE.rouge_l_f1(["a", "b"], ["a", "b"]) == 1.0
    assert MODULE.rouge_l_f1(["a"], ["b"]) == 0.0


def test_jaccard_handles_empty_and_overlap():
    assert MODULE.jaccard([], []) == 1.0
    assert MODULE.jaccard(["a", "b"], ["b", "c"]) == 1 / 3


def test_bootstrap_is_deterministic_and_ordered():
    first = MODULE.bootstrap_interval(np.arange(10), "median", seed=123, draws=1000)
    second = MODULE.bootstrap_interval(np.arange(10), "median", seed=123, draws=1000)
    assert first == second
    assert first[0] <= np.median(np.arange(10)) <= first[1]


def test_successful_mask_rejects_api_error_and_non_stop():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "generated_text": ["valid", "ERROR_GENERATION_FAILED", "valid"],
            "finish_reason": ["stop", "unknown", "length"],
        }
    )
    assert MODULE.successful_mask(frame).tolist() == [True, False, False]
