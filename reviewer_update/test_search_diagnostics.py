"""Regression tests for passive TSCP branch instrumentation."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from utility.res_rescaled import standardized_prediction

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "before_diagnostics", ROOT / "reviewer_update/pre_real_diagnostics/res_rescaled.py"
)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_diagnostics_preserve_old_rectangle(seed, alpha):
    scores = np.random.default_rng(seed).lognormal(0, 3, (30, 3))
    expected = reference.standardized_prediction(scores, alpha)
    diagnostics = {"stale": True}
    actual = standardized_prediction(scores, alpha, diagnostics=diagnostics)
    plain = standardized_prediction(scores, alpha)
    np.testing.assert_array_equal(actual.info(), expected.info())
    np.testing.assert_array_equal(plain.info(), expected.info())
    assert "stale" not in diagnostics
    branches = diagnostics["coordinate_branch"]
    if diagnostics["fallback"]:
        assert branches == ["not_searched"] * 3
        assert sum(diagnostics["binary_evaluations"] + diagnostics["backward_evaluations"]) == 0
    else:
        assert set(branches) <= {"binary", "backward"}
        for j, branch in enumerate(branches):
            assert diagnostics[f"{branch}_evaluations"][j] > 0
            other = "binary" if branch == "backward" else "backward"
            assert diagnostics[f"{other}_evaluations"][j] == 0


def test_all_three_regimes_are_exercised():
    found = set()
    for seed in range(12):
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
            scores = np.random.default_rng(seed).lognormal(0, 3, (30, 3))
            diagnostics = {}
            standardized_prediction(scores, alpha, diagnostics=diagnostics)
            if diagnostics["fallback"]:
                found.add("fallback")
            else:
                found.update(diagnostics["coordinate_branch"])
    scores = np.random.default_rng(1).lognormal(0, 3, (5, 1))
    diagnostics = {}
    actual = standardized_prediction(scores, 0.9, diagnostics=diagnostics)
    expected = reference.standardized_prediction(scores, 0.9)
    np.testing.assert_array_equal(actual.info(), expected.info())
    assert diagnostics["backward_evaluations"] == [1]
    found.update(diagnostics["coordinate_branch"])
    assert found == {"binary", "backward", "fallback"}, found


@pytest.mark.parametrize("mode,shortcut", [("GWC", True), ("LWC", False)])
def test_alternative_modes(mode, shortcut):
    scores = np.random.default_rng(4).exponential(size=(8, 2))
    diagnostics = {}
    actual = standardized_prediction(scores, 0.3, mode, shortcut, diagnostics=diagnostics)
    expected = reference.standardized_prediction(scores, 0.3, mode, shortcut)
    if not shortcut:
        assert len(actual[0]) == len(expected[0])
        for left, right in zip(actual[0], expected[0]):
            np.testing.assert_array_equal(left.info(), right.info())
        actual, expected = actual[1], expected[1]
    np.testing.assert_array_equal(actual.info(), expected.info())
    assert diagnostics["coordinate_branch"] == ["not_searched"] * 2


if __name__ == "__main__":
    for seed in range(12):
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
            diagnostics = {}
            standardized_prediction(np.random.default_rng(seed).lognormal(0, 3, (30, 3)), alpha, diagnostics=diagnostics)
            print(seed, alpha, diagnostics["fallback"], diagnostics["coordinate_branch"], diagnostics["backward_evaluations"])
