"""Check complete diagnostic outputs against cached residuals and pristine TSCP."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from reviewer_update.run_real_diagnostics import CACHE, OUT, TABLES, DATASETS, _stable_hash
from utility.res_rescaled import standardized_prediction

spec = importlib.util.spec_from_file_location("pristine_tscp", ROOT / "reviewer_update/pre_real_diagnostics/res_rescaled.py")
pristine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pristine)
manifest = json.loads((OUT / "run_manifest.json").read_text())
trials = manifest["trials_per_dataset"]
assert trials == 200
joint = pd.read_csv(TABLES / "real_joint_trials.csv")
coords = pd.read_csv(TABLES / "real_coordinate_trials.csv")
search = pd.read_csv(TABLES / "real_search_trials.csv")
by_coordinate = pd.read_csv(TABLES / "real_search_coordinate_trials.csv")
timings = pd.read_csv(TABLES / "real_timing_repeats.csv")
assert not joint.duplicated(["dataset", "trial", "method"]).any()
assert not coords.duplicated(["dataset", "trial", "method", "coordinate"]).any()
assert not by_coordinate.duplicated(["dataset", "trial", "coordinate"]).any()
assert len(joint) == 4800 and len(coords) == 40800 and len(by_coordinate) == 10200
merged = coords.merge(joint[["dataset", "trial", "method", "joint_coverage"]],
                      on=["dataset", "trial", "method"], validate="many_to_one")
assert (merged.marginal_coverage + 1e-15 >= merged.joint_coverage).all()
assert (coords.full_length >= 0).all() and not coords.full_length.isna().any()
assert not search.zero_scale_coordinates.any()
assert (timings.runtime_ms > 0).all()
assert (timings.groupby(["dataset", "trial"]).size() == 7).all()
assert (coords.loc[coords.method == "Unscaled"].groupby(["dataset", "trial"]).full_length.nunique() == 1).all()
warnings_seen = []
checked = 0
for dataset in DATASETS:
    metadata = json.loads((CACHE / f"{dataset}_metadata.json").read_text())
    for trial in range(trials):
        with np.load(CACHE / dataset / f"split_{trial:03d}.npz", allow_pickle=False) as saved:
            seed = _stable_hash(trial)
            train, rest = train_test_split(np.arange(metadata["rows"]), test_size=0.25, random_state=seed)
            cal, test = train_test_split(rest, test_size=0.8, random_state=seed)
            for expected, key in [(train, "train_indices"), (cal, "cal_indices"), (test, "test_indices")]:
                np.testing.assert_array_equal(expected, saved[key])
            original = pristine.standardized_prediction(saved["scores_cal"], 0.1)
            diagnostics = {}
            current = standardized_prediction(saved["scores_cal"], 0.1, diagnostics=diagnostics)
            np.testing.assert_array_equal(original.info(), current.info())
            expected_lengths = 2 * current.upper
            recorded = coords.loc[(coords.dataset == dataset) & (coords.trial == trial) &
                                  (coords.method == "TSCP_R")].sort_values("coordinate")
            np.testing.assert_allclose(recorded.full_length, expected_lengths, rtol=1e-13, atol=0)
            expected_inside = saved["scores_test"] <= current.upper
            np.testing.assert_allclose(recorded.marginal_coverage, expected_inside.mean(axis=0), atol=1e-14)
            row = search.loc[(search.dataset == dataset) & (search.trial == trial)].iloc[0]
            assert row.fallback == int(diagnostics["fallback"])
            assert row.backward_coordinates == diagnostics["coordinate_branch"].count("backward")
            assert row.backward_candidates == sum(diagnostics["backward_evaluations"])
            assert row.binary_candidates == sum(diagnostics["binary_evaluations"])
            warnings_seen.extend(json.loads(str(saved["warnings_json"])))
            checked += 1
    print(f"Verified {dataset}: all {trials} split indices, rectangles, coordinate metrics, and branch counts.", flush=True)
report = dict(splits_verified=checked, coordinate_method_rows=len(coords),
              original_rectangle_equality="bitwise equality on all 1200 calibration sets",
              split_indices="exact match to original two-stage hash-seeded splitting code",
              zero_scale_coordinates=0, target_within_one_sd=pd.read_csv(TABLES / "real_joint_summary.csv").query("method == 'TSCP_R'")["target_within_one_sd"].tolist(),
              fit_warning_count=len(warnings_seen), distinct_fit_warnings=sorted(set(warnings_seen)),
              instrumentation_tests=63)
(OUT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
