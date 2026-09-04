"""Find deterministic branch-test fixtures; not a reported experiment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from utility.res_rescaled import standardized_prediction

for n in [5, 10, 30, 100]:
    for d in [1, 2, 3, 10]:
        for seed in range(40):
            for scale in [0.1, 1, 3]:
                scores = np.random.default_rng(seed).lognormal(0, scale, (n, d))
                for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    diagnostics = {}
                    with np.errstate(divide="ignore", invalid="ignore"):
                        standardized_prediction(scores, alpha, diagnostics=diagnostics)
                    if "backward" in diagnostics["coordinate_branch"]:
                        print("FOUND", n, d, seed, scale, alpha, diagnostics, flush=True)
                        raise SystemExit(0)
        print("Searched", n, d, flush=True)
raise RuntimeError("No actual backward case found in this fixture search.")
