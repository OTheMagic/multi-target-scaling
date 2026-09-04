"""Check plot provenance, labels, fonts, and scientific-unit conversions."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from pypdf import PdfReader
from reviewer_update import build_experiment_update as style
from reviewer_update import build_real_diagnostics as real

LATEX = ROOT / "multi_target_scaling_latex"


class FigureUnificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = json.loads((LATEX / "figure_style_audit.json").read_text())

    def test_saved_plot_values_match_csvs(self):
        for figure in self.audit["figures"]:
            sources = {Path(item["path"]).name: pd.read_csv(ROOT / item["path"])
                       for item in figure["sources"]}
            for row in figure["plotted_rows"]:
                candidates = [frame for name, frame in sources.items()
                              if name.startswith(row["method"].lower() + "_")]
                self.assertEqual(len(candidates), 1)
                frame = candidates[0]
                mask = frame.n_dim.eq(row["n_dim"]) & frame.n_cals.eq(row["n_cals"])
                if "df" in row:
                    mask &= frame.df.eq(row["df"])
                source = frame.loc[mask]
                self.assertEqual(len(source), 1)
                for key in ["test_coverage_avg", "coverage_vol_avg", "runtime_avg", "n_trials"]:
                    self.assertTrue(np.isclose(row[key], source.iloc[0][key], rtol=1e-13, atol=1e-15))
                if "runtime_ms" in row:
                    self.assertAlmostEqual(row["runtime_ms"], 1000 * row["runtime_avg"], places=8)

    def test_coverage_range_and_no_error_bars(self):
        for item in self.audit["figures"]:
            self.assertEqual(item["coverage_ylim"], [0.6, 1.0])
            self.assertFalse(item["error_bars"])

    def test_no_invented_lwc_dimensions(self):
        item = next(f for f in self.audit["figures"] if f["figure"] == "fig_app_dimension_scaling.pdf")
        dims = {row["n_dim"] for row in item["plotted_rows"] if row["method"] == "TSCP_LWC"}
        self.assertEqual(dims, {2, 3, 4})

    def test_one_canonical_method_style(self):
        for item in self.audit["figures"]:
            for method, specification in item["method_styles"].items():
                label = style._display_name(method)
                self.assertEqual(specification, {"label": label, "color": style.COLORS[label],
                                                  "marker": style.MARKERS[label]})
        for method, label in real.LABELS.items():
            self.assertEqual(label, style._display_name(method))
            self.assertEqual(real.COLORS[method], style.COLORS[label])

    def test_single_page_vector_pdfs_and_legends(self):
        for item in self.audit["figures"]:
            reader = PdfReader(LATEX / "figures" / item["figure"])
            self.assertEqual(len(reader.pages), 1)
            page = reader.pages[0]
            text = page.extract_text()
            self.assertIn("Joint coverage", text)
            self.assertIn("Residual-space volume", text)
            for specification in item["method_styles"].values():
                self.assertIn(specification["label"], text)
            fonts = page["/Resources"]["/Font"].get_object()
            self.assertTrue(fonts)
            for ref in fonts.values():
                font = ref.get_object()
                self.assertNotEqual(font["/Subtype"], "/Type3")
                self.assertIn("DejaVuSans", font["/BaseFont"])


if __name__ == "__main__":
    unittest.main()
