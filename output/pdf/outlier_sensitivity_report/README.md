# Outlier sensitivity report

Open `outlier_sensitivity_report.tex` to edit the formal report. All table data are embedded in this single LaTeX source; figure assets are in `figures/`.

Compile from this folder with `pdflatex outlier_sensitivity_report.tex` twice or `tectonic outlier_sensitivity_report.tex`. The report uses standard LaTeX packages and vector PDF figures.

The selected unchanged evidence is in `data/output/pdf/outlier_sensitivity_report/data/` under the repository root. Its relocated `source_manifest.json` records original paths and checksums. Full residual archives are under `data/envelope_method/results/`. See [figure data sources](DATA_SOURCES.md).

To regenerate the document from the project root, run `envelope_method/build_outlier_sensitivity_report.py` with the project's configured Python dependencies. This verifies the 12 summary rows against 2,400 trial-method records and performs no new model fitting.
