# Surface envelopes versus the separated shortcut

Open `signed_envelope.pdf` to read the 20-page research report. The TeX file,
nine vector PDF figures and two generated table inputs are self-contained.

Compile **from this folder** with `tectonic signed_envelope.tex`, or run
`pdflatex signed_envelope.tex` twice. Standard LaTeX packages and Latin Modern
fonts are needed. The included reading copy was compiled with Tectonic 0.17.0.

The report includes 2,400 fresh fitted trials with absolute, capped quantile,
and raw signed quantile residuals. It reports uniform weak size containment
for identical nonnegative scores and explicitly discusses signed-score and
runtime exceptions. Coverage, volume, coordinates, fixed-input geometry,
target sensitivity, infinity rates and construction timing are visualized.

Compact design, summary, timing and verification records are in `data/output/pdf/envelope_shortcut_report/evidence/` under the repository root. See [figure data sources](DATA_SOURCES.md).
The full experiment code, per-trial metrics and 2,400 full observation archives
are separated: code in `envelope_method/report_revision/`, numerical evidence in `data/envelope_method/report_revision/` under the repository root.
See that directory's README for complete reproduction commands and provenance.
The large raw trial archives are not duplicated in this portable reading package.
