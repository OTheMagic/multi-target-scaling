# Envelope versus the old shortcut: research meeting report

The report sources, PDF, and `figures/` remain here. Numeric evidence and audit JSON now live under the repository's `data/envelope_method/meeting_report/`, mirroring their former paths. `code/reproduce.py` uses that central directory; reproducing the report requires the repository's local data tree. Historical manifest contents and scientific source snapshots retain their original provenance.

Open **[report.pdf](report.pdf)**. This is the revised **10-page report**, with plain serif typography, mathematical derivations and proofs in the main text, and five figures in the retained palette.

The reading order is: original link and its signed extension; GWC and the original full LWC; the old shortcut; the envelope construction; a separate validity and uniform-containment proof; absolute-residual experiments; signed-envelope comparisons against strong capped and shifted old methods.

## Report files and centrally stored evidence

| Path | Contents |
|---|---|
| `report.pdf`, `report.tex` | Compiled report and master TeX source |
| `tex/` | Theory and experiment text, including reported numerical statements |
| `figures/` | All five vector PDF figures and PNG previews |
| `data/absolute/` | 18 compact calibration/test-score archives, all paired trial results, summary and source hashes |
| `data/signed/` | 360 compact trial archives, transformations and thresholds, metrics, comparisons, design and source hashes |
| `code/` | Portable reproduction scripts and a frozen snapshot of the method implementations |
| `sources/` | Original manuscript sections, original experiment driver, design and historical verification/provenance |
| `qa/` | Independent computational audits, PDF checks and rendered pages |
| `theory_audit.md` | Additional mathematical review notes |
| `manifest.json` | SHA-256 inventory of the final report package |

The source files and figures here are sufficient to compile the report. Replaying comparisons and regenerating figures also requires the centrally stored evidence under `data/envelope_method/meeting_report/`; numeric paths in the table above are relative to that directory. Historical provenance paths are resolved without rewriting the saved records. The copied numerical method implementations are unchanged.

## Evidence and fair comparison

The absolute study uses 2,160 previously generated independent fitted trials: 800 new training observations, 30/80/200 calibration observations, and 1,200 test observations per trial. Both methods were rerun from those scores for this revision. These are retained evidence from the previous report, not new independent replicates. The signed study reuses 360 fitted trials; its constant-shifted and width-shifted old comparators are newly evaluated. The Laplace signed study shares 120 fits with the absolute study, giving **2,400 distinct fitted trials across the two sections**, plus one separately seeded geometry example.

The compact data retain the complete calibration and test scores needed for every measurement, fitted parameters or offsets, seeds, and fitted thresholds. Original full observation archives are preserved in their existing location; this package avoids duplicating their larger training and prediction arrays. `code/data_model.py` regenerates complete training, calibration and test observations from the saved seeds. Independent QA regenerated 27 predetermined fitted trials and reconstructed their outcome-space intervals directly.

Absolute-score gains are largest for small calibration samples (3.11–16.44% mean paired volume reductions at n=30), and fall below 0.35% at n=200. Raw signed envelope gains against the strongest average old baseline are about 1.5–1.8% in two settings; ordinary two-output Gaussian data show no gain over capping. The report includes this unfavorable comparison. All 29 infinite capped regions in the wide-base setting arise from the stated conservative zero-calibration-scale fallback.

“Best old version” is a descriptive comparison of separately reported fixed methods, not an adaptive rule chosen on test outcomes. All 120 trials contribute to coverage. Volume ratios exclude nonfinite denominators explicitly; only the wide-base capped comparison has exclusions (91 of 120 finite pairs). Error bars are pointwise 95% Monte Carlo intervals over fitted trials. No envelope-on-capped comparison is performed. Full LWC is explained mathematically; exponential full-cell enumeration is not launched by these scripts.

## Reproduce

Use Python 3.12 with the packages in `requirements.txt`. From this folder:

```text
python code/reproduce.py
```

This recomputes all paired methods, results, figures and independent audits using only local archives. To regenerate and compare all absolute data from their original seeds as well:

```text
python code/reproduce.py --regenerate
```

The independent signed audit also reconstructs raw observations and fitted intervals for trials 0, 59 and 119 of each signed configuration. It checks outcome endpoints without calling either reporting metric helper.

Compile with a standard TeX installation:

```text
tectonic report.tex
```

Alternatively, run `pdflatex report.tex` twice. The bibliography is included in the TeX source; BibTeX is unnecessary. The delivered PDF was built with Tectonic 0.17.0 and embedded Latin Modern fonts. The page layout is paper-like, with black headings and equations; plot colors remain confined to the figures.

For the installed workspace runtime on this computer, PowerShell commands are:

```powershell
$env:PYTHONPATH='E:\multi-target-scaling\tmp\diagnostic_packages'
$env:PYTHONDONTWRITEBYTECODE='1'
$reportPython='C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $reportPython code/reproduce.py
$env:TECTONIC_CACHE_DIR='E:\multi-target-scaling\tmp\tectonic-cache'
& 'E:\multi-target-scaling\tmp\tectonic-0.17.0\tectonic.exe' -C --keep-logs report.tex
```

The absolute paths above locate locally installed dependencies only. The report/data scripts use their own location and remain portable.

## Verify PDF layout

```text
pdftoppm -r 105 -png report.pdf qa/pages/page
python qa/layout_audit.py
```

The final review checks all pages in contact sheets and enlarges the equations, figure labels and captions. The TeX log has no overfull/underfull boxes, undefined references, undefined citations, or missing characters. Independent mathematical/numerical checks and result checks are recorded in `qa/core_audit.json` and `qa/independent_audit.json`.
