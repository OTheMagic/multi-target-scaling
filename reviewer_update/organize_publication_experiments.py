"""Mechanically assemble edited experiment sections in scientific topic order."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / "multi_target_scaling_latex"
appendix = (LATEX / "experiments_appendix.tex").read_text(encoding="utf-8")
if r"\input{experiments_legacy}" not in appendix:
    raise SystemExit("The appendix is already consolidated.")
legacy = (LATEX / "experiments_legacy.tex").read_text(encoding="utf-8")
appendix = appendix[:appendix.rindex("\\FloatBarrier\n\\endgroup")]
boundaries = list(re.finditer(r"(?m)^\\subsection\{([^}]+)\}", appendix))
header = appendix[:boundaries[0].start()]
sections = {
    match.group(1): appendix[match.start():boundaries[i + 1].start() if i + 1 < len(boundaries) else len(appendix)].strip()
    for i, match in enumerate(boundaries)
}

def between(text, start, end):
    return text[text.index(start):text.index(end)].strip()

gaussian = between(
    legacy, r"Figure~\ref{fig:gaussian-heter} and Table",
    r"\subsubsection{Dataset-Level Coverage and Volume}",
)
tables = between(
    legacy, r"\subsubsection{Distribution-Specific Tables}",
    r"\subsection{Oracle Approximation and Computational Scaling}",
)
tables = tables.replace(r"\label{app:simulations}", "").strip()
tables = re.sub(r"\\clearpage\s*$", "", tables).strip()
distribution = (
    "\\subsection{Independent-Noise Distribution Benchmarks}\n"
    "\\label{app:simulations}\n\\label{app:legacy-supplement}\n\n"
    "The following comparisons use the fixed-pool protocol with independent\n"
    "noise coordinates and calibration sizes $n\\in\\{30,50,100,300,500\\}$.\n\n"
    + gaussian + "\n\n" + tables
)

oracle = between(
    legacy, r"\subsection{Oracle Approximation and Computational Scaling}",
    r"\subsubsection{Fixed-Pool Evaluation}",
)
oracle = re.sub(r"\\clearpage\s*$", "", oracle).strip()
oracle = oracle.replace(
    r"\label{app:ours_simulation}",
    "\\label{app:ours_simulation}\n\nThese comparisons use the fixed-pool protocol\n"
    "in Appendix~\\ref{app:experimental-protocols}.",
)
oracle = oracle.replace("under the original resampling protocol", "under the fixed-pool protocol")
heavy_fixed = legacy[legacy.index(r"\subsubsection{Fixed-Pool Evaluation}"):legacy.rindex("\\FloatBarrier\n\\endgroup")].strip()
heavy = sections["Heavy-Tailed Noise and the Role of Moment Conditions"] + "\n\n\\FloatBarrier\n" + heavy_fixed

real_table = between(
    legacy, r"\subsubsection{Dataset-Level Coverage and Volume}",
    r"\subsection{Distribution-Specific Results}",
)
real_coordinate = sections["Real-Data Coordinate Coverage and Uncertainty"]
real_coordinate = real_coordinate.replace(
    "\\subsection{Real-Data Coordinate Coverage and Uncertainty}\n\\label{app:real-coordinate-audit}", "", 1
).strip()
real = (
    "\\subsection{Real-Data Coverage and Efficiency}\n\\label{app:real-coordinate-audit}\n\n"
    + real_table + "\n\n\\subsubsection{Coordinate-Wise Coverage}\n" + real_coordinate
)

ordered = [
    sections["Experimental Protocols"], distribution,
    sections["Dependence Strength"], sections["Severity of Coordinate Heterogeneity"],
    sections["Target Level and Small Calibration Samples"], heavy,
    sections["Exchangeable Contamination Stress Test"],
    sections["Additional CQR Sensitivity Analyses"],
    sections["Low-Dimensional Shape-Template Baseline"], oracle, real,
    sections["Search-Regime Sensitivity to the Target Level"],
    sections["Monte Carlo Uncertainty"],
]
combined = header + "\n\n\\FloatBarrier\n".join(ordered) + "\n\n\\FloatBarrier\n\\endgroup\n"
labels = re.findall(r"\\label\{([^}]+)\}", re.sub(r"(?m)^%.*$", "", combined))
assert len(labels) == len(set(labels)), "Duplicate label after section assembly."
assert r"\input{experiments_legacy}" not in combined
(LATEX / "experiments_appendix.tex").write_text(combined, encoding="utf-8")
print("Consolidated all supplementary studies into experiments_appendix.tex.")
