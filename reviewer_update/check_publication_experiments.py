"""Verify that the publication edit preserves every experimental asset and result table."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / "multi_target_scaling_latex"
BEFORE = ROOT / "reviewer_update/pre_publication_experiments"

def read(path):
    return path.read_text(encoding="utf-8-sig")

def active(text):
    return re.sub(r"(?<!\\)%[^\n]*", "", text)

old = active("\n".join(read(BEFORE / name) for name in
    ["experiments_body.tex", "experiments_appendix.tex", "experiments_legacy.tex"]))
current = active("\n".join(read(LATEX / name) for name in
    ["experiments_body.tex", "experiments_appendix.tex"]))
labels = set(re.findall(r"\\label\{([^}]+)\}", old))
assert labels <= set(re.findall(r"\\label\{([^}]+)\}", current))
figures = r"\\includegraphics(?:\[[^]]*\])?\{[^}]+\}"
assert Counter(re.findall(figures, old)) == Counter(re.findall(figures, current))
data_inputs = r"\\input\{((?:figures|experiment_data)/[^}]+)\}"
assert Counter(re.findall(data_inputs, old)) == Counter(re.findall(data_inputs, current))
tables = r"\\begin\{tabular\}.*?\\end\{tabular\}"
assert re.findall(tables, old, re.DOTALL) == re.findall(tables, current, re.DOTALL)
assets = json.loads(read(BEFORE / "asset_hashes.json"))
for row in assets:
    assert hashlib.sha256((LATEX / row["path"]).read_bytes()).hexdigest().upper() == row["sha256"], row["path"]
assert r"\input{experiments_legacy}" not in current
visible = re.sub(r"\\(?:label|ref|eqref)\{[^}]+\}", "", current)
for phrase in ["legacy", "retained original", "by the reviewers", "historical", "new reruns",
               "follow-up", "pre-instrumentation", "proof and endpoint-inequality"]:
    assert phrase not in visible.lower(), f"Editorial framing remains: {phrase}"
report = {
    "experimental_assets_unchanged": len(assets),
    "experimental_graphic_inclusions_preserved": len(re.findall(figures, old)),
    "experimental_data_inputs_preserved": len(re.findall(data_inputs, old)),
    "pre_edit_experimental_labels_preserved": len(labels),
    "inline_result_tables_unchanged": True,
    "all_experiments_in_two_tex_files": True,
    "revision_log_framing_removed": True,
    "scope": "Narrative and organization only; no numerical results, figures, tables, or data files changed.",
}
(LATEX / "publication_edit_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
