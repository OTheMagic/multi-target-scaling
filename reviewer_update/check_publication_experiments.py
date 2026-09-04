"""Verify numerical preservation during the author's ordering and figure-style pass."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / "multi_target_scaling_latex"
BEFORE = ROOT / "reviewer_update/pre_figure_unification"

def read(path):
    return path.read_text(encoding="utf-8-sig")

def active(text):
    return re.sub(r"(?<!\\)%[^\n]*", "", text)

old = active("\n".join(read(BEFORE / name) for name in
    ["experiments_body.tex", "experiments_appendix.tex"]))
current = active("\n".join(read(LATEX / name) for name in
    ["experiments_body.tex", "experiments_appendix.tex"]))
labels = set(re.findall(r"\\label\{([^}]+)\}", old))
assert labels <= set(re.findall(r"\\label\{([^}]+)\}", current))
figures = r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}"
style_audit = json.loads(read(LATEX / "figure_style_audit.json"))
renamed = {"figures/" + item["replaces"]: "figures/" + item["figure"]
           for item in style_audit["figures"]}
assert Counter(renamed.get(path, path) for path in re.findall(figures, old)) == Counter(re.findall(figures, current))
assert all(re.fullmatch(r"figures/fig_(body|app)_[a-z0-9_]+\.pdf", path)
           for path in re.findall(figures, current))
data_inputs = r"\\input\{((?:figures|experiment_data)/[^}]+)\}"
assert Counter(re.findall(data_inputs, old)) == Counter(re.findall(data_inputs, current))
tables = r"\\begin\{tabular\}.*?\\end\{tabular\}"
assert re.findall(tables, old, re.DOTALL) == re.findall(tables, current, re.DOTALL)
restyled = {f"figures/{name}.pdf" for name in style_audit["restyled_real_figures"]}
assets = [path for folder in ["figures", "experiment_data"]
          for path in (BEFORE / folder).rglob("*") if path.is_file()]
for path in assets:
    relative = path.relative_to(BEFORE)
    if relative.as_posix() not in restyled:
        assert path.read_bytes() == (LATEX / relative).read_bytes(), relative
assert (BEFORE / "supplementary.tex").read_bytes() == (LATEX / "supplementary.tex").read_bytes()
author_body = read(BEFORE / "body.tex")
restored_body = author_body.replace("doing so involves technical challenges.}\n",
    "doing so involves technical challenges.}\n    " + r"\label{fig:joint-prediction}" + "\n")
assert read(LATEX / "body.tex") in (author_body, restored_body)
for item in style_audit["figures"]:
    for source in item["sources"]:
        assert hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest() == source["sha256"]
assert r"\input{experiments_legacy}" not in current
visible = re.sub(r"\\(?:label|ref|eqref)\{[^}]+\}", "", current)
for phrase in ["legacy", "retained original", "by the reviewers", "historical", "new reruns",
               "follow-up", "pre-instrumentation", "proof and endpoint-inequality"]:
    assert phrase not in visible.lower(), f"Editorial framing remains: {phrase}"
report = {
    "pre_edit_assets_preserved_unchanged": len(assets) - len(restyled),
    "real_figures_restyled_from_unchanged_summaries": len(restyled),
    "older_figures_restyled_under_new_names": len(renamed),
    "experimental_graphic_inclusions_preserved": len(re.findall(figures, old)),
    "experimental_data_inputs_preserved": len(re.findall(data_inputs, old)),
    "pre_edit_experimental_labels_preserved": len(labels),
    "inline_result_tables_unchanged": True,
    "all_experiments_in_two_tex_files": True,
    "all_active_experiment_figure_names_unified": True,
    "nonexperimental_prose_unchanged": True,
    "only_body_edit_is_previously_authorized_figure_label_restoration": True,
    "revision_log_framing_removed": True,
    "scope": "Narrative and plot-style edit; numerical source files and tables unchanged. Sampling provenance requires a separate audit.",
}
(LATEX / "publication_edit_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
