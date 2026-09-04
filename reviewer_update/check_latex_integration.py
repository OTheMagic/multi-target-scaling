"""Audit the integrated manuscript without modifying its scientific content."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "multi_target_scaling_latex"
BEFORE = ROOT / "reviewer_update/pre_integration_latex"


def read(path):
    return path.read_text(encoding="utf-8")


def active(text):
    text = re.sub(r"(?<!\\)%[^\n]*", "", text)
    return text.split(r"\end{document}", 1)[0]


def expand(root, relative, seen=None):
    seen = set() if seen is None else seen
    path = root / relative
    if not path.suffix:
        path = path.with_suffix(".tex")
    assert path.exists(), f"Missing TeX input: {path}"
    assert path not in seen, f"Repeated/recursive input: {path}"
    seen.add(path)
    text = active(read(path))
    return re.sub(
        r"\\(?:input|include)\{([^}]+)\}",
        lambda m: expand(root, m.group(1), seen),
        text,
    )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


before_body = read(BEFORE / "body.tex")
current_body = read(CURRENT / "body.tex")
before_supp = read(BEFORE / "supplementary.tex")
current_supp = read(CURRENT / "supplementary.tex")
start = r"\section{Numerical Experiments}"
end = r"\section{Discussion and Future Work}"
assert before_body.split(start)[0] == current_body.split(r"\input{experiments_body}")[0]
assert before_body[before_body.index(end):] == current_body[current_body.index(end):]
assert before_supp.split(r"\section{Additional Numerical Results}")[0] == current_supp.split(
    r"\input{experiments_appendix}"
)[0]
for name in ["main.tex", "abstract.tex", "jmlr2e.sty"]:
    assert digest(BEFORE / name) == digest(CURRENT / name), f"Protected file changed: {name}"

original = expand(BEFORE, "main.tex")
combined = expand(CURRENT, "main.tex")
original_labels = set(re.findall(r"\\label\{([^}]+)\}", original))
labels = re.findall(r"\\label\{([^}]+)\}", combined)
label_counts = Counter(labels)
assert all(n == 1 for n in label_counts.values()), {
    k: v for k, v in label_counts.items() if v > 1
}
assert original_labels <= set(labels), f"Lost labels: {original_labels - set(labels)}"
references = set(re.findall(r"\\(?:ref|eqref|pageref)\{([^}]+)\}", combined))
assert references <= set(labels) | {"LastPage"}, f"Missing references: {references - set(labels)}"

image_pattern = r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}"
original_images = Counter(re.findall(image_pattern, original))
images = Counter(re.findall(image_pattern, combined))
assert all(images[k] >= v for k, v in original_images.items()), "An original image was removed"
for relative in images:
    path = CURRENT / relative
    if not path.suffix:
        path = path.with_suffix(".pdf")
    assert path.exists(), f"Missing figure: {path}"
    assert len(PdfReader(path).pages) > 0

old_figure_files = list((BEFORE / "figures").rglob("*"))
authorized_figure_updates = {
    "figures/joint_prediction.pdf", "figures/joint_prediction_draw.tex",
    "figures/partition.pdf", "figures/partition_draw.tex",
}
schematic_backup = ROOT / "reviewer_update/pre_schematic_labels"
for path in old_figure_files:
    if path.is_file():
        relative = path.relative_to(BEFORE)
        if relative.as_posix() in authorized_figure_updates:
            assert digest(path) == digest(schematic_backup / relative), path
            if path.suffix == ".tex":
                drawing = read(CURRENT / relative)
                unannotated = re.sub(
                    r"^% BEGIN SCHEMATIC ANNOTATION R2m[13]\n.*?"
                    r"^% END SCHEMATIC ANNOTATION R2m[13]\n", "", drawing,
                    flags=re.MULTILINE | re.DOTALL,
                )
                assert unannotated == read(path), f"Non-annotation drawing edit: {relative}"
        else:
            assert digest(path) == digest(CURRENT / relative), path
joint_drawing = read(CURRENT / "figures/joint_prediction_draw.tex")
assert all(label in joint_drawing for label in [r"92\%", r"93\%", "Coverage percentages are illustrative."])
assert 0.92 + 0.93 - 1 <= 0.90 <= min(0.92, 0.93)
partition_drawing = read(CURRENT / "figures/partition_draw.tex")
assert r"at (\vone,-0.12) {$\mathcal{L}_1$}" in partition_drawing
assert r"at (-0.12,\vtwo) {$\mathcal{L}_2$}" in partition_drawing
for path in (ROOT / "reviewer_update/figures").glob("*.pdf"):
    assert digest(path) == digest(CURRENT / "figures" / path.name), path

citation_keys = set()
for group in re.findall(r"\\cite[a-zA-Z*]*(?:\[[^]]*\])*\{([^}]+)\}", combined):
    citation_keys.update(k.strip() for k in group.split(","))
bibkeys = set(re.findall(r"@\w+\s*\{\s*([^,]+)", read(CURRENT / "biblio.bib")))
assert citation_keys <= bibkeys, f"Missing bibliography keys: {citation_keys - bibkeys}"

environments = []
for match in re.finditer(r"\\(begin|end)\{([^}]+)\}", combined):
    direction, name = match.groups()
    if direction == "begin":
        environments.append(name)
    else:
        assert environments and environments.pop() == name, match.group()
assert environments == ["document"], environments

response = read(CURRENT / "response_to_reviewers.tex")
points = re.findall(r"\\point\{([^}]+)\}\{([^}]+)\}", response)
assert len(points) == 22, points
statuses = Counter(status for heading, status in points if heading.startswith("R"))
assert statuses == {"Addressed": 13, "Partially addressed": 5, "Open": 2}, statuses
response_refs = set(re.findall(r"\\(?:mssec|msapp|msfig|mstab)\{([^}]+)\}", response))
response_refs.update(re.findall(r"\\ref\*?\{M-([^}]+)\}", response))
response_refs.discard("#1")
assert response_refs <= set(labels), response_refs - set(labels)

report = {
    "protected_nonexperimental_sections_unchanged": True,
    "protected_main_abstract_style_unchanged": True,
    "original_labels_preserved": len(original_labels),
    "total_unique_labels": len(label_counts),
    "original_graphic_inclusions_preserved": sum(original_images.values()),
    "new_figures_copied_unchanged": len(list((ROOT / "reviewer_update/figures").glob("*.pdf"))),
    "original_figure_files_unchanged": sum(p.is_file() for p in old_figure_files) - len(authorized_figure_updates),
    "authorized_schematic_updates": sorted(authorized_figure_updates),
    "schematic_sources_unchanged_except_annotation_blocks": True,
    "pre_edit_schematics_preserved": True,
    "original_table_labels_preserved": sorted(k for k in original_labels if k.startswith("tab:")),
    "missing_tex_inputs": [],
    "missing_graphics": [],
    "unresolved_internal_references": [],
    "unresolved_citations": [],
    "response_blocks": len(points),
    "reviewer_point_statuses": dict(statuses),
    "unresolved_response_manuscript_references": [],
    "scope": "Static integration audit; compilation and visual QA are checked separately.",
}
(CURRENT / "integration_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
