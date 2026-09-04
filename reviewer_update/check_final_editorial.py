"""Check the final copyedit against the author's latest source snapshot."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "multi_target_scaling_latex"
BEFORE = ROOT / "reviewer_update/pre_final_editorial"


def read(path):
    return path.read_text(encoding="utf-8-sig")


def active(text):
    return re.sub(r"(?<!\\)%[^\n]*", "", text).split(r"\end{document}", 1)[0]


def expand(root, relative, seen=None):
    seen = set() if seen is None else seen
    path = root / relative
    path = path if path.suffix else path.with_suffix(".tex")
    assert path.is_file() and path not in seen, path
    seen.add(path)
    return re.sub(r"\\(?:input|include)\{([^}]+)\}",
                  lambda m: expand(root, m[1], seen), active(read(path)))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def labels(text):
    return re.findall(r"\\label\{([^}]+)\}", text)


def figures(text):
    return re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text)


def validate():
    before = expand(BEFORE, "main.tex")
    current = expand(CURRENT, "main.tex")
    label_counts = Counter(labels(current))
    assert all(count == 1 for count in label_counts.values()), label_counts
    assert set(labels(before)) <= set(label_counts), "An author label was removed"
    references = set(re.findall(r"\\(?:ref|eqref|pageref)\*?\{([^}]+)\}", current))
    assert references <= set(label_counts) | {"LastPage"}, references - set(label_counts)
    assert Counter(figures(before)) == Counter(figures(current)), "Figure inclusions changed"
    before_tables = re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", before, re.S)
    tables = re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", current, re.S)
    assert [t.replace("Emp. copula", "Emp. Copula") for t in before_tables] == tables

    assets = [p for folder in ("figures", "experiment_data")
              for p in (BEFORE / folder).rglob("*") if p.is_file()]
    for source in assets:
        relative = source.relative_to(BEFORE)
        destination = CURRENT / relative
        assert destination.is_file(), relative
        if relative.as_posix() == "figures/toy_real.tex":
            assert read(source).replace("Emp. copula", "Emp. Copula") == read(destination)
        else:
            assert digest(source) == digest(destination), relative
    assert digest(BEFORE / "biblio.bib") == digest(CURRENT / "biblio.bib")
    assert digest(BEFORE / "jmlr2e.sty") == digest(CURRENT / "jmlr2e.sty")

    bibkeys = set(re.findall(r"@\w+\s*\{\s*([^,]+)", read(CURRENT / "biblio.bib")))
    citations = {k.strip() for group in re.findall(
        r"\\cite[a-zA-Z*]*(?:\[[^]]*\])*\{([^}]+)\}", current) for k in group.split(",")}
    assert citations <= bibkeys, citations - bibkeys

    response = read(CURRENT / "response_to_reviewers.tex")
    points = re.findall(r"\\point\{([^}]+)\}\{([^}]+)\}", response)
    assert len(points) == 22
    statuses = Counter(status for name, status in points if name.startswith("R"))
    assert statuses == {"Addressed": 14, "Partially addressed": 5, "Open": 1}, statuses
    for name in ("response_to_reviewers.tex", "revision_cover.tex"):
        text = read(CURRENT / name)
        refs = set(re.findall(r"\\(?:mssec|msapp|msfig|mstab)\{([^}]+)\}", text))
        refs.update(re.findall(r"\\ref\*?\{M-([^}]+)\}", text))
        refs = {ref for ref in refs if not ref.startswith("#")}
        assert refs <= set(label_counts), (name, refs - set(label_counts))

    inventory = json.loads(read(CURRENT / "experiment_inventory.json"))
    studies = inventory["studies"]
    assert len(studies) == inventory["total_added_studies"] == 12
    assert Counter(s["kind"] for s in studies) == {"simulation": 10, "real-data diagnostic": 2}
    study_figures = [f for s in studies for f in s["figures"]]
    assert len(study_figures) == len(set(study_figures)) == 16
    added = set(study_figures) | {inventory["additional_reanalysis"]["figure"]}
    assert len(added) == 17
    experimental = figures(active(read(CURRENT / "experiments_body.tex")) +
                           active(read(CURRENT / "experiments_appendix.tex")))
    assert len(experimental) == 25
    assert added <= {Path(f).name for f in experimental}
    assert len({Path(f).name for f in experimental} - added) == 8
    for study in studies:
        assert study["location"] in label_counts
        assert all((ROOT / source).is_file() for source in study["data"]), study["id"]
        assert study["id"] in read(CURRENT / "revision_cover.tex"), study["id"]

    tagged = []
    for name in ("main.tex", "body.tex", "supplementary.tex",
                 "experiments_body.tex", "experiments_appendix.tex"):
        for line_number, line in enumerate(read(CURRENT / name).splitlines(), 1):
            found = re.search(r"% AUTHOR-CHECK ([A-Z]+): (.*)", line)
            if found:
                tagged.append(dict(tag=found[1], file=name, line=line_number, note=found[2]))
    assert len({item["tag"] for item in tagged}) == 16
    concerns = read(CURRENT / "unresolved_reviewer_concerns.md")
    assert all("AUTHOR-CHECK " + item["tag"] in concerns for item in tagged)

    documents = {}
    for name in ("main", "response_to_reviewers", "revision_cover"):
        pdf = PdfReader(CURRENT / (name + ".pdf"))
        text = "\n".join(page.extract_text() for page in pdf.pages)
        assert "??" not in text, (name, "Unresolved printed reference")
        if name == "main":
            assert "AUTHOR-CHECK" not in text, "Author comments leaked into paper"
        documents[name] = {"pages": len(pdf.pages), "sha256": digest(CURRENT / (name + ".pdf"))}
    for relative in figures(current):
        assert len(PdfReader(CURRENT / relative).pages) == 1, relative

    report = {
        "baseline": str(BEFORE.relative_to(ROOT)),
        "scope": "Full language/notation copyedit; substantive scientific questions remain tagged.",
        "author_labels_preserved": len(set(labels(before))),
        "current_labels": len(label_counts),
        "experimental_pdf_inclusions_preserved": len(experimental),
        "all_graphic_inclusions_preserved": len(figures(current)),
        "all_figure_pdfs_and_csvs_unchanged": True,
        "assets_checked": len(assets),
        "numerical_tables_unchanged": True,
        "only_table_text_change": "Emp. copula -> Emp. Copula in the energy illustration",
        "missing_references": [], "missing_citations": [],
        "reviewer_point_statuses": dict(statuses),
        "added_studies": 12, "added_figure_assets": 17,
        "author_check_tags": tagged, "documents": documents,
        "scientific_submission_readiness": "Not certified; author decisions remain open.",
    }
    (CURRENT / "final_editorial_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "author_check_tags"}, indent=2))


if __name__ == "__main__":
    validate()
