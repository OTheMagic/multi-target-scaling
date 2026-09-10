"""Read-only static audit of the project's active LaTeX asset dependencies.

Run with Python from any directory. Prints JSON; does not compile, regenerate,
move, or edit assets. External TeX-distribution packages are listed separately.
This checks literal dependency commands rather than executing the TeX language.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROOTS = [
    ("manuscript", "multi_target_scaling_latex/main.tex", "multi_target_scaling_latex"),
    ("manuscript_wrapper", "multi_target_scaling_latex/compile_main.tex", "multi_target_scaling_latex"),
    ("response", "multi_target_scaling_latex/response_to_reviewers.tex", "multi_target_scaling_latex"),
    ("revision_cover", "multi_target_scaling_latex/revision_cover.tex", "multi_target_scaling_latex"),
    ("envelope_report", "envelope_method/signed_envelope.tex", "envelope_method"),
    ("envelope_report_template", "envelope_method/report_revision/report_main.tex", "envelope_method/report_revision"),
    ("meeting_report", "envelope_method/meeting_report/report.tex", "envelope_method/meeting_report"),
    ("portable_envelope_report", "output/pdf/envelope_shortcut_report/signed_envelope.tex", "output/pdf/envelope_shortcut_report"),
    ("portable_outlier_report", "output/pdf/outlier_sensitivity_report/outlier_sensitivity_report.tex", "output/pdf/outlier_sensitivity_report"),
]
COMMAND = re.compile(
    r"\\(input|include|includegraphics|bibliography|bibliographystyle|addbibresource|"
    r"usepackage|RequirePackage|documentclass|externaldocument)\*?\s*"
    r"(?:\[[^\]]*\]\s*)?\{([^{}]+)\}"
)
GRAPHICSPATH = re.compile(r"\\graphicspath\s*\{((?:\s*\{[^{}]*\})+)\s*\}")
DATA_COMMAND = re.compile(
    r"\\(?:pgfplotstableread|csvreader|csvautotabular|DTLloaddb|read|openin|"
    r"lstinputlisting|verbatiminput|includepdf|import|subimport|inputminted)\b"
)


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def content(path: Path) -> str:
    return re.sub(r"(?<!\\)%[^\n]*", "", path.read_text(encoding="utf-8", errors="replace"))


def audit(label: str, source: str, working_directory: str) -> dict:
    cwd = ROOT / working_directory
    queue = [ROOT / source]
    scanned = set()
    assets = set()
    dependencies = []
    missing = []
    standard = set()
    graphicpaths = []
    review_commands = []
    dynamic = []
    external = []
    while queue:
        path = queue.pop(0).resolve()
        if path in scanned:
            continue
        if not path.is_file():
            missing.append({"source": relative(path), "command": "entry", "target": relative(path)})
            continue
        scanned.add(path)
        assets.add(path)
        text = content(path)
        for match in GRAPHICSPATH.finditer(text):
            for value in re.findall(r"\{([^{}]*)\}", match.group(1)):
                candidate = cwd / value
                if candidate not in graphicpaths:
                    graphicpaths.append(candidate)
        for match in DATA_COMMAND.finditer(text):
            review_commands.append({"source": relative(path), "command": match.group()})
        for match in COMMAND.finditer(text):
            command, target = match.groups()
            if "\\" in target or "#" in target:
                dynamic.append({"source": relative(path), "command": command, "target": target})
                continue
            targets = target.split(",") if command in {"bibliography", "usepackage", "RequirePackage"} else [target]
            for target in map(str.strip, targets):
                suffixes = {
                    "input": ["", ".tex"], "include": ["", ".tex"],
                    "includegraphics": ["", ".pdf", ".png", ".jpg", ".jpeg", ".eps"],
                    "bibliography": ["", ".bib"], "addbibresource": ["", ".bib"],
                    "bibliographystyle": ["", ".bst"], "documentclass": ["", ".cls"],
                    "usepackage": ["", ".sty"], "RequirePackage": ["", ".sty"],
                    "externaldocument": [".aux"],
                }[command]
                directories = [cwd] + (graphicpaths if command == "includegraphics" else [])
                candidates = [(base / (target + ext)).resolve() for base in directories for ext in suffixes]
                resolved = next((candidate for candidate in candidates if candidate.is_file()), None)
                record = {"source": relative(path), "command": command, "target": target}
                if command in {"documentclass", "usepackage", "RequirePackage", "bibliographystyle"} and resolved is None:
                    standard.add(f"{command}:{target}")
                    continue
                if command == "externaldocument":
                    external.append({**record, "resolved": relative(resolved) if resolved else None,
                                     "note": "Generated by compiling main first; absence affects cross-references."})
                    continue
                if resolved is None:
                    missing.append(record)
                    continue
                assets.add(resolved)
                dependencies.append({**record, "resolved": relative(resolved)})
                if command in {"input", "include", "usepackage", "RequirePackage"}:
                    queue.append(resolved)
    data_root = (ROOT / "data").resolve()
    root_data = sorted(relative(path) for path in assets if path == data_root or data_root in path.parents)
    return {
        "name": label, "entry": source, "working_directory": working_directory,
        "resolved_dependency_references": len(dependencies),
        "unique_assets_including_entry": len(assets),
        "asset_extensions": dict(sorted(Counter(path.suffix for path in assets).items())),
        "root_data_dependencies": root_data,
        "missing_literal_dependencies": missing,
        "external_cross_reference_aux": external,
        "tex_distribution_dependencies": sorted(standard),
        "dynamic_targets_for_review": dynamic,
        "other_dependency_commands_for_review": review_commands,
        "dependencies": dependencies,
    }


if __name__ == "__main__":
    results = [audit(*item) for item in ROOTS]
    report = {
        "scope": "Active literal LaTeX file dependencies; no compilation or numerical data access.",
        "caveats": [
            "Standard TeX distribution packages, fonts, bibliography styles and compiler availability are not tested.",
            "This is static analysis of literal commands, not arbitrary TeX macro execution.",
            "report_revision/report_main.tex has local figure and table inputs and is checked from its own parent directory.",
            "Response and cover cross-references need a prior main build; generated AUX files are not source inputs.",
            "Textual data-path mentions and provenance links are not compile-time dependencies.",
        ],
        "roots": results,
        "all_literal_assets_resolve": all(not result["missing_literal_dependencies"] for result in results),
        "no_root_data_dependency": all(not result["root_data_dependencies"] for result in results),
        "no_dependency_commands_require_review": all(
            not result["dynamic_targets_for_review"] and not result["other_dependency_commands_for_review"]
            for result in results
        ),
    }
    print(json.dumps(report, indent=2))
    sys.exit(0 if all(report[key] for key in (
        "all_literal_assets_resolve", "no_root_data_dependency", "no_dependency_commands_require_review"
    )) else 1)
