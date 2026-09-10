"""Regression coverage for audits of relocated and retained document inputs."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reviewer_update.artifact_paths import iter_artifact_files


def test_split_document_and_data_tree_keeps_all_audit_inputs(tmp_path):
    original = tmp_path / "reviewer_update/pre_final_editorial/experiment_data"
    relocated = tmp_path / "data/reviewer_update/pre_final_editorial/experiment_data"
    original.mkdir(parents=True)
    (relocated / "real_diagnostics").mkdir(parents=True)
    (original / "table.tex").write_text("retained table")
    (relocated / "real_diagnostics/trials.csv").write_text("trial,value\n0,1\n")
    (relocated / "real_diagnostics/manifest.json").write_text("{}")
    found = dict(iter_artifact_files(original, root=tmp_path))
    assert set(found) == {Path("table.tex"), Path("real_diagnostics/trials.csv"),
                          Path("real_diagnostics/manifest.json")}
    assert found[Path("table.tex")] == original / "table.tex"
    assert found[Path("real_diagnostics/trials.csv")] == relocated / "real_diagnostics/trials.csv"


def test_relocated_copy_is_checked_once_when_legacy_copy_also_exists(tmp_path):
    original = tmp_path / "reviewer_update/data"
    relocated = tmp_path / "data/reviewer_update/data"
    original.mkdir(parents=True)
    relocated.mkdir(parents=True)
    (original / "trials.csv").write_text("legacy")
    (relocated / "trials.csv").write_text("central")
    assert list(iter_artifact_files(original, root=tmp_path)) == [
        (Path("trials.csv"), relocated / "trials.csv")]


def test_fully_moved_directory_is_not_silently_skipped(tmp_path):
    original = tmp_path / "reviewer_update/data"
    relocated = tmp_path / "data/reviewer_update/data"
    relocated.mkdir(parents=True)
    (relocated / "summary.csv").write_text("value\n1\n")
    assert list(iter_artifact_files(original, root=tmp_path)) == [
        (Path("summary.csv"), relocated / "summary.csv")]
