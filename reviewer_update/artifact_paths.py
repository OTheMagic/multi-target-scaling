"""Enumerate retained document assets together with relocated numerical inputs."""
from pathlib import Path

from utility.project_paths import ROOT, resolve_artifact


def iter_artifact_files(directory: Path, *, root: Path = ROOT):
    """Yield (relative name, actual file) once across the source/data trees.

    Historical preservation audits must still check moved CSV/JSON files rather
    than silently omitting them when only the document assets remain in place.
    """
    directory = Path(directory)
    relative_directory = directory.relative_to(root)
    names = set()
    for candidate in (directory, root / "data" / relative_directory):
        if candidate.is_dir():
            names.update(path.relative_to(candidate) for path in candidate.rglob("*")
                         if path.is_file())
    for name in sorted(names):
        yield name, resolve_artifact(directory / name, root=root)
