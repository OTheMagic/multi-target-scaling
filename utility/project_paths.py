"""Separate portable source/figure assets from local experiment data.

Data mirrors its original repository-relative path below ``data/``. Saved
provenance strings remain immutable; resolve_artifact locates their moved files.
"""
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / 'data'
# Recorded in the original experiment manifests; keep this explicit instead of
# guessing arbitrary suffixes of an unrelated absolute path.
LEGACY_REPOSITORY_PREFIX = 'E:/multi-target-scaling/'


def data_path(*parts):
    """Return a local data path; callers create output directories as needed."""
    relative = Path(*[str(part).replace('\\', '/') for part in parts])
    if relative.anchor or PureWindowsPath(str(relative)).drive or '..' in relative.parts:
        raise ValueError('data_path requires a repository-relative path without ..')
    if relative.parts and relative.parts[0] == 'data':
        relative = Path(*relative.parts[1:])
    return DATA_ROOT / relative


def resolve_artifact(value, *, root=ROOT):
    """Find a saved source reference without rewriting historical metadata.

    Prefer its centralized copy when present, otherwise retain the old path.
    ``root`` supports independent archives and test fixtures outside this repo.
    """
    root = Path(root).resolve()
    text = str(value).replace('\\', '/')
    if text.casefold().startswith(LEGACY_REPOSITORY_PREFIX.casefold()):
        text = text[len(LEGACY_REPOSITORY_PREFIX):]
    candidate = Path(text)
    original = candidate if candidate.is_absolute() else root / candidate
    try:
        relative = original.resolve().relative_to(root)
    except ValueError:
        return original
    if relative.parts and relative.parts[0] == 'data':
        return original
    relocated = root / 'data' / relative
    compact_checkpoint = (relocated.suffix == '.npz' and
                          relocated.with_suffix('.json').exists())
    return relocated if relocated.exists() or compact_checkpoint else original
