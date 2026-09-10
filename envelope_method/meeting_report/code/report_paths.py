"""Paths for the report after its numeric evidence moved to project data/."""
from pathlib import Path

REPORT = Path(__file__).resolve().parents[1]
PROJECT = REPORT.parents[1]
DATA = PROJECT / 'data/envelope_method/meeting_report'


def data_path(relative):
    path = DATA / Path(str(relative).replace('\\', '/'))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def artifact(value):
    """Resolve immutable report-relative manifest paths without rewriting them."""
    path = Path(str(value).replace('\\', '/'))
    original = path if path.is_absolute() else REPORT / path
    try:
        relative = original.resolve().relative_to(PROJECT)
    except ValueError:
        return original
    if relative.parts and relative.parts[0] == 'data':
        return original
    relocated = PROJECT / 'data' / relative
    return relocated if relocated.exists() else original
