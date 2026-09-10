"""Prepare a restricted, hashed cleanup plan; never moves or deletes files."""
from pathlib import Path
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import os
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STAGE = ROOT / 'deletable' / 'cleanup_2026-09-09'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

candidates = {}
def add(path, category, reason):
    path = path.resolve(strict=True)
    path.relative_to(ROOT)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f'Not an ordinary file: {path}')
    candidates[path] = (category, reason)

for folder in ('tmp/pdfs', 'reviewer_update/qa_renders', 'envelope_method/qa'):
    for path in (ROOT / folder).rglob('*'):
        if path.is_file() and path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            add(path, 'render_previews', 'Regenerable PDF review raster/contact sheet; source PDF and JSON/text QA retained.')

cache_dirs = ('__pycache__', 'utility/__pycache__', 'reviewer_update/__pycache__',
              'reviewer_update/pre_real_diagnostics/__pycache__', 'envelope_method/__pycache__',
              'envelope_method/cqr10/__pycache__', 'tmp/__pycache__')
for folder in cache_dirs:
    for path in (ROOT / folder).glob('*.pyc'):
        add(path, 'interpreter_cache', 'Regenerable Python bytecode; maintained source retained.')
for path in (ROOT / '.pytest_cache').rglob('*'):
    if path.is_file():
        add(path, 'test_cache', 'Regenerable pytest discovery and last-run cache.')
for folder in ('tmp/mpl', 'tmp/mpl-diagnostics'):
    for path in (ROOT / folder).glob('fontlist-*.json'):
        add(path, 'font_cache', 'Regenerable Matplotlib font cache; installed plotting packages retained.')
for stem in ('main', 'revision_cover'):
    for suffix in ('.fdb_latexmk', '.fls', '.synctex.gz'):
        path = ROOT / 'multi_target_scaling_latex' / (stem + suffix)
        if path.exists():
            add(path, 'editor_tracking', 'Regenerable editor/build tracking; TeX, PDF, aux, bibliography and log files retained.')

compiler_zip = ROOT / 'tmp/tectonic-0.17.0.zip'
if compiler_zip.exists():
    with zipfile.ZipFile(compiler_zip) as archive:
        member = next(name for name in archive.namelist() if name.endswith('tectonic.exe'))
        contained_hash = hashlib.sha256(archive.read(member)).hexdigest()
    assert contained_hash == digest(ROOT / 'tmp/tectonic-0.17.0/tectonic.exe')
    add(compiler_zip, 'duplicate_download', 'Contained compiler is byte-identical to retained extracted tectonic.exe; TeX cache retained.')

old_index = ROOT / 'output/project_audit_2026-09-09/file_inventory.json'
compressed_index = old_index.with_suffix('.json.gz')
if old_index.exists():
    with gzip.open(compressed_index, 'rb') as stream:
        assert hashlib.sha256(stream.read()).hexdigest() == digest(old_index)
    add(old_index, 'duplicate_inventory', 'Exact data retained in verified gzip index; audit tools accept compressed fallback.')

records = []
for path, (category, reason) in sorted(candidates.items()):
    relative = path.relative_to(ROOT)
    destination = STAGE / relative
    if destination.exists():
        raise FileExistsError(destination)
    stat = path.stat()
    records.append(dict(source=str(path), destination=str(destination), path=relative.as_posix(),
                        bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest(path),
                        category=category, reason=reason))

# Capture metadata of all research evidence/source. No large array contents are loaded.
protected = {}
extensions = {'.py', '.ipynb', '.tex', '.bib', '.sty', '.csv', '.json', '.npz', '.npy',
              '.joblib', '.pkl', '.arff', '.pdf', '.md'}
excluded = [ROOT / '.git', ROOT / 'docs', ROOT / 'deletable', ROOT / 'tmp/diagnostic_packages',
            ROOT / 'output/project_audit_2026-09-09']
for directory, dirs, files in os.walk(ROOT, followlinks=False):
    here = Path(directory)
    dirs[:] = [d for d in dirs if here / d not in excluded]
    for name in files:
        path = here / name
        if path in candidates or path.suffix.lower() not in extensions:
            continue
        # Navigation documentation is generated after the census, not scientific evidence.
        if name in ('README.md', 'INDEX.md') and ('results' in path.parts or path == ROOT / 'README.md'):
            continue
        stat = path.stat()
        protected[path.relative_to(ROOT).as_posix()] = [stat.st_size, stat.st_mtime_ns]
with gzip.open(OUT / 'protected_file_state.json.gz', 'wt', encoding='utf-8') as stream:
    json.dump(protected, stream, separators=(',', ':'))
source_hashes = {}
for folder in ('utility', 'envelope_method', 'reviewer_update', 'multi_target_scaling_latex'):
    for path in (ROOT / folder).rglob('*'):
        if path.is_file() and path.suffix.lower() in ('.py', '.ipynb', '.tex', '.bib', '.sty'):
            source_hashes[path.relative_to(ROOT).as_posix()] = digest(path)
(OUT / 'protected_source_hashes.json').write_text(json.dumps(source_hashes, indent=2), encoding='utf-8')
groups = defaultdict(lambda: {'files': 0, 'bytes': 0})
for row in records:
    groups[row['category']]['files'] += 1
    groups[row['category']]['bytes'] += row['bytes']
plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), workspace=str(ROOT),
            staging_root=str(STAGE), total_files=len(records), total_bytes=sum(r['bytes'] for r in records),
            groups=dict(groups), records=records)
(OUT / 'move_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in plan.items() if k != 'records'}, indent=2))
print(f'Protected metadata records: {len(protected)}; content-hashed scientific source files: {len(source_hashes)}')
