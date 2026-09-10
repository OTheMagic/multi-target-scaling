"""Move experiment data below data/ with an immutable, resumable hash manifest.

No file is deleted or overwritten. All paths are checked within this checkout.
Source code, experiment settings, LaTeX inputs, figures and prose stay portable.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / 'data' / '_organization' / '2026-09-10'
PLAN = STATE / 'manifest.json.gz'
SKIP = {'.git', '.agents', '.codex', 'tmp', 'deletable',
        '__pycache__', '.pytest_cache'}
DATA_SUFFIXES = {'.csv', '.tsv', '.npz', '.npy', '.arff', '.pkl', '.pickle',
                 '.parquet', '.feather', '.h5', '.hdf5', '.xlsx', '.xls', '.rds',
                 '.joblib', '.mat', '.rda', '.rdata', '.pt', '.pth'}
SOURCE_JSON = {
    'envelope_method/settings.json', 'envelope_method/notebook_settings.json',
    'envelope_method/repair_settings.json', 'envelope_method/toy_designs.json',
    'docs/cleanup/retention_policy.json',
    'output/cqr10_cluster_source_manifest.json',
}


def is_data(relative):
    p = Path(relative)
    if p.suffix.lower() in DATA_SUFFIXES:
        return True
    if 'source_snapshot' in p.parts:
        return False
    if p.as_posix() in SOURCE_JSON:
        return False
    if p.suffix.lower() in {'.json', '.jsonl'}:
        return True
    return (p.suffix.lower() in {'.gz', '.xml'} and
            p.as_posix().startswith(('docs/cleanup/', 'output/')))


@lru_cache(maxsize=None)
def safe_parent(parent):
    resolved = parent.resolve()
    if not resolved.is_relative_to(ROOT) or resolved != parent.absolute():
        raise ValueError(f'Linked or outside parent directory: {parent}')
    return resolved


def safe_path(relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts or any(
            p in {'.git', '.agents', '.codex'} for p in relative.parts):
        raise ValueError(f'Invalid or protected repository-relative path: {relative}')
    candidate = ROOT / relative
    resolved = safe_parent(candidate.parent) / candidate.name
    if not resolved.is_relative_to(ROOT) or resolved == ROOT:
        raise ValueError(f'Path is outside the repository: {relative}')
    if candidate.is_symlink() or (hasattr(candidate, 'is_junction') and candidate.is_junction()):
        raise ValueError(f'Linked or noncanonical path: {relative}')
    return resolved


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def candidates():
    for folder, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and
                         not (Path(folder) == ROOT and d == 'data'))
        for name in sorted(files):
            path = Path(folder) / name
            relative = path.relative_to(ROOT).as_posix()
            if is_data(relative):
                yield relative


def prepare():
    if PLAN.exists():
        raise FileExistsError(f'Existing migration plan: {PLAN}')
    names = list(candidates())
    # Preflight every destination before hashing or moving anything.
    for name in names:
        safe_path(name)
        target = safe_path('data/' + name)
        if target.exists():
            raise FileExistsError(target)
    def record(name):
        source = safe_path(name)
        return dict(source=name, destination='data/' + name,
                    bytes=source.stat().st_size, sha256=digest(source))
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(record, names))
    STATE.mkdir(parents=True, exist_ok=True)
    with gzip.open(PLAN, 'wt', encoding='utf-8') as stream:
        json.dump(dict(version=1, files=records), stream)
    print(json.dumps(dict(files=len(records), bytes=sum(r['bytes'] for r in records),
                          plan=str(PLAN))), flush=True)


def records():
    with gzip.open(PLAN, 'rt', encoding='utf-8') as stream:
        return json.load(stream)['files']


def move():
    rows = records()
    for i, row in enumerate(rows):
        source = safe_path(row['source'])
        target = safe_path(row['destination'])
        if not source.exists() and target.is_file():
            if target.stat().st_size != row['bytes'] or digest(target) != row['sha256']:
                raise ValueError(f'Resumed destination changed: {target}')
            continue
        if target.exists():
            raise FileExistsError(target)
        if source.stat().st_size != row['bytes'] or digest(source) != row['sha256']:
            raise ValueError(f'Source changed after planning: {source}')
        target.parent.mkdir(parents=True, exist_ok=True)
        # Windows rename fails if another process creates the destination.
        source.rename(target)
        if (i + 1) % 10000 == 0:
            print(f'Moved {i + 1}/{len(rows)} files', flush=True)
    print(f'Move complete: {len(rows)} files. Run verify.', flush=True)


def verify():
    rows = records()
    def check(row):
        source = safe_path(row['source'])
        target = safe_path(row['destination'])
        if source.exists():
            raise ValueError(f'Original file still present: {source}')
        if target.stat().st_size != row['bytes'] or digest(target) != row['sha256']:
            raise ValueError(f'Destination failed verification: {target}')
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(check, rows))
    result = dict(status='verified', files=len(rows),
                  bytes=sum(r['bytes'] for r in rows),
                  manifest_sha256=digest(PLAN), data_remaining_outside=list(candidates()))
    (STATE / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'move', 'verify'])
    parser.add_argument('--batch', choices=['primary', 'nested_data'], default='primary')
    args = parser.parse_args()
    if args.batch != 'primary':
        STATE = STATE / args.batch
        PLAN = STATE / 'manifest.json.gz'
    globals()[args.command]()
