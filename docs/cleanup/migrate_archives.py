"""Safely replace full simulation NPZs with score archives; stage originals.

All paths are resolved and constrained to this workspace before any individual
file move. No recursive deletion is performed. A persistent per-trial transaction
marker allows an interrupted commit to be finished on rerun. Existing originals
and JSON checkpoints are moved intact under deletable, never overwritten.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from envelope_method.archive_storage import (RAW_OBSERVATION_KEYS, get_storage_mode,
    records_sha256, sha256_file, validate_archive)

WORK = ROOT / 'docs/cleanup/archive_migration_2026-09-09'
STAGE = ROOT / 'deletable/archive_migration_2026-09-09'
PLAN = WORK / 'plan.json.gz'


def checked(path, within=ROOT):
    path = Path(path).absolute()
    resolved = path.resolve()
    boundary = Path(within).resolve()
    if not resolved.is_relative_to(boundary) or resolved == boundary:
        raise ValueError(f'Path outside intended directory: {path}; resolved={resolved}; boundary={boundary}')
    # Refuse redirected ancestors even when a link happens to remain in workspace.
    ancestor = path
    while ancestor != ROOT and ancestor != ancestor.parent:
        if ancestor.is_symlink() or (hasattr(ancestor, 'is_junction') and ancestor.is_junction()):
            raise ValueError(f'Redirected filesystem path: {ancestor}')
        ancestor = ancestor.parent
    if any(part in {'.git', '.codex', '.agents'} for part in resolved.relative_to(ROOT).parts):
        raise ValueError(f'Protected workspace path: {path}')
    return resolved


def atomic_json(path, data):
    path = checked(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checked(path.with_name(path.name + '.writing'))
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_plan():
    with gzip.open(PLAN, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def prepare():
    if PLAN.exists():
        raise FileExistsError(f'Existing plan: {PLAN}. Use stage/verify to resume it.')
    policy_path = ROOT / 'docs/cleanup/retention_policy.json'
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    if policy['default_mode'] != 'scores' or set(policy['drop_members']) != RAW_OBSERVATION_KEYS:
        raise ValueError('Unsupported retention policy.')
    results = checked(ROOT / policy['results_root'])
    keep = {(entry['kind'], entry['config_id']): entry['trials'] for entry in policy['keep_full']}
    if set(policy['scope']) != {'absolute', 'cqr'}:
        raise ValueError('Migration scope must be the reviewed absolute and cqr directories.')
    rows, preserved = [], []
    for kind in policy['scope']:
        for folder in sorted((results / kind).iterdir()):
            if not folder.is_dir():
                continue
            archives = sorted(folder.glob('trial_*.npz'))
            if not archives:
                continue
            if (kind, folder.name) not in keep:
                raise ValueError(f'Unreviewed configuration: {folder}')
            selected = keep[(kind, folder.name)]
            for archive in archives:
                archive = checked(archive, results / kind)
                trial = int(archive.stem.split('_')[-1])
                if selected == 'all' or trial in selected:
                    preserved.append(dict(path=archive.relative_to(ROOT).as_posix(),
                                          bytes=archive.stat().st_size, sha256=sha256_file(archive)))
                    continue
                checkpoint_path = checked(archive.with_suffix('.json'))
                checkpoint = json.loads(checkpoint_path.read_text(encoding='utf-8'))
                if checkpoint.get('version') != 2 or get_storage_mode(checkpoint) != 'full':
                    raise ValueError(f'Unexpected checkpoint format: {checkpoint_path}')
                original_hash = sha256_file(archive)
                if original_hash != checkpoint['archive_sha256']:
                    raise ValueError(f'Original archive hash differs from checkpoint: {archive}')
                with zipfile.ZipFile(archive) as source:
                    names = source.namelist()
                    removed = [key for key in names if key.removesuffix('.npy') in RAW_OBSERVATION_KEYS]
                    if len(removed) != 6 or len(set(names)) != len(names):
                        raise ValueError(f'Unexpected archive members: {archive}')
                    retained = sorted(set(names) - set(removed))
                relative = archive.relative_to(ROOT).as_posix()
                staged = checked(STAGE / relative, STAGE)
                if staged.exists() or staged.with_suffix('.json').exists():
                    raise FileExistsError(f'Destination already exists: {staged}')
                rows.append(dict(path=relative, bytes=archive.stat().st_size, sha256=original_hash,
                                 json_sha256=sha256_file(checkpoint_path),
                                 records_sha256=records_sha256(checkpoint['records']),
                                 retained_members=retained, removed_members=sorted(removed)))
        print(f'Prepared {kind}: {len(rows)} replacements; {len(preserved)} full retained', flush=True)
    # Every current table is protected, including sidecars/real summaries outside scope.
    tables = [dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size, sha256=sha256_file(path))
              for path in sorted(results.rglob('*.csv'))]
    data = dict(schema_version=1, workspace=str(ROOT), staging_root=str(STAGE),
                created_utc=datetime.now(timezone.utc).isoformat(),
                policy_sha256=sha256_file(policy_path), records=rows, preserved_full=preserved,
                protected_tables=tables)
    checked(WORK).mkdir(parents=True, exist_ok=True)
    with gzip.open(PLAN, 'wt', encoding='utf-8') as stream:
        json.dump(data, stream)
    summary = dict(status='prepared', replacements=len(rows), original_bytes=sum(row['bytes'] for row in rows),
                   preserved_full=len(preserved), preserved_full_bytes=sum(row['bytes'] for row in preserved),
                   protected_tables=len(tables), policy_sha256=data['policy_sha256'])
    atomic_json(WORK / 'prepared.json', summary)
    print(json.dumps(summary), flush=True)


def payload_digest(archive, names):
    digest = hashlib.sha256()
    for name in sorted(names):
        value = archive.read(name)
        digest.update(name.encode() + b'\0' + hashlib.sha256(value).digest())
    return digest.hexdigest()


def prepare_replacement(source_path, temporary, row):
    """Keep exact NPY payloads and record non-reconstructive removed-data witnesses."""
    removed = {}
    with zipfile.ZipFile(source_path) as source, zipfile.ZipFile(
            temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as target:
        digest = hashlib.sha256()
        for name in row['retained_members']:
            value = source.read(name)
            digest.update(name.encode() + b'\0' + hashlib.sha256(value).digest())
            target.writestr(name, value)
        for name in row['removed_members']:
            value = source.read(name)
            stream = io.BytesIO(value)
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, order, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, order, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f'Unsupported NPY header version {version}: {source_path}/{name}')
            removed[name.removesuffix('.npy')] = dict(shape=list(shape), dtype=str(dtype),
                fortran_order=order, npy_sha256=hashlib.sha256(value).hexdigest(), npy_bytes=len(value))
    payload_sha256 = digest.hexdigest()
    with zipfile.ZipFile(temporary) as target:
        if payload_digest(target, row['retained_members']) != payload_sha256:
            raise ValueError(f'Retained numerical payload changed: {temporary}')
    return dict(schema_version=1, mode='scores', source_archive_sha256=row['sha256'],
                archive_sha256=sha256_file(temporary), archive_bytes=temporary.stat().st_size,
                retained_keys=[name.removesuffix('.npy') for name in row['retained_members']],
                removed_arrays=removed, records_sha256=row['records_sha256'],
                retained_payload_sha256=payload_sha256,
                migration_policy_sha256=sha256_file(ROOT / 'docs/cleanup/retention_policy.json'))


def stage_one(row):
    original = checked(ROOT / row['path'], ROOT / 'envelope_method/results')
    checkpoint_path = checked(original.with_suffix('.json'))
    backup = checked(STAGE / row['path'], STAGE)
    backup_json = checked(backup.with_suffix('.json'), STAGE)
    transaction = checked(backup.with_suffix('.transaction.json'), STAGE)
    temporary = checked(original.with_suffix('.scores.pending.npz'))
    if transaction.exists():
        pending = json.loads(transaction.read_text(encoding='utf-8'))
        if pending['source_sha256'] != row['sha256']:
            raise ValueError(f'Transaction belongs to a different source: {transaction}')
        if pending['status'] == 'restored':
            raise ValueError(f'This transaction was restored; create a new reviewed migration instead of overwriting it: {transaction}')
        if pending['status'] == 'complete':
            if sha256_file(original) != pending['checkpoint']['archive_sha256'] or sha256_file(checkpoint_path) != pending['active_checkpoint_sha256']:
                raise ValueError(f'Completed replacement changed after migration: {original}')
            if sha256_file(backup) != row['sha256'] or sha256_file(backup_json) != row['json_sha256']:
                raise ValueError(f'Completed original backup changed: {backup}')
            return dict(path=row['path'], old_bytes=pending['old_bytes'], new_bytes=pending['new_bytes'])
    else:
        if backup.exists() or backup_json.exists():
            raise FileExistsError(f'Unjournaled staging collision: {backup}')
        if sha256_file(original) != row['sha256'] or sha256_file(checkpoint_path) != row['json_sha256']:
            raise ValueError(f'Source changed after preparation: {original}')
        checkpoint = json.loads(checkpoint_path.read_text(encoding='utf-8'))
        if records_sha256(checkpoint['records']) != row['records_sha256']:
            raise ValueError(f'Trial measurements changed: {checkpoint_path}')
        storage = prepare_replacement(original, temporary, row)
        checkpoint['archive_sha256'] = storage['archive_sha256']
        checkpoint['storage'] = storage
        pending = dict(status='prepared', source_sha256=row['sha256'], checkpoint=checkpoint,
                       old_bytes=row['bytes'], new_bytes=storage['archive_bytes'],
                       payload_sha256=storage['retained_payload_sha256'])
        backup.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(transaction, pending)
    # Validate/rebuild the pending replacement BEFORE moving either original.
    already_installed = original.exists() and sha256_file(original) == pending['checkpoint']['archive_sha256']
    if not already_installed and (not temporary.exists() or sha256_file(temporary) != pending['checkpoint']['archive_sha256']):
        source_path = backup if backup.exists() else original
        if sha256_file(source_path) != row['sha256']:
            raise ValueError(f'Cannot recover pending archive from altered source: {source_path}')
        storage = prepare_replacement(source_path, temporary, row)
        pending['checkpoint']['storage'] = storage
        pending['checkpoint']['archive_sha256'] = storage['archive_sha256']
        pending['new_bytes'] = storage['archive_bytes']
        pending['payload_sha256'] = storage['retained_payload_sha256']
        atomic_json(transaction, pending)
    # Each operation below is independently recoverable after interruption.
    if not backup.exists():
        if sha256_file(original) != row['sha256']:
            raise ValueError(f'Original changed before move: {original}')
        original.rename(backup)
    elif sha256_file(backup) != row['sha256']:
        raise ValueError(f'Staged original checksum mismatch: {backup}')
    if not backup_json.exists():
        if sha256_file(checkpoint_path) != row['json_sha256']:
            raise ValueError(f'Original checkpoint changed before move: {checkpoint_path}')
        checkpoint_path.rename(backup_json)
    elif sha256_file(backup_json) != row['json_sha256']:
        raise ValueError(f'Staged checkpoint checksum mismatch: {backup_json}')
    if temporary.exists():
        if original.exists():
            raise FileExistsError(f'Refusing to overwrite active replacement: {original}')
        if sha256_file(temporary) != pending['checkpoint']['archive_sha256']:
            raise ValueError(f'Pending replacement changed: {temporary}')
        temporary.rename(original)
    elif not original.exists():
        raise FileNotFoundError(f'Missing prepared replacement: {temporary}')
    if sha256_file(original) != pending['checkpoint']['archive_sha256']:
        raise ValueError(f'Active replacement checksum mismatch: {original}')
    atomic_json(checkpoint_path, pending['checkpoint'])
    validate_archive(original, pending['checkpoint'], verify_hash=False)
    pending['status'] = 'complete'
    pending['active_checkpoint_sha256'] = sha256_file(checkpoint_path)
    atomic_json(transaction, pending)
    return dict(path=row['path'], old_bytes=pending['old_bytes'], new_bytes=pending['new_bytes'])


def stage(workers):
    plan = read_plan()
    if Path(plan['workspace']).resolve() != ROOT or Path(plan['staging_root']).resolve() != STAGE:
        raise ValueError('Plan workspace/staging roots differ.')
    if sha256_file(ROOT / 'docs/cleanup/retention_policy.json') != plan['policy_sha256']:
        raise ValueError('Retention policy changed after preparation.')
    # Resolve ALL targets before the first move.
    for row in plan['records']:
        checked(ROOT / row['path'], ROOT / 'envelope_method/results')
        checked(STAGE / row['path'], STAGE)
    # Create checked destination parents serially. Windows path resolution can
    # otherwise race with another worker creating the same ancestor chain.
    for parent in sorted({(STAGE / row['path']).parent for row in plan['records']}):
        checked(parent, STAGE).mkdir(parents=True, exist_ok=True)
    completed = []
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(stage_one, row) for row in plan['records']]
        for future in as_completed(futures):
            try:
                completed.append(future.result())
            except Exception as exc:
                errors.append(repr(exc))
                print('Migration operation failed safely: ' + repr(exc), flush=True)
                continue
            if len(completed) % 500 == 0:
                progress = dict(status='running', completed=len(completed), total=len(plan['records']))
                atomic_json(WORK / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
    if errors:
        atomic_json(WORK / 'stage_errors.json', dict(status='needs_recovery', completed=len(completed), errors=errors))
        raise RuntimeError(f'{len(errors)} migration operations need recovery; originals and transactions are preserved. See stage_errors.json.')
    report = dict(status='staged', trials=len(completed),
                  original_npz_bytes=sum(row['old_bytes'] for row in completed),
                  active_score_npz_bytes=sum(row['new_bytes'] for row in completed),
                  active_npz_reduction_bytes=sum(row['old_bytes']-row['new_bytes'] for row in completed),
                  staging_root=str(STAGE), deleted_files=0)
    atomic_json(WORK / 'stage_result.json', report)
    print(json.dumps(report), flush=True)


def verify_one(row):
    active = checked(ROOT / row['path'])
    backup = checked(STAGE / row['path'], STAGE)
    checkpoint = json.loads(active.with_suffix('.json').read_text(encoding='utf-8'))
    if sha256_file(backup) != row['sha256'] or sha256_file(backup.with_suffix('.json')) != row['json_sha256']:
        raise ValueError(f'Original backup changed: {backup}')
    if records_sha256(checkpoint['records']) != row['records_sha256']:
        raise ValueError(f'Trial measurements changed: {active}')
    validate_archive(active, checkpoint)
    with zipfile.ZipFile(active) as archive, zipfile.ZipFile(backup) as source:
        active_digest = payload_digest(archive, row['retained_members'])
        if active_digest != payload_digest(source, row['retained_members']) or active_digest != checkpoint['storage']['retained_payload_sha256']:
            raise ValueError(f'Retained array bytes changed: {active}')
    return True


def verify(workers=4):
    plan = read_plan()
    count = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(verify_one, row) for row in plan['records']]
        for future in as_completed(futures):
            future.result()
            count += 1
            if count % 2500 == 0:
                print(f'Verified {count}/{len(plan["records"])} replacements', flush=True)
    for row in plan['preserved_full'] + plan['protected_tables']:
        if sha256_file(checked(ROOT / row['path'])) != row['sha256']:
            raise ValueError(f'Protected full archive or table changed: {row["path"]}')
    report = dict(status='passed', converted_trials=count, original_archives_and_checkpoints_verified=count,
                  unchanged_trial_measurements=count, retained_payloads_verified=count,
                  full_archives_unchanged=len(plan['preserved_full']), tables_unchanged=len(plan['protected_tables']),
                  completed_utc=datetime.now(timezone.utc).isoformat(), deleted_files=0)
    atomic_json(WORK / 'validation.json', report)
    print(json.dumps(report), flush=True)


def restore():
    """Restore original pairs; preserve reduced replacements in the staging tree."""
    plan = read_plan()
    operations = []
    for row in plan['records']:
        original = checked(ROOT / row['path'], ROOT / 'envelope_method/results')
        backup = checked(STAGE / row['path'], STAGE)
        transaction = checked(backup.with_suffix('.transaction.json'), STAGE)
        reduced = checked(STAGE / 'replaced_scores' / row['path'], STAGE)
        pending = json.loads(transaction.read_text(encoding='utf-8'))
        if pending['status'] not in ('complete', 'restored'):
            raise ValueError(f'Finish staging before restore: {transaction}')
        # Validate every location before moving any files. Half-completed restore
        # locations are accepted only when each file still matches a known hash.
        for suffix, old_hash, new_hash in [
                ('.npz', row['sha256'], pending['checkpoint']['archive_sha256']),
                ('.json', row['json_sha256'], pending['active_checkpoint_sha256'])]:
            live, old, lean = (checked(original.with_suffix(suffix), ROOT / 'envelope_method/results'),
                               checked(backup.with_suffix(suffix), STAGE), checked(reduced.with_suffix(suffix), STAGE))
            if old.exists():
                if sha256_file(old) != old_hash:
                    raise ValueError(f'Original backup changed: {old}')
            elif not live.exists() or sha256_file(live) != old_hash:
                raise ValueError(f'Original evidence is missing: {old}')
            if lean.exists():
                if sha256_file(lean) != new_hash:
                    raise ValueError(f'Staged reduced replacement changed: {lean}')
            elif not live.exists() or sha256_file(live) != new_hash:
                raise ValueError(f'Reduced evidence is missing: {live}')
        operations.append((row, original, backup, reduced, transaction, pending))
    for count, (row, original, backup, reduced, transaction, pending) in enumerate(operations, 1):
        reduced.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ('.npz', '.json'):
            live, old, lean = (checked(original.with_suffix(suffix), ROOT / 'envelope_method/results'),
                               checked(backup.with_suffix(suffix), STAGE), checked(reduced.with_suffix(suffix), STAGE))
            if not lean.exists():
                live.rename(lean)
            if old.exists():
                if live.exists():
                    raise FileExistsError(f'Refusing to overwrite during restore: {live}')
                old.rename(live)
        pending['status'] = 'restored'
        atomic_json(transaction, pending)
        if count % 1000 == 0:
            print(f'Restored {count}/{len(operations)} original archive/checkpoint pairs', flush=True)
    atomic_json(WORK / 'restore_result.json', dict(status='restored', trials=len(operations), deleted_files=0))
    print(f'Restored {len(operations)} original pairs; reduced replacements retained in {STAGE / "replaced_scores"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'stage', 'verify', 'restore'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        parser.error('--workers must be between 1 and 8')
    {'prepare': prepare, 'stage': lambda: stage(args.workers), 'verify': lambda: verify(args.workers), 'restore': restore}[args.operation]()


if __name__ == '__main__':
    main()
