"""Build and validate a small CQR10 source bundle; never start the formal study."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_BUNDLE = ROOT/'output/cqr10_cluster_source.zip'
CQR_FILES = ('runner.py', 'planning.py', 'requirements.txt', 'README.md',
             'RUNTIME_ESTIMATES.md', 'WORKLOADS.md')
SHARED_FILES = ('envelope_method/experiments.py', 'envelope_method/archive_storage.py',
                'envelope_method/settings.json')

README = '''# CQR10 cluster source bundle

Extract this ZIP into a new source directory and run commands from its root.
It contains code and frozen settings only: no results, fitted models, data
archives, timing pilots, compiler/runtime installation or cleanup history.

Use a supported Python environment (the local validation used Python 3.12):

```bash
python -m pip install -r envelope_method/cqr10/requirements.txt
python envelope_method/cqr10/runner.py --storage compact --out /scratch/YOUR_NAME/cqr10
```

The second command is a dry run: expect 22 configurations and 1,430 trials.
To run, replace the scratch path and worker count with your allocated resources:

```bash
python envelope_method/cqr10/runner.py --run --storage compact --out /scratch/YOUR_NAME/cqr10 --workers 1
```

This is the application command, not a scheduler submission script. Use your
cluster's normal allocation/submission procedure. No upload or submission has
been performed. Keep the source directory unchanged during a study, use the same
environment when resuming, and avoid overlapping jobs writing the same settings.

Compact saves every trial's metrics, coordinates, diagnostics, seeds, provenance
and checksums, plus CSV summaries. It writes no NPZ or joblib files. `--storage
scores` additionally saves scores/base widths/method bounds; `--storage full`
also saves raw observations and fitted models. Storage does not change trial
IDs, seeds, fitting or peak working memory. Richer retention after a compact run
requires a separate output directory and refitting; no silent upgrades occur.

The notebook at `envelope_method/cqr10/cqr10_experiments.ipynb` is the current
manual notebook with two optional timing-pilot preview cells guarded because
this bundle omits pilot data. Its run switch remains OFF. Numerical experiment,
summary and plotting code is unchanged. `RUNTIME_ESTIMATES.md` records the local
planning estimates; it does not establish cluster timing or parallel speedup.

`SOURCE_MANIFEST.json` records every payload member's SHA-256 and origin. All
scientific Python source is copied byte-for-byte. The notebook derivation is
identified separately with both the original and packaged hashes. The shared
experiment module includes optional non-CQR branches whose datasets/generators
are outside this CQR-only bundle; the documented CQR10 entry point is complete.
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def cluster_notebook(source):
    """Keep scientific cells intact; make unavailable pilot previews optional."""
    notebook = json.loads(source)
    changed = []
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        code = ''.join(cell['source'])
        marker = 'estimates = estimate_runtime(items, storage=STORAGE)'
        if marker in code:
            prefix, tail = code.split(marker, 1)
            original = marker + tail
            guarded = "if (DATA / 'timing_pilot/study.json').is_file():\n" + ''.join(
                '    '+line if line.strip() else line for line in original.splitlines(keepends=True))
            guarded += "\nelse:\n    estimates = None\n"
            guarded += "    display(pd.DataFrame([dict(config_id=digest(i['config'])[:16], trials=i['trials'], "
            guarded += "n_features=i['config']['n_features'], n_cal=i['config']['n_cal'], "
            guarded += "base_alpha=i['config']['base_alpha']) for i in items]))\n"
            guarded += "    print('Timing pilots omitted from source bundle; see RUNTIME_ESTIMATES.md.')\n"
            guarded += "    print('Selected trials:', sum(i['trials'] for i in items), 'Storage:', STORAGE)\n"
            guarded += "    print('Output directory:', OUTPUT.resolve())\n"
            code = prefix + guarded
            changed.append(index)
        elif "pilot = json.loads((DATA / 'timing_pilot/study.json').read_text())" in code:
            code = "if (DATA / 'timing_pilot/study.json').is_file():\n" + ''.join(
                '    '+line if line.strip() else line for line in code.splitlines(keepends=True))
            code += "\nelse:\n    print('Saved pilot preview omitted from this source-only bundle.')\n"
            changed.append(index)
        cell['source'] = code.splitlines(keepends=True)
        cell['execution_count'], cell['outputs'] = None, []
    if len(changed) != 2:
        raise ValueError('Notebook structure changed: review optional pilot cells before packaging.')
    notebook['metadata']['cluster_bundle'] = dict(
        source_notebook_sha256=sha(source), optional_pilot_cells=changed,
        note='Only optional pilot preview cells changed; formal execution remains disabled.')
    return (json.dumps(notebook, indent=1)+'\n').encode(), changed


def collect_payload(root=ROOT):
    root = Path(root)
    payload, entries = {}, []
    paths = [root/name for name in SHARED_FILES]
    paths += [root/'envelope_method/cqr10'/name for name in CQR_FILES]
    paths += sorted((root/'utility').glob('*.py'))
    for path in paths:
        relative = path.relative_to(root).as_posix()
        payload[relative] = path.read_bytes()
        entries.append(dict(path=relative, bytes=len(payload[relative]), sha256=sha(payload[relative]),
                            origin='source', source_path=relative, source_sha256=sha(payload[relative])))
    name = 'envelope_method/cqr10/cqr10_experiments.ipynb'
    original = (root/name).read_bytes()
    payload[name], cells = cluster_notebook(original)
    entries.append(dict(path=name, bytes=len(payload[name]), sha256=sha(payload[name]),
                        origin='generated notebook with optional pilot previews', source_path=name,
                        source_sha256=sha(original), changed_code_cells=cells))
    payload['README_CLUSTER.md'] = README.encode()
    entries.append(dict(path='README_CLUSTER.md', bytes=len(payload['README_CLUSTER.md']),
                        sha256=sha(payload['README_CLUSTER.md']), origin='generated cluster instructions'))
    return payload, sorted(entries, key=lambda entry: entry['path'])


SMOKE = r'''
import contextlib, io, json
from pathlib import Path
from unittest.mock import patch
from envelope_method.cqr10 import runner
import utility.exps
import envelope_method.experiments

root = Path.cwd().resolve()
for module in [runner, utility.exps, envelope_method.experiments]:
    assert Path(module.__file__).resolve().is_relative_to(root), module.__file__
items = runner.experiment_grid()
assert len(items) == 22 and sum(i['trials'] for i in items) == 1430
notebook = json.loads((runner.HERE/'cqr10_experiments.ipynb').read_text())
scope = {'__name__': '__bundle_validation__'}
with patch.object(runner, 'run_study', side_effect=AssertionError('Formal study unexpectedly started')), \
     patch.object(runner, 'fit_models', side_effect=AssertionError('Notebook unexpectedly fitted')), \
     contextlib.redirect_stdout(io.StringIO()):
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            exec(compile(''.join(cell['source']), 'bundle_notebook', 'exec'), scope)
assert scope['RUN_EXPERIMENTS'] is False and scope['STORAGE'] == 'compact'
assert not runner.DEFAULT_OUT.exists()
cfg = runner.make_config(n_cal=15)
cfg.update(n_train=50, n_validation=20, n_test=8, seed=909001)
cfg['model_params'].update(n_estimators=3, n_iter_no_change=None, min_samples_leaf=3)
out = root/'_validation_scratch'
runner.run_study([dict(config=cfg, trials=2)], out=out, storage='compact')
directory = out/runner.digest(cfg)[:16]
assert not list(out.rglob('*.npz')) and not list(out.rglob('*.joblib'))
assert (directory/'trials.csv').is_file() and (directory/'summary.csv').is_file()
for trial in range(2):
    assert runner.verified_checkpoint(directory, cfg, trial)['storage'] == 'compact'
print(json.dumps(dict(status='passed', configurations=22, formal_trials=1430,
                     tiny_smoke_trials=2, compact_no_npz_or_joblib=True,
                     notebook_training_blocked=True, formal_study_started=runner.DEFAULT_OUT.exists(),
                     implementation_hash=runner.implementation_hash(), versions=runner.versions())))
'''


def validate_bundle(bundle):
    # Extraction is into a newly allocated OS temporary directory. It is retained
    # for inspection; validation never writes into the scientific result trees.
    extracted = Path(tempfile.mkdtemp(prefix='cqr10_cluster_source_')).resolve()
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read('SOURCE_MANIFEST.json'))
        names = set(archive.namelist())
        if names != {entry['path'] for entry in manifest['files']} | {'SOURCE_MANIFEST.json'}:
            raise ValueError('Bundle member list differs from source manifest.')
        for entry in manifest['files']:
            destination = (extracted/entry['path']).resolve()
            if not destination.is_relative_to(extracted):
                raise ValueError('Unsafe bundle path.')
            data = archive.read(entry['path'])
            if len(data) != entry['bytes'] or sha(data) != entry['sha256']:
                raise ValueError(f'Bundle checksum mismatch: {entry["path"]}')
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    environment = os.environ.copy()
    # Keep installed scientific dependency paths, but prevent repo-root imports
    # from concealing a missing source member in the extracted package.
    paths = [p for p in environment.get('PYTHONPATH', '').split(os.pathsep)
             if p and Path(p).resolve() != ROOT.resolve()]
    environment['PYTHONPATH'] = os.pathsep.join(paths)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment['MPLBACKEND'] = 'Agg'
    environment['MPLCONFIGDIR'] = str(extracted/'_validation_mpl')
    command = [sys.executable, '-B', 'envelope_method/cqr10/runner.py', '--storage', 'compact',
               '--out', str(extracted/'dry_run_does_not_create_results')]
    dry = subprocess.run(command, cwd=extracted, env=environment, text=True,
                         capture_output=True, check=True)
    if '22 configurations, 1430 fresh fitted trials' not in dry.stdout:
        raise ValueError(dry.stdout)
    if (extracted/'dry_run_does_not_create_results').exists():
        raise ValueError('Dry run unexpectedly created a result directory.')
    smoke = subprocess.run([sys.executable, '-B', '-c', SMOKE], cwd=extracted,
                           env=environment, text=True, capture_output=True, check=True)
    result = json.loads(smoke.stdout.splitlines()[-1])
    result.update(extraction_directory=str(extracted), dry_run=dry.stdout.strip(),
                  checked_members=len(manifest['files']))
    return result


def build_bundle(destination=DEFAULT_BUNDLE):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload, entries = collect_payload()
    manifest = dict(schema_version=1, purpose='Source-only CQR10 compact cluster execution',
                    created_utc=datetime.now(timezone.utc).isoformat(), files=entries,
                    omitted=['experiment archives/results', 'models', 'timing pilots',
                             'runtime/compiler dependencies', 'tmp/deletable/history'])
    payload['SOURCE_MANIFEST.json'] = (json.dumps(manifest, indent=2)+'\n').encode()
    temporary = destination.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payload.items()):
            archive.writestr(name, data)
    os.replace(temporary, destination)
    validation = validate_bundle(destination)
    external = dict(manifest, archive=str(destination.resolve()), archive_bytes=destination.stat().st_size,
                    archive_sha256=sha(destination.read_bytes()), validation=validation)
    manifest_path = destination.with_name(destination.stem+'_manifest.json')
    # This source-bundle manifest is tracked with the code artifact, not data.
    manifest_path.parent.mkdir(parents=True,exist_ok=True)
    manifest_path.write_text(json.dumps(external, indent=2)+'\n', encoding='utf-8')
    return dict(archive=str(destination.resolve()), manifest=str(manifest_path.resolve()),
                bytes=destination.stat().st_size, files=len(entries), validation=validation)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    print(json.dumps(build_bundle(args.out), indent=2))


if __name__ == '__main__':
    main()
