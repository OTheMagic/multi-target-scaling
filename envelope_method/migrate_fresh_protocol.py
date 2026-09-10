"""One-time, checked migration of synthetic APIs and obsolete result archives.

This maintenance script is not an experimental protocol. All moved files retain
their relative paths and SHA-256 hashes in the review folder's manifest.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'quarantine/obsolete_synthetic_2026-09-08'


def replace_function(source, name, replacement):
    lines = source.splitlines(keepends=True)
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ''.join(lines[:node.lineno-1]) + replacement.rstrip() + '\n' + ''.join(lines[node.end_lineno:])


def change_api():
    path = ROOT / 'utility/exps.py'
    source = path.read_text(encoding='utf-8')
    for name, nt, nv in [('run_abs_res_synthetic_experiment', 7200, 800),
                         ('run_cqr_synthetic_experiment', 2400, 600)]:
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
        block = ast.get_source_segment(source, node)
        if 'n_train_pool:' not in block:
            continue
        block = block.replace('    n_train_pool: int = 8000,\n', '')
        block = block.replace('    test_size: float = 0.2,\n', '')
        block = block.replace('    redraw_train_test: bool = False,\n', '')
        block = block.replace('n_train: Optional[int] = None', f'n_train: int = {nt}')
        block = block.replace('n_test: Optional[int] = None', f'n_test: int = {nv}')
        if '    n_train_pool : int' in block:
            start = block.index('    n_train_pool : int')
            end = block.index('    oracle_n_samples : int', start)
            block = block[:start] + f'''    n_features : int, default=10
        Number of predictor features.
    n_informative : int, default=10
        Number of informative predictor features.
    n_train : int, default={nt}
        Fresh training observations generated and fitted in every trial.
    n_test : int, default={nv}
        Fresh independent test observations generated in every trial.
''' + block[end:]
        start = block.index('    if redraw_train_test:\n')
        end = block.index('\n    ', block.index('raise ValueError("n_train and n_test must be positive.")', start))
        block = block[:start] + '''    n_train, n_test = int(n_train), int(n_test)
    if n_train <= 0 or n_test <= 0:
        raise ValueError("n_train and n_test must be positive.")''' + block[end:]
        start = block.index('    protocol_n_test = (')
        end = block.index('    metadata.update(', start)
        block = block[:start] + block[end:]
        block = block.replace('"redraw_train_test": redraw_train_test', '"redraw_train_test": True')
        block = block.replace('protocol_n_train', 'n_train').replace('protocol_n_test', 'n_test')
        block = block.replace('        X, y, coef_true = data_generator(',
            '        # Retain only the DGP coefficients; no observations are reused.\n        _, _, coef_true = data_generator(')
        block = block.replace('n_samples=n_train_pool', 'n_samples=n_train + n_test')
        start = block.index('                if redraw_train_test:\n')
        otherwise = block.index('                else:\n', start)
        end = block.index('                    calibration_seed = seed', otherwise) + len('                    calibration_seed = seed')
        fresh = block[start + len('                if redraw_train_test:\n'):otherwise]
        fresh = '\n'.join(line[4:] if line.startswith('    ') else line for line in fresh.split('\n'))
        block = block[:start] + fresh.rstrip() + block[end:]
        block = block.replace('''                            if redraw_train_test
                            else seed + oracle_seed_offset
''', '')
        block = block.replace('        "n_train_pool": n_train_pool,\n', '')
        block = block.replace('        "test_size": test_size,\n', '')
        block = block.replace('Since TSCP-style methods require nonnegative scores, this runner supports:',
            'For comparing the nonnegative shortcut with signed baselines, this runner supports:')
        source = replace_function(source, name, block)
    ast.parse(source)
    path.write_text(source, encoding='utf-8')
    for path in (ROOT / 'reviewer_update').rglob('build_experiment_update.py'):
        lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
        lines = [line for line in lines if not any(v in line for v in
                 ['"n_train_pool":', 'n_train_pool=', 'redraw_train_test=True,'])]
        # Metadata elsewhere retains the truthful, non-selectable True column.
        for i, line in enumerate(lines):
            if '"redraw_train_test": True,' in line and len(line)-len(line.lstrip()) == 8:
                lines[i] = ''
        new = ''.join(lines)
        ast.parse(new)
        path.write_text(new, encoding='utf-8')


def freeze_settings():
    frozen = ROOT / 'envelope_method/settings.json'
    if frozen.exists():
        return
    prior = json.loads((ROOT / 'envelope_method/results/simulation_inventory.json').read_text())
    registry = {}
    for old in prior:
        cfg = dict(old['config'], redraw=True)
        ident = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
        item = registry.setdefault(ident, dict(id=ident, config=cfg, trials=0, sources=[], transforms=[]))
        item['trials'] = max(item['trials'], old['trials'])
        for key in ['sources', 'transforms']:
            for value in old[key]:
                if value not in item[key]:
                    item[key].append(value)
    frozen.write_text(json.dumps(list(registry.values()), indent=2), encoding='utf-8')


def change_runner():
    path = ROOT / 'envelope_method/experiments.py'
    source = path.read_text(encoding='utf-8')
    source = replace_function(source, 'specs', '''def specs():
    """Frozen scenario design; execution always redraws and refits every trial."""
    items = json.loads((ROOT / 'envelope_method/settings.json').read_text())
    if not all(i['config'].get('redraw') is True for i in items):
        raise ValueError('Only fresh-per-trial configurations are supported.')
    return items
''')
    source = source.replace('from sklearn.model_selection import train_test_split\n', '')
    start = source.find("            if cfg['redraw']:\n")
    if start >= 0:
        end = source.index('            xc, yc = gen(', start)
        otherwise = source.index('            else:\n', start)
        fresh = source[start + len("            if cfg['redraw']:\n"):otherwise]
        fresh = '\n'.join(line[4:] if line.startswith('    ') else line for line in fresh.split('\n'))
        source = source[:start] + fresh.rstrip() + '\n' + source[end:]
    source = source.replace('    X, y, coef = gen(', '    _, _, coef = gen(')
    ast.parse(source)
    path.write_text(source, encoding='utf-8')


def move_data():
    REVIEW.mkdir(parents=True, exist_ok=True)
    manifest_path = REVIEW / 'manifest.json'
    if manifest_path.exists():
        raise RuntimeError('Migration already has a manifest; review it before rerunning.')
    targets = {}
    def add(path, reason):
        if path.is_dir():
            for child in path.rglob('*'):
                if child.is_file():
                    targets[child.resolve()] = reason
        elif path.is_file():
            targets[path.resolve()] = reason
    for folder in ['syn_exps', 'reviewer_exps']:
        add(ROOT / folder, 'Known or undocumented fixed-pool synthetic results and derived figures.')
    out = ROOT / 'envelope_method/results'
    for kind in ['absolute', 'cqr']:
        for path in (out / kind).glob('*/config.json'):
            if not json.loads(path.read_text())['config']['redraw']:
                add(path.parent, 'Preliminary envelope run used a reused synthetic train/test pool.')
    add(out / 'legacy_auxiliary', 'Auxiliary run used a reused synthetic train/test pool.')
    for path in (out / 'toys').iterdir():
        if path.name != 'boundary':
            add(path, 'Superseded residual-only empirical toy: no fitted training model.')
    for path in out.iterdir():
        if path.is_file():
            add(path, 'Derived aggregate mixes obsolete synthetic results; will regenerate.')
    add(ROOT / 'envelope_method/comparison_audit', 'Mathematical witnesses from superseded experiments; retained for review.')
    # Find duplicates of obsolete data anywhere outside caches and real cohorts.
    hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in targets if p.suffix in ['.csv', '.npz']}
    for folder in ['multi_target_scaling_latex', 'reviewer_update']:
        for path in (ROOT / folder).rglob('*.csv'):
            if hashlib.sha256(path.read_bytes()).hexdigest() in hashes:
                add(path, 'Exact SHA-256 duplicate of obsolete synthetic data.')
    records = []
    for src, reason in sorted(targets.items()):
        src = src.resolve(strict=True)
        dst = (REVIEW / src.relative_to(ROOT)).resolve()
        if not src.is_relative_to(ROOT) or src.is_relative_to(REVIEW) or not dst.is_relative_to(REVIEW):
            raise ValueError(f'Unsafe archive path: {src} -> {dst}')
        if dst.exists():
            raise FileExistsError(dst)
        records.append(dict(source=str(src.relative_to(ROOT)), destination=str(dst.relative_to(REVIEW)),
                            bytes=src.stat().st_size, sha256=hashlib.sha256(src.read_bytes()).hexdigest(), reason=reason))
    manifest_path.write_text(json.dumps(dict(status='planned', files=records), indent=2), encoding='utf-8')
    for record in records:
        src, dst = ROOT / record['source'], REVIEW / record['destination']
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.rename(src, dst)
    manifest_path.write_text(json.dumps(dict(status='complete', files=records), indent=2), encoding='utf-8')
    print('Archived files:', len(records), 'bytes:', sum(r['bytes'] for r in records), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=['code', 'data'])
    args = p.parse_args()
    freeze_settings()
    if args.step == 'code':
        change_api()
        change_runner()
    else:
        move_data()
