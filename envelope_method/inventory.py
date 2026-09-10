"""Extract runnable experiment cells and saved-study sizes without notebook execution."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent


def main():
    notebooks = []
    for path in ROOT.glob('*.ipynb'):
        if path.name.startswith('smoke'):
            continue
        doc = json.loads(path.read_text(encoding='utf-8'))
        cells = []
        for i, cell in enumerate(doc['cells']):
            if cell.get('cell_type') != 'code':
                continue
            source = ''.join(cell['source'])
            if any(s in source for s in ('run_', 'experiment', 'sample_list', 'num_splits', 'trials', 'heavy_t')):
                cells.append({'cell': i, 'source': source})
        notebooks.append({'path': str(path.relative_to(ROOT)), 'cells': cells})
    tables = []
    for folder in ['syn_exps', 'real_exps', 'reviewer_exps', 'reviewer_update/data']:
        for path in (ROOT / 'data' / folder).rglob('*.csv'):
            if '_smoke' in str(path):
                continue
            frame = pd.read_csv(path)
            record = {'path': str(path.relative_to(ROOT)), 'rows': len(frame), 'columns': list(frame.columns)}
            for key in ['trial', 'method', 'n_trials', 'n_cals', 'n_cal', 'n_dim', 'alpha', 'base_alpha', 'noise_type', 'dataset']:
                if key in frame:
                    values = frame[key].drop_duplicates().tolist()
                    record[key] = values if len(values) < 60 else {'count': len(values), 'min': min(values), 'max': max(values)}
            tables.append(record)
    output = {'notebooks': notebooks, 'tables': tables}
    (ROOT / 'data/envelope_method/inventory.json').write_text(json.dumps(output, indent=2, default=str), encoding='utf-8')
    for nb in notebooks:
        print('\nNOTEBOOK', nb['path'])
        for cell in nb['cells']:
            print('CELL', cell['cell'], cell['source'][:18000])
    print('\nTABLES', len(tables))


if __name__ == '__main__':
    main()
