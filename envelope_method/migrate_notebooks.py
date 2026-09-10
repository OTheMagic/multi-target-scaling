"""Update notebook callers structurally and archive superseded cell outputs."""
import ast
import json
from pathlib import Path
from migrate_fresh_protocol import ROOT, REVIEW, replace_function


def replace_calls(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.keyword) or node.arg != 'n_train_pool':
            continue
        value = ast.get_source_segment(source, node.value)
        text = f'n_train=int(0.8 * ({value})), n_test=({value}) - int(0.8 * ({value}))'
        edits.append((offsets[node.lineno-1]+node.col_offset,
                      offsets[node.end_lineno-1]+node.end_col_offset, text))
    for start, end, value in sorted(edits, reverse=True):
        source = source[:start] + value + source[end:]
    return source.replace('N_TRAIN_POOL', 'N_SYNTHETIC_OBSERVATIONS')


def main():
    for path in ROOT.glob('*.ipynb'):
        if path.name == 'real_exps.ipynb':
            continue
        nb = json.loads(path.read_text(encoding='utf-8'))
        archived = []
        for i, cell in enumerate(nb['cells']):
            source = ''.join(cell.get('source', []))
            if cell['cell_type'] == 'code':
                if path.name == 'exps.ipynb' and 'def run_synthetic_experiment(' in source:
                    source = replace_function(source, 'run_synthetic_experiment',
                        'from utility.exps import run_synthetic_experiment')
                if path.name == 'exps.ipynb' and 'def run_synthetic_experiment_quantile(' in source:
                    source = replace_function(source, 'run_synthetic_experiment_quantile', '''def run_synthetic_experiment_quantile(
    dim_list, sample_list, alpha_list, noise_type="Gaussian", noises_list=None,
    trials=300, method="TSCP_R", log_scale=False, quantile_model_params=None,
):
    from utility.exps import run_cqr_synthetic_experiment
    return run_cqr_synthetic_experiment(
        dim_list, sample_list, alpha_list, noise_type=noise_type,
        noises_list=noises_list, trials=trials, method=method,
        log_scale=log_scale, quantile_model_params=quantile_model_params,
        base_interval_alpha=0.8, n_train=800, n_test=200,
        return_dataclass=False,
    )''')
                source = replace_calls(source)
                if cell.get('outputs'):
                    archived.append(dict(cell=i, execution_count=cell.get('execution_count'), outputs=cell['outputs']))
                    cell['outputs'] = []
                cell['execution_count'] = None
            else:
                source = source.replace('N_TRAIN_POOL', 'N_SYNTHETIC_OBSERVATIONS')
            cell['source'] = source.splitlines(keepends=True)
        if archived:
            dest = REVIEW / 'notebook_outputs' / (path.stem + '.json')
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                raise FileExistsError(dest)
            dest.write_text(json.dumps(dict(source=path.name,
                reason='Superseded synthetic or mixed notebook execution output; sampling provenance not guaranteed.',
                cells=archived), indent=2), encoding='utf-8')
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
        print(path.name, 'updated;', len(archived), 'outputs archived')


if __name__ == '__main__':
    main()
