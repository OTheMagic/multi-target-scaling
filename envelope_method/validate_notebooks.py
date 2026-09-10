"""Compile notebook cells and check migrated keyword dictionaries without running studies."""
import ast
import inspect
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'envelope_method')]
from utility import exps


def main():
    cells, special = 0, []
    paths = list(ROOT.glob('*.ipynb')) + [ROOT/'envelope_method/full_lwc_manual.ipynb']
    for path in paths:
        notebook = json.loads(path.read_text(encoding='utf-8'))
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            try:
                compile(source, f'{path.name}:cell_{index}', 'exec')
            except SyntaxError:
                from IPython.core.inputtransformer2 import TransformerManager
                compile(TransformerManager().transform_cell(source), f'{path.name}:cell_{index}', 'exec')
                special.append(dict(notebook=path.name, cell=index))
            cells += 1
            if path.name != 'real_exps.ipynb':
                assert 'n_train_pool' not in source
                assert 'redraw_train_test=False' not in source

    # **kwargs dictionaries can retain obsolete API keys even when direct calls
    # have been migrated. Bind the real reviewer helper to the current signature.
    notebook = json.loads((ROOT/'reviewer_cqr_experiments.ipynb').read_text(encoding='utf-8'))
    namespace = dict(DIM=3, ALPHA=.1, TRIALS=2, N_SYNTHETIC_OBSERVATIONS=3000,
                     N_FEATURES=5, N_INFORMATIVE=5, QUANTILE_MODEL_PARAMS={})
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        tree = ast.parse(''.join(cell['source']))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == 'cqr_runner_kwargs':
                exec(compile(ast.Module(body=[node], type_ignores=[]), 'cqr_runner_kwargs', 'exec'), namespace)
    kwargs = namespace['cqr_runner_kwargs']()
    inspect.signature(exps.run_cqr_synthetic_experiment).bind_partial(**kwargs)
    assert kwargs['n_train'] == 2400 and kwargs['n_test'] == 600

    manual = json.loads((ROOT/'envelope_method/full_lwc_manual.ipynb').read_text(encoding='utf-8'))
    namespace = {'display': lambda value: None}
    with patch('run_full_lwc_scaling.run', side_effect=AssertionError('Full LWC must not run')) as runner:
        for cell in manual['cells']:
            if cell['cell_type'] == 'code':
                exec(compile(''.join(cell['source']), 'full_lwc_manual', 'exec'), namespace)
        runner.assert_not_called()
        assert namespace['RUN_FULL_LWC'] is False
    report = dict(status='passed', notebooks=len(paths), compiled_cells=cells,
                  ipython_cells=special, reviewer_kwargs='bound to current API',
                  full_lwc_manual='all cells executed with run switch false; runner not called')
    (ROOT/'data/envelope_method/notebook_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
