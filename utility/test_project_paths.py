from pathlib import Path

import pytest

from utility.project_paths import DATA_ROOT, data_path, resolve_artifact


def test_data_destination_is_portable_and_not_double_prefixed():
    assert data_path('envelope_method/results', 'trials.csv') == DATA_ROOT / 'envelope_method/results/trials.csv'
    assert data_path(r'data\real_exps\data\rf2.arff') == DATA_ROOT / 'real_exps/data/rf2.arff'
    for invalid in ('../outside', '/outside', 'E:/outside'):
        with pytest.raises(ValueError):
            data_path(invalid)


def test_historical_reference_resolves_on_another_checkout(tmp_path):
    relative = Path('envelope_method/results/demo/trial_000.json')
    relocated = tmp_path / 'data' / relative
    relocated.parent.mkdir(parents=True)
    relocated.write_text('unchanged checkpoint')
    references = (str(relative), relative.as_posix(),
                  'E:/multi-target-scaling/' + relative.as_posix(),
                  ('E:/multi-target-scaling/' + relative.as_posix()).replace('/', '\\'))
    for reference in references:
        assert resolve_artifact(reference, root=tmp_path) == relocated
    assert resolve_artifact(relocated, root=tmp_path) == relocated
    # Compact trials intentionally have the sidecar but no NPZ payload.
    assert resolve_artifact(relative.with_suffix('.npz'), root=tmp_path) == relocated.with_suffix('.npz')


def test_original_source_and_external_fixtures_stay_addressable(tmp_path):
    source = tmp_path / 'source.py'
    source.write_text('# source')
    assert resolve_artifact('source.py', root=tmp_path) == source
    external = tmp_path.parent / 'external.npz'
    assert resolve_artifact(external, root=tmp_path) == external
