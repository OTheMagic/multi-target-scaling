# Data organization and historical cleanup records

Current utilities:

- `check_repository_layout.py`: check the source/data boundary and Git exclusions.
- `check_latex_assets.py`: check report assets without reading research data.
- `build_indexes.py`: refresh source-side experiment indexes from central data.
- `centralize_data.py`: the September 10 relocation, with separate `primary` and `nested_data` manifests; use `verify` (and `verify --batch nested_data`) to repeat byte checks.

The earlier September 9 cleanup and archive-migration scripts are retained as dated operation source. Their output records now live under `data/docs/cleanup/`. Their original `deletable/` rollback copies were removed by the author, and their one-time stage/restore/finalization commands are not the current maintenance workflow. Do not rerun them to organize the new layout.

See [data layout](../DATA_LAYOUT.md) and [reorganization report](../DATA_REORGANIZATION.md).
