# Data reorganization verification

Completed September 10, 2026. See [data layout and figure-source map](DATA_LAYOUT.md) for the resulting structure.

**217,802 numerical files (10,137,937,601 bytes; 10.138 GB) moved under root `data/`. Every moved file passed SHA-256 verification against its original contents.** No scientific data files were deleted, compacted further or regenerated.

## What changed

- Raw datasets, CSV tables, residual/model archives and generated numerical audit records are centralized under `data/`, preserving their original relative paths.
- Current experiment, plotting, notebook, real-data and report readers/writers use the central paths. Saved provenance strings remain unchanged and resolve through the relocation helper.
- LaTeX figures and generated `.tex` table inputs remain source-side. `report_revision/report_main.tex` can now compile from its own folder with local `figures/` and table inputs.
- Each active report folder has `DATA_SOURCES.md` identifying its corresponding central evidence.
- The CQR10 cluster bundle was refreshed and tested; it contains source only. No formal experiment sweep was launched.
- Empty former data directories were removed. Existing scientific files and report figure content were preserved.

## Verification

| Check | Result |
|---|---|
| Original-to-destination file hashes | All 217,802 match |
| Main full/scores archive index | 225 configurations, 35,463 trials; 1,319 full +34,144 scores |
| Current Python/notebook syntax | 123 Python files and checked notebook cells parse |
| Storage, CQR10, relocation and reviewer tests | 56 passed; one pre-existing figure-provenance mismatch retained |
| Reviewer CSV readers | 59 files /492,859 rows load from central data |
| Historical reviewer numerical mirrors | All 134 compared files unchanged |
| Real/CQR saved sources and models | Legacy links resolve; six full pilot checkpoints verify |
| Meeting report independent replay | Both audits pass, including 12,960 saved containment comparisons |
| LaTeX without numerical data | Six isolated offline builds pass |
| Research data outside root data (excluding ignored runtime scratch) | None found |
| Git-visible working files over 100 MB | None |

The first pass excluded nested folders named `data` and omitted `.joblib` model files. Live reader checks caught both; the supplemental manifest includes all 5,630 affected files. Both batches were moved and independently hash-verified before completion.

The remaining reviewer test failure concerns old saved figure values versus already-updated CSVs (21 source hashes and 448 plotted cells). It was present before relocation. No figures or scientific measurements were regenerated to conceal it; figure reconciliation remains part of the planned manuscript work.

## GitHub preparation

`/data/` and `/tmp/` are ignored. Previously tracked numerical files and local runtime/build artifacts were removed from the Git index while retaining the local files. Those exclusions are staged; source/documentation changes are available for your normal review and commit.
The prospective source/document/figure working files total about **26.7 MB** (excluding data, scratch and `.git` history). The inspected existing Git history has no individual blobs at or above 100 MB. History itself is unchanged and may still contain older data/runtime files; no commit, push or history rewrite was performed.

A source-only clone can compile the retained reports. Regenerating existing plots or auditing saved trials requires a separately supplied copy of `data/`. The old `deletable/` folder was removed by the author before this task; those prior full-data rollback copies are no longer local.

## Local audit records

- [Primary relocation manifest](../data/_organization/2026-09-10/manifest.json.gz)
- [Supplemental relocation manifest](../data/_organization/2026-09-10/nested_data/manifest.json.gz)
- [Completion and Git boundary checks](../data/_organization/2026-09-10/completion.json)
- [Experiment consumers and independent audits](../data/_organization/2026-09-10/envelope_consumer_validation.json)
- [Reviewer data checks](../data/_organization/2026-09-10/reviewer_data_location_checks.json)
- [Six source-only compilation results](../data/_organization/2026-09-10/latex_compilation.json)

These machine-readable records are local data and therefore excluded from Git. This Markdown report remains in the shareable source tree.

Repeat the quick source/data check with `python docs/cleanup/check_repository_layout.py`; check retained LaTeX inputs with `python docs/cleanup/check_latex_assets.py`.
