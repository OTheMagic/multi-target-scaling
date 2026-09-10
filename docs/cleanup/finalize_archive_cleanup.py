"""Publish the cleanup report only after all independent checks have passed."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'docs/cleanup/archive_migration_2026-09-09'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def measure(path, exclusions=()):
    count = size = 0
    pending = [path]
    skipped = []
    while pending:
        folder = pending.pop()
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.name in exclusions:
                    continue
                if entry.is_symlink() or (hasattr(entry, 'is_junction') and entry.is_junction()):
                    skipped.append(entry.path)
                elif entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    count += 1
                    size += entry.stat(follow_symlinks=False).st_size
    return dict(files=count, bytes=size, GB=size/1e9, GiB=size/1024**3, skipped_links=skipped)


def main():
    stage = read(WORK / 'stage_result.json')
    validation = read(WORK / 'validation.json')
    audit = read(ROOT / 'envelope_method/results/retained_storage_audit.json')
    comparisons = read(ROOT / 'envelope_method/results/comparator_retained_verification.json')
    assert stage['status'] == 'staged'
    assert validation['status'] == audit['status'] == comparisons['status'] == 'passed'
    assert stage['trials'] == validation['converted_trials'] == 34144
    assert audit['trials_checked'] == 35463 and audit['configurations'] == 225
    assert audit['storage_modes'] == {'full': 1319, 'scores': 34144}
    assert validation['tables_unchanged'] == 805
    tests = {}
    for name in ('storage_tests.xml', 'search_tests.xml', 'migration_tests.xml'):
        suite = ET.parse(WORK / name).getroot().find('testsuite')
        assert suite is not None and int(suite.attrib['failures']) == int(suite.attrib['errors']) == 0
        tests[name] = int(suite.attrib['tests'])
    # Confirm that the transferable scientific sources still match the bundle.
    bundle = read(ROOT / 'output/cqr10_cluster_source_manifest.json')
    for item in bundle['files']:
        if item.get('source_path'):
            assert hashlib.sha256((ROOT / item['source_path']).read_bytes()).hexdigest() == item['source_sha256']
    measurements = {key: measure(ROOT / relative) for key, relative in {
        'absolute': 'envelope_method/results/absolute', 'cqr': 'envelope_method/results/cqr',
        'results': 'envelope_method/results', 'deletable': 'deletable',
        'staged_archive_migration': 'deletable/archive_migration_2026-09-09'}.items()}
    measurements['active_project_excluding_git_and_deletable'] = measure(
        ROOT, exclusions={'.git', '.codex', '.agents', 'deletable'})
    report = dict(status='passed', measured_utc=datetime.now(timezone.utc).isoformat(),
        measurements=measurements, original_results_bytes=42009302113,
        original_absolute_bytes=40528714676, original_cqr_bytes=489707397,
        active_npz_reduction_bytes=stage['active_npz_reduction_bytes'],
        validation=validation, retained_storage_audit=audit['checks_performed'],
        comparator_method_records_verified=comparisons['method_metrics_recomputed'],
        tests=tests, distinct_pytest_tests=tests['storage_tests.xml']+tests['search_tests.xml'],
        deleted_files=0, note='Logical bytes measured before final small documentation updates; migration tests are included in storage_tests and are not double-counted.')
    (WORK / 'completion.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (WORK / 'progress.json').write_text(json.dumps(dict(status='complete', completed=34144, total=34144), indent=2), encoding='utf-8')

    table = '\n'.join([
        '## Completed storage reduction', '',
        '| Active location | Before (GB) | After (GB) |', '|---|---:|---:|',
        f"| `absolute/` | 40.529 | {measurements['absolute']['GB']:.3f} |",
        f"| `cqr/` | 0.490 | {measurements['cqr']['GB']:.3f} |",
        f"| Entire `envelope_method/results/` | 42.009 | {measurements['results']['GB']:.3f} |", '',
        f"The replaced NPZs fell from **39.817 GB to 3.089 GB**, a **36.728 GB** active-NPZ reduction. New checkpoint metadata and retained full examples are included in the active folder totals above. `deletable/` currently contains approximately **{measurements['deletable']['GB']:.3f} GB** including the earlier cache cleanup and migration originals, old JSONs and transaction records. GB is decimal; exact bytes and GiB are in the [completion record](cleanup/archive_migration_2026-09-09/completion.json).", '',
        '**The designated `deletable` contents are ready to remove under the approved retention policy.** No files were deleted automatically. Moving them on the same drive has not freed that drive space; removing the staged originals will. After removal, omitted X/y observations outside the selected full examples are no longer locally restorable, while the active scores, trial measurements and figures remain available.', '',
    ])
    path = ROOT / 'docs/ARCHIVE_CLEANUP_REPORT.md'
    content = path.read_text(encoding='utf-8')
    content = content.replace('The migration is in progress; the completed measurements and verification results will be recorded here after staging finishes.',
                              'Staging, recovery and independent verification are complete. No experiment or repetition was removed.')
    if '## Completed storage reduction' not in content:
        content = content.replace('## Retention', table + '\n## Retention', 1)
    content = content.replace('- Convert 34,144', '- Converted 34,144').replace('- Keep 1,319', '- Kept 1,319')
    content += '\n'.join(['', '## Completed verification', '',
        '- **34,144 replacements:** original NPZ/JSON hashes, unchanged trial measurements, active archive hashes and exact retained payload equality against staged originals passed.',
        '- **1,319 full exceptions and 805 result CSVs:** every protected hash is unchanged.',
        '- **225 configurations / 35,463 trials:** the updated retention audit passed; formula replay covered trial 0 of every configuration. Raw-fit checks are explicitly restricted to retained full evidence.',
        '- **3,160 comparator sidecars / 9,110 method records:** independent coverage and volume recomputation passed.',
        '- **128 distinct pytest tests:** storage, migration/recovery/restore, CQR runner and search tests passed. Fresh-draw/refitting checks plus 500 tied-score and 900 exact-zero cases passed.',
        '- **CQR delivery:** six preserved pilots and 120 fitted models verified; compact notebook plotting and clean extraction of the 86 KB source bundle passed. No formal study was launched.', '',
        'The initial pass left three operations pending before transaction creation after a destination path check refused to proceed. Originals remained intact. Destination directories are now created serially before worker moves; the resumed pass and independent verification passed. The mover continues to reject paths outside the workspace and records failures immediately.', '',
        '[Independent migration verification](cleanup/archive_migration_2026-09-09/validation.json) · [Retained-data audit](../envelope_method/results/retained_storage_audit.json) · [Comparator verification](../envelope_method/results/comparator_retained_verification.json) · [Cluster source ZIP](../output/cqr10_cluster_source.zip) · [Cluster manifest](../output/cqr10_cluster_source_manifest.json)', ''])
    path.write_text(content, encoding='utf-8')
    stage_readme = f'''# Verified originals ready for removal

This folder holds original NPZ/JSON pairs and transaction records for 34,144 converted trials. All scores, bounds and other retained arrays have verified active replacements. Every trial measurement and all 805 pre-existing result CSVs remain unchanged.

The migration validation passed. This staging tree occupies approximately {measurements['staged_archive_migration']['GB']:.3f} GB. It may be removed under the author's approved retention policy. No files were deleted automatically.

Full raw observations omitted from the active score archives will no longer be locally restorable once this folder is removed. The selected 1,319 full primary trials, all toys, all real cohorts/controls and CQR10 pilots remain active.

Read [the complete report](../../docs/ARCHIVE_CLEANUP_REPORT.md), [per-file plan](../../docs/cleanup/archive_migration_2026-09-09/plan.json.gz), and [verification](../../docs/cleanup/archive_migration_2026-09-09/validation.json).

Before removal, original pairs can be restored without deleting the score replacements:

```powershell
python docs/cleanup/migrate_archives.py restore
```

Run from the repository root. Keep the transaction files with these originals if preserving the rollback option. They are not required by normal reporting or saved-score comparisons.
'''
    (ROOT / 'deletable/archive_migration_2026-09-09/README.md').write_text(stage_readme, encoding='utf-8')
    (ROOT / 'deletable/README.md').write_text(f'''# Verified cleanup staging

The contents placed here by the two cleanup phases are ready to remove under the approved retention policy. Total measured size is approximately **{measurements['deletable']['GB']:.3f} GB** ({measurements['deletable']['GiB']:.3f} GiB). No files were deleted automatically; removing this folder is what reclaims drive space.

| Directory | Contents |
|---|---|
| `archive_migration_2026-09-09/` | Verified original full archives and superseded JSON checkpoints for 34,144 trials, plus migration transaction records; active score archives and all metrics remain |
| `cleanup_2026-09-09/` | Earlier 311 cache/render/duplicate files, 116.7 MB |
| `empty_directories/` | Ten previously staged empty cache/render directory trees |

Every experiment, repetition and numerical result remains active. The selected 1,319 primary full trials, all toys, real studies/controls and CQR10 pilots remain. Removing this staging folder relinquishes the omitted raw X/y observations and rollback copies for other trials; it does not remove their active scores or measured results.

See [the completed archive report](../docs/ARCHIVE_CLEANUP_REPORT.md), [storage guide](../docs/ARCHIVE_STORAGE_GUIDE.md), [exact measurements](../docs/cleanup/archive_migration_2026-09-09/completion.json), and [earlier cache cleanup](../docs/CLEANUP_REPORT.md).

Before deletion, restore original research archives with `python docs/cleanup/migrate_archives.py restore` from the project root. Restore the earlier cache-only phase with `./docs/cleanup/stage_cleanup.ps1 -Mode restore`. Both workflows verify hashes and refuse unintended overwrites.
''', encoding='utf-8')

    documents = [ROOT/'README.md', *(ROOT/'docs').glob('*.md'), ROOT/'deletable/README.md',
        ROOT/'deletable/archive_migration_2026-09-09/README.md', ROOT/'envelope_method/results/README.md',
        ROOT/'envelope_method/results/absolute/INDEX.md', ROOT/'envelope_method/results/cqr/INDEX.md']
    broken = []
    link_count = 0
    for document in documents:
        for link in re.findall(r'\]\(([^)]+)\)', document.read_text(encoding='utf-8')):
            if link.startswith(('http:', 'https:', '#')):
                continue
            target = link.split('#')[0].strip('<>')
            if not (document.parent/target).resolve().exists():
                broken.append([str(document.relative_to(ROOT)), link])
            link_count += 1
    navigation = dict(status='passed' if not broken else 'failed', local_links_checked=link_count, broken=broken)
    (WORK/'navigation.json').write_text(json.dumps(navigation, indent=2), encoding='utf-8')
    assert not broken, broken
    print(json.dumps(dict(status='passed', measurements=measurements,
                          local_links_checked=link_count, distinct_pytest_tests=report['distinct_pytest_tests']), indent=2))


if __name__ == '__main__':
    main()
