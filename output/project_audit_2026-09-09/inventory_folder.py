"""Read-only folder census; output goes only to this audit directory."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
rows = []
errors = []
def onerror(exc):
    errors.append(str(exc))
for directory, dirs, files in os.walk(ROOT, onerror=onerror, followlinks=False):
    path = Path(directory)
    dirs[:] = [d for d in dirs if path / d != OUT]
    for name in files:
        item = path / name
        try:
            stat = item.stat()
            relative = item.relative_to(ROOT).as_posix()
            rows.append({'path': relative, 'bytes': stat.st_size,
                         'modified_utc': datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                         'extension': item.suffix.lower() or '[none]'})
        except OSError as exc:
            errors.append(str(exc))
rows.sort(key=lambda x: x['path'])
top = defaultdict(lambda: {'files': 0, 'bytes': 0, 'extensions': Counter()})
second = defaultdict(lambda: {'files': 0, 'bytes': 0})
for row in rows:
    parts = row['path'].split('/')
    group = parts[0] if len(parts) > 1 else '[root files]'
    top[group]['files'] += 1
    top[group]['bytes'] += row['bytes']
    top[group]['extensions'][row['extension']] += 1
    key = '/'.join(parts[:2]) if len(parts) > 2 else group + '/[direct files]'
    second[key]['files'] += 1
    second[key]['bytes'] += row['bytes']
project_rows = [x for x in rows if not x['path'].startswith('.git/')]
git_status = subprocess.run(['git', 'status', '--porcelain=v1', '-uall'], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout
status_counts = Counter(line[:2] for line in git_status.splitlines()
                        if 'output/project_audit_2026-09-09/' not in line)
summary = {'captured_utc': datetime.now(timezone.utc).isoformat(), 'root': str(ROOT),
           'excluded': [str(OUT)], 'errors': errors,
           'total_files_including_git': len(rows), 'total_bytes_including_git': sum(x['bytes'] for x in rows),
           'project_files_excluding_git': len(project_rows), 'project_bytes_excluding_git': sum(x['bytes'] for x in project_rows),
           'git_status_counts_excluding_audit': dict(status_counts),
           'top_level': dict(sorted(top.items())), 'second_level': dict(sorted(second.items())),
           'largest_project_files': sorted(project_rows, key=lambda x: x['bytes'], reverse=True)[:25]}
(OUT / 'file_inventory.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
(OUT / 'inventory_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
lines = ['# Folder census', '', 'Snapshot: ' + summary['captured_utc'], '',
         'All files were enumerated, including hidden files and Git metadata. This audit directory is excluded. Sizes are logical file sizes, not disk allocation. Source content review is focused on authored research; installed dependencies and caches are categorized rather than reviewed as contributions.', '',
         '| Top-level location | Files | GiB | Dominant file types |', '|---|---:|---:|---|']
for name, item in sorted(top.items()):
    ext = ', '.join(f'{k}: {v:,}' for k, v in item['extensions'].most_common(5))
    lines.append(f"| `{name}` | {item['files']:,} | {item['bytes']/1024**3:.3f} | {ext} |")
lines += ['', '## Second-level directory detail', '', '| Location | Files | MiB |', '|---|---:|---:|']
for name, item in sorted(second.items()):
    lines.append(f"| `{name}` | {item['files']:,} | {item['bytes']/1024**2:.2f} |")
lines += ['', '## Largest files outside Git', '', '| Path | MiB |', '|---|---:|']
for row in summary['largest_project_files']:
    lines.append(f"| `{row['path']}` | {row['bytes']/1024**2:.2f} |")
lines += ['', '## Git working-tree status', '', json.dumps(dict(status_counts), indent=2), '', 'Read/stat errors: ' + str(len(errors)), '']
(OUT / 'FOLDER_INVENTORY.md').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps({k: v for k, v in summary.items() if k not in ('second_level', 'largest_project_files')}, indent=2))
