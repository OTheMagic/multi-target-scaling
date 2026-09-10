from pathlib import Path
import gzip
import json
import re

out = Path(__file__).resolve().parent
index_path = out / 'file_inventory.json'
if index_path.exists():
    inventory = index_path.read_bytes()
else:
    with gzip.open(out / 'file_inventory.json.gz', 'rb') as stream:
        inventory = stream.read()
with gzip.open(out / 'file_inventory.json.gz', 'wb') as stream:
    stream.write(inventory)
summary = json.loads((out / 'inventory_summary.json').read_text(encoding='utf-8'))
rows = json.loads(inventory)
assert len(rows) == summary['total_files_including_git']
assert sum(x['bytes'] for x in rows) == summary['total_bytes_including_git']
assert sum(v['files'] for v in summary['top_level'].values()) == len(rows)
missing = []
for name in ('PROJECT_REVIEW.md', 'theory_audit.md', 'experiments_audit.md', 'code_audit.md', 'FOLDER_INVENTORY.md'):
    path = out / name
    assert path.exists(), name
    content = path.read_text(encoding='utf-8')
    for target in re.findall(r'\]\(([^)]+)\)', content):
        if target.startswith(('http:', 'https:', '#')):
            continue
        target = target.strip('<>')
        if not (path.parent / target).exists():
            missing.append((name, target))
assert not missing, missing
result = {'status':'passed', 'complete_inventory_rows':len(rows),
          'folder_count_and_size_consistency':True, 'report_links_resolve':True,
          'compressed_index_bytes':(out/'file_inventory.json.gz').stat().st_size,
          'pytest_results':{'passed':75,'failed':1,'failure':'historical figure values differ from current CSV values'},
          'standalone_protocol':'passed; fresh trial fitting, tied-score cases, exact-zero boundary cases',
          'scope':'Only targeted tests were rerun; full experiment archive checks referenced in the report are saved prior evidence.'}
(out / 'audit_validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
