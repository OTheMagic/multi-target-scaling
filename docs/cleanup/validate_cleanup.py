"""Validate staged hashes, retained data metadata, source hashes and navigation."""
from pathlib import Path
import gzip
import hashlib
import json
import re
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()
plan=json.loads((OUT/'move_plan.json').read_text(encoding='utf-8'))
for record in plan['records']:
    original=Path(record['source']); staged=Path(record['destination'])
    assert not original.exists(), f'Staged source unexpectedly exists: {original}'
    assert staged.is_file() and staged.stat().st_size==record['bytes']
    assert sha(staged)==record['sha256'], f'Staged hash changed: {staged}'
with gzip.open(OUT/'protected_file_state.json.gz','rt',encoding='utf-8') as stream:
    protected=json.load(stream)
changed=[]
for relative, expected in protected.items():
    path=ROOT/relative
    if not path.is_file():
        changed.append([relative,'missing'])
        continue
    stat=path.stat()
    if [stat.st_size,stat.st_mtime_ns]!=expected:
        changed.append([relative,'size/mtime changed'])
assert not changed, changed[:30]
source_hashes=json.loads((OUT/'protected_source_hashes.json').read_text(encoding='utf-8'))
for relative, expected in source_hashes.items():
    assert sha(ROOT/relative)==expected, f'Source changed: {relative}'
with gzip.open(ROOT/'output/project_audit_2026-09-09/file_inventory.json.gz','rb') as stream:
    inventory_hash=hashlib.sha256(stream.read()).hexdigest()
inventory_record=next(r for r in plan['records'] if r['category']=='duplicate_inventory')
assert inventory_hash==inventory_record['sha256']

documents=[ROOT/'README.md',*(ROOT/'docs').glob('*.md'), ROOT/'deletable/README.md',
           ROOT/'envelope_method/results/README.md',ROOT/'envelope_method/results/absolute/INDEX.md',
           ROOT/'envelope_method/results/cqr/INDEX.md']
broken=[]; checked_links=0
for document in documents:
    for link in re.findall(r'\]\(([^)]+)\)',document.read_text(encoding='utf-8')):
        if link.startswith(('http:','https:','#')): continue
        target=link.split('#')[0].strip('<>')
        resolved=(document.parent/target).resolve()
        if not resolved.exists() and resolved != OUT/'validation.json':
            broken.append([str(document),link])
        checked_links+=1
assert not broken, broken
result=dict(status='passed', checked_utc=datetime.now(timezone.utc).isoformat(),
            staged_files=plan['total_files'], staged_bytes=plan['total_bytes'],
            staged_files_content_hash_verified=True,
            protected_research_files_metadata_unchanged=len(protected),
            protected_scientific_source_files_sha256_unchanged=len(source_hashes),
            local_navigation_links_checked=checked_links, compressed_inventory_identical=True,
            original_trial_arrays_moved=0, deleted_files=0,
            scope='Retained large archives checked for unchanged size/mtime; all scientific sources and staged files content-hashed. Not a new full-data algorithm rerun.')
(OUT/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
assert (OUT/'validation.json').exists()
print(json.dumps(result,indent=2))
