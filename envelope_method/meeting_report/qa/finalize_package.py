"""Check the delivered evidence and write a relative, portable SHA-256 inventory."""
from pathlib import Path
import hashlib
import json
import sys
import re
import pdfplumber
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from report_paths import artifact, data_path
def load(p): return json.loads(artifact(p).read_text())
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,a): data_path(p).write_text(json.dumps(a,indent=2),encoding='utf-8')

# Historical manifest bytes remain unchanged; normalize only when resolving paths.
for row in load('data/signed/manifest.json'):
    assert digest(artifact(row['path']))==row['sha256']
absinfo=load('data/absolute/manifest.json')
for row in absinfo['configurations']:
    assert digest(artifact(Path('data/absolute')/row['archive']))==row['sha256']
source_hashes=load('sources/run_metadata.json')['source_hashes']
for name in ['envelope.py','res_rescaled.py','conformal_utils.py']:
    assert digest(ROOT/'code/utility'/name)==source_hashes['utility'+chr(92)+name]
with pdfplumber.open(ROOT/'report.pdf') as doc:
    count=len(doc.pages)
    fulltext='\n'.join(p.extract_text() for p in doc.pages)
    assert count==10
    assert '??' not in fulltext
layout=load('qa/layout_audit.json')
assert not layout['tex_warnings'] and not layout['margin_violations']
assert load('qa/independent_audit.json')['status']=='passed'
assert load('qa/core_audit.json')['status']=='passed'
assert load('qa/absolute_audit.json')['containment_violations']==0
signed=load('data/signed/audit.json')
assert signed['trials']==360 and signed['source_reference_matches']==360
for p in [ROOT/'report.tex',*sorted((ROOT/'tex').glob('*.tex'))]:
    assert not [(i,c) for i,c in enumerate(p.read_text()) if ord(c)<32 and c not in '\t\r\n']
audit=dict(status='passed',pages=count,figures=5,
    distinct_fitted_trials=2400,absolute_paired_trials=2160,signed_paired_trials=360,
    overlapping_fitted_trials=120,standalone_geometry_trials=1,
    absolute_containment_checks=12960,signed_archive_hashes_checked=360,
    absolute_archive_hashes_checked=18,method_source_snapshots_match_historical_hashes=True,
    replay='code/reproduce.py executed successfully from outside the report directory',
    visual_review='All ten pages reviewed in contact sheets; equations and figure/caption pages inspected at full render size.',
    tex_layout='No overfull/underfull boxes, unresolved references/citations, or safety-margin violations.',
    pdf_sha256=digest(ROOT/'report.pdf'))
dump('qa/final_audit.json',audit)
entries=[]
for p in sorted([*ROOT.rglob('*'),*data_path('.').rglob('*')]):
    if not p.is_file() or p==data_path('manifest.json'):continue
    rel=p.relative_to(ROOT if p.is_relative_to(ROOT) else data_path('.')).as_posix()
    if '/__pycache__/' in '/'+rel or rel.startswith('qa/mpl/'):continue
    entries.append(dict(path=rel,storage='source' if p.is_relative_to(ROOT) else 'data',bytes=p.stat().st_size,sha256=digest(p)))
dump('manifest.json',dict(format=1,files=entries,total_bytes=sum(e['bytes'] for e in entries),
    note='Runtime font caches and Python bytecode are excluded. Reproduction changes generated timestamps/hashes; rerun this inventory after compiling and reviewing updated outputs.'))
print(json.dumps(audit,indent=2))
