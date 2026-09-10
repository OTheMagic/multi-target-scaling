"""Package the verified TeX/PDF and compact evidence; keep raw trial data in place."""
from pathlib import Path
import hashlib,json,shutil,zipfile
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
DATA=ROOT/'data/envelope_method/report_revision'
EVIDENCE=ROOT/'data/output/pdf/envelope_shortcut_report/evidence'
DEST=ROOT/'output/pdf/envelope_shortcut_report'
source=ROOT/'envelope_method/signed_envelope.pdf'
portable=DEST/'signed_envelope.pdf'
assert len(PdfReader(source).pages)==20
assert len(PdfReader(portable).pages)==20
assert [p.extract_text() for p in PdfReader(source).pages]==[p.extract_text() for p in PdfReader(portable).pages]
meta=json.loads((DATA/'run_metadata.json').read_text())
for name in ['utility\\envelope.py','utility\\res_rescaled.py','utility\\conformal_utils.py']:
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==meta['source_hashes'][name]
qa=json.loads((ROOT/'tmp/pdfs/envelope_revision_qa/report.json').read_text())
assert qa['pages']==20 and not any(qa[k] for k in ['overfull','underfull','unresolved_references','unresolved_citations','page_safety_margin_violations'])
qa['visual_review']='All 20 pages reviewed in contact sheets; enlarged figure, table and proof pages checked after final changes.'
qa['portable_copy']='Compiled independently; all extracted page text matches the main report.'
qa['core_implementations']='SHA-256 unchanged from the start of the fresh experiment run.'
qa['artifact_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [source,ROOT/'envelope_method/signed_envelope.tex',HERE/'experiment_suite.py',HERE/'plot_report.py',HERE/'build_report.py']}
(DATA/'final_qa.json').write_text(json.dumps(qa,indent=2))
EVIDENCE.mkdir(parents=True,exist_ok=True)
for name in ['design.json','run_metadata.json','summary.csv','table_n80.csv','signed_comparison.csv',
             'runtime.csv','runtime_summary.csv','evidence.json','verification.json','final_qa.json',
             'geometry_design.json','geometry_audit.json','signed_geometry.json']:
    shutil.copy2(DATA/name,EVIDENCE/name)
text='''# Surface envelopes versus the separated shortcut

Open `signed_envelope.pdf` to read the 20-page research report. The TeX file,
nine vector PDF figures and two generated table inputs are self-contained.

Compile **from this folder** with `tectonic signed_envelope.tex`, or run
`pdflatex signed_envelope.tex` twice. Standard LaTeX packages and Latin Modern
fonts are needed. The included reading copy was compiled with Tectonic 0.17.0.

The report includes 2,400 fresh fitted trials with absolute, capped quantile,
and raw signed quantile residuals. It reports uniform weak size containment
for identical nonnegative scores and explicitly discusses signed-score and
runtime exceptions. Coverage, volume, coordinates, fixed-input geometry,
target sensitivity, infinity rates and construction timing are visualized.

`evidence/` contains compact design, summary, timing and verification records.
The full experiment code, per-trial metrics and 2,400 full observation archives
remain at `E:/multi-target-scaling/envelope_method/report_revision/`.
See that directory's README for complete reproduction commands and provenance.
The large raw trial archives are not duplicated in this portable reading package.
'''
(DEST/'README.md').write_text(text)
zip_path=ROOT/'output/pdf/envelope_shortcut_report.zip'
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(DEST.rglob('*')):
        if p.is_file() and p.suffix not in ['.log','.png']:
            z.write(p,p.relative_to(DEST.parent))
with zipfile.ZipFile(zip_path) as z:assert z.testzip() is None
print(json.dumps(dict(pdf=str(portable),source=str(DEST/'signed_envelope.tex'),pages=20,
    archive=str(zip_path),archive_bytes=zip_path.stat().st_size,qa=qa),indent=2))
