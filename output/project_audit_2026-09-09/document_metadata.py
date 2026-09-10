from pathlib import Path
import json
import re
from pypdf import PdfReader

root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent
paths = [
    'multi_target_scaling_latex/main.pdf',
    'multi_target_scaling_latex/compile_main.pdf',
    'multi_target_scaling_latex/response_to_reviewers.pdf',
    'multi_target_scaling_latex/revision_cover.pdf',
    'envelope_method/signed_envelope.pdf',
    'output/pdf/outlier_sensitivity_report/outlier_sensitivity_report.pdf',
]
records = []
for name in paths:
    path = root / name
    if not path.exists():
        records.append({'path':name,'exists':False})
        continue
    reader = PdfReader(path)
    first = reader.pages[0].extract_text() or ''
    records.append({'path': name, 'pages': len(reader.pages), 'bytes': path.stat().st_size,
                    'first_page_excerpt': first[:1300]})
abstract = (root / 'multi_target_scaling_latex/abstract.tex').read_text(encoding='utf-8')
abstract = re.sub(r'%[^\n]*', '', abstract)
abstract = re.sub(r'\\(?:begin|end)\{[^}]*\}', '', abstract)
abstract = re.sub(r'\\[a-zA-Z]+', '', abstract)
abstract = abstract.replace('{','').replace('}','')
records.append({'abstract_approx_word_count':len(abstract.split())})
(out / 'document_metadata.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
print(json.dumps(records, indent=2))
