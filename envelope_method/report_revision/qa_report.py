"""Inspect compiled PDF text, page extents and render contact sheets."""
from pathlib import Path
import json
from PIL import Image,ImageOps,ImageDraw
import pdfplumber
ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'tmp/pdfs/envelope_revision_qa'
pdf=pdfplumber.open(ROOT/'envelope_method/signed_envelope.pdf')
records=[]
for i,page in enumerate(pdf.pages):
    text=page.extract_text() or ''
    records.append(dict(page=i+1,characters=len(text),figures=len(page.images),
        first_lines=text.splitlines()[:6],last_lines=text.splitlines()[-5:]))
    bad=[]
    for b in page.chars:
        if b['x0']<20 or b['top']<15 or b['x1']>page.width-20 or b['bottom']>page.height-15:
            bad.append(str((b['x0'],b['top'],b['x1'],b['bottom'])))
    records[-1]['out_of_page_safety_margin']=bad
(DEST/'pages.json').write_text(json.dumps(records,indent=2))
(DEST/'extracted.txt').write_text('\n\n'.join(p.extract_text() or '' for p in pdf.pages),encoding='utf-8')
images=[DEST/f'page-{i+1:02d}.png' for i in range(len(pdf.pages))]
for start in range(0,len(images),6):
    sheet=Image.new('RGB',(1050,950),'#E5EAF0')
    for j,path in enumerate(images[start:start+6]):
        im=ImageOps.contain(Image.open(path).convert('RGB'),(332,435))
        x=(j%3)*350+(350-im.width)//2;y=(j//3)*475+8
        sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text(((j%3)*350+12,(j//3)*475+450),f'Page {start+j+1}',fill='#213547')
    sheet.save(DEST/f'contact_{start//6+1}.png')
log=(ROOT/'envelope_method/signed_envelope.log').read_text(errors='replace')
report=dict(pages=len(pdf.pages),overfull=log.count('Overfull'),underfull=log.count('Underfull'),
    unresolved_references='undefined references' in log,unresolved_citations='undefined citations' in log,
    page_safety_margin_violations=sum(len(r['out_of_page_safety_margin']) for r in records))
(DEST/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
for r in records:print(r['page'],r['characters'],(' | '.join(r['first_lines'][:4])).encode('ascii','replace').decode())
