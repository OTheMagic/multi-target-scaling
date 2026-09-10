"""Inspect PDF text bounds and create contact sheets for visual review."""
from pathlib import Path
import json
import sys
import pdfplumber
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from report_paths import artifact, data_path
with pdfplumber.open(ROOT/'report.pdf') as doc:
    pages=[];bad=[]
    for i,p in enumerate(doc.pages,1):
        text=p.extract_text() or ''
        pages.append(dict(page=i,characters=len(text),first_line=text.splitlines()[0],last_line=text.splitlines()[-1]))
        for ch in p.chars:
            if ch['x0']<55 or ch['x1']>p.width-55 or ch['top']<35 or ch['bottom']>p.height-35:
                bad.append(dict(page=i,text=ch['text'],box=[ch['x0'],ch['top'],ch['x1'],ch['bottom']]))
log=(ROOT/'report.log').read_text()
warnings=[x for x in log.splitlines() if any(q in x for q in ['Overfull','Underfull','undefined','Missing character'])]
record=dict(page_count=len(pages),pages=pages,margin_violations=bad,tex_warnings=warnings)
(data_path('qa/layout_audit.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
images=sorted((ROOT/'qa/pages').glob('page-*.png'))[:len(pages)]
for start in range(0,len(images),6):
    sheet=Image.new('RGB',(1200,1100),'#e5e5e5');draw=ImageDraw.Draw(sheet)
    for offset,p in enumerate(images[start:start+6]):
        im=Image.open(p).convert('RGB');im.thumbnail((385,505))
        x=(offset%3)*400+(400-im.width)//2;y=(offset//3)*550+30
        sheet.paste(im,(x,y));draw.text((x,y-20),p.stem,fill='black')
    sheet.save(ROOT/f'qa/contact_{start//6+1}.png')
print(json.dumps(dict(pages=len(pages),margin_violations=len(bad),tex_warnings=warnings)))
