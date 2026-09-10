"""Render the compiled mathematical note and audit text/layout metadata."""
import json
from pathlib import Path

import pypdfium2 as pdfium
from pypdf import PdfReader
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parent


def main():
    source = ROOT / 'signed_envelope.pdf'
    target = ROOT / 'qa'
    target.mkdir(exist_ok=True)
    doc = pdfium.PdfDocument(str(source))
    thumbs = []
    for index, page in enumerate(doc):
        im = page.render(scale=1.5).to_pil().convert('RGB')
        im.save(target/f'page_{index+1:02d}.png')
        thumb = ImageOps.contain(im,(306,420))
        tile = Image.new('RGB',(326,450),'#dddddd')
        tile.paste(thumb,((326-thumb.width)//2,8))
        ImageDraw.Draw(tile).text((10,430),f'Page {index+1}',fill='black')
        thumbs.append(tile)
    sheet=Image.new('RGB',(326*4,450*((len(thumbs)+3)//4)),'white')
    for i,tile in enumerate(thumbs):
        sheet.paste(tile,((i%4)*326,(i//4)*450))
    sheet.save(target/'contact_sheet.png')
    reader=PdfReader(source)
    text='\n'.join(page.extract_text() for page in reader.pages)
    (target/'extracted.txt').write_text(text,encoding='utf-8')
    log=(ROOT/'signed_envelope.log').read_text(errors='replace')
    report=dict(pages=len(reader.pages),unresolved_references='undefined references' in log,
                unresolved_citations='Citation ' in log and 'undefined' in log,
                overfull_boxes=log.count('Overfull'),underfull_boxes=log.count('Underfull'),
                text_characters=len(text))
    data_target = ROOT.parent / 'data/envelope_method/qa'
    data_target.mkdir(parents=True,exist_ok=True)
    (data_target/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
