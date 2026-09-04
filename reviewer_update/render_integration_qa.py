"""Render the three integrated documents and make page-layout contact sheets."""
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "multi_target_scaling_latex"
OUT = ROOT / "tmp" / "pdfs" / "integration_qa"
POPPLER = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe"
OUT.mkdir(parents=True, exist_ok=True)

for name in ("main", "response_to_reviewers", "revision_cover"):
    pdf = SOURCE / f"{name}.pdf"
    reader = PdfReader(pdf)
    subprocess.run([str(POPPLER), "-r", "100", "-png", str(pdf), str(OUT / name)], check=True)
    digits = len(str(len(reader.pages)))
    pages = [OUT / f"{name}-{i:0{digits}d}.png" for i in range(1, len(reader.pages) + 1)]
    assert all(path.exists() for path in pages)
    for start in range(0, len(pages), 12):
        sheet = Image.new("RGB", (1280, 1350), "#d9d9d9")
        draw = ImageDraw.Draw(sheet)
        for offset, path in enumerate(pages[start:start + 12]):
            picture = Image.open(path).convert("RGB")
            picture.thumbnail((310, 418))
            x, y = (offset % 4) * 320 + 5, (offset // 4) * 450 + 25
            sheet.paste(picture, (x, y))
            draw.text((x, y - 20), f"{name}: {start + offset + 1}", fill="black")
        sheet.save(OUT / f"contact-{name}-{start // 12 + 1}.jpg", quality=90)
    (OUT / f"{name}.txt").write_text("\n\n".join(f"PAGE {i+1}\n{p.extract_text()}" for i, p in enumerate(reader.pages)), encoding="utf-8")
    print(f"{name}: {len(pages)} pages rendered", flush=True)
