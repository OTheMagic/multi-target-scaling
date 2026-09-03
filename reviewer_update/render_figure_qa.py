"""Render reviewer figure PDFs and assemble contact sheets for visual QA."""

from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "figures"
OUT_DIR = ROOT / "qa_renders"


def render_pdf(path: Path) -> Image.Image:
    document = pymupdf.open(path)
    pixmap = document[0].get_pixmap(dpi=160, alpha=False)
    output = OUT_DIR / f"{path.stem}.png"
    pixmap.save(output)
    return Image.open(output).convert("RGB")


def make_sheet(
    names: list[str],
    rendered: dict[str, Image.Image],
    filename: str,
    columns: int,
) -> None:
    tile_width, tile_height = 1050, 720
    rows = (len(names) + columns - 1) // columns
    canvas = Image.new(
        "RGB", (columns * tile_width, rows * tile_height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for index, name in enumerate(names):
        image = rendered[name].copy()
        image.thumbnail((tile_width - 30, tile_height - 55))
        column = index % columns
        row = index // columns
        x = column * tile_width + (tile_width - image.width) // 2
        y = row * tile_height + 30
        canvas.paste(image, (x, y))
        draw.text(
            (column * tile_width + 12, row * tile_height + 8),
            name,
            fill="black",
        )
    canvas.save(OUT_DIR / filename)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(FIGURE_DIR.glob("*.pdf"))
    rendered = {path.stem: render_pdf(path) for path in paths}
    body = [path.stem for path in paths if path.stem.startswith("fig_body_")]
    appendix = [path.stem for path in paths if path.stem.startswith("fig_app_")]
    make_sheet(body, rendered, "contact_body.png", columns=2)
    make_sheet(appendix, rendered, "contact_appendix.png", columns=3)
    print(f"Rendered {len(paths)} PDFs")


if __name__ == "__main__":
    main()
