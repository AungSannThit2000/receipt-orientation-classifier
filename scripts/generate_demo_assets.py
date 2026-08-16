from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets"
CANVAS_SIZE = (1200, 900)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def create_upright_demo() -> Image.Image:
    random.seed(4201)
    width, height = CANVAS_SIZE
    background = Image.new("RGB", CANVAS_SIZE, (58, 68, 72))
    pixels = background.load()
    for y in range(height):
        for x in range(width):
            noise = random.randint(-9, 9)
            base = 62 + int(11 * y / height)
            pixels[x, y] = (base + noise, base + 6 + noise, base + 8 + noise)

    receipt_box = (390, 45, 810, 855)
    draw = ImageDraw.Draw(background)
    draw.rounded_rectangle((402, 57, 822, 867), radius=7, fill=(28, 34, 36))
    draw.rectangle(receipt_box, fill=(250, 249, 243), outline=(218, 216, 207), width=3)

    receipt = background.crop(receipt_box)
    paper = ImageDraw.Draw(receipt)
    title_font = load_font(30, bold=True)
    body_font = load_font(21)
    total_font = load_font(28, bold=True)
    center = receipt.width // 2
    title = "NORTH STAR CAFE"
    title_width = paper.textbbox((0, 0), title, font=title_font)[2]
    paper.text((center - title_width // 2, 42), title, font=title_font, fill=(25, 29, 30))
    paper.text((72, 92), "18 MARKET STREET", font=body_font, fill=(42, 45, 46))
    paper.text((72, 122), "DEMO RECEIPT", font=body_font, fill=(42, 45, 46))
    paper.line((44, 164, 376, 164), fill=(80, 83, 82), width=2)

    lines = (
        "DATE       2026-08-16",
        "ORDER      004201",
        "COFFEE         3.50",
        "SANDWICH       6.75",
        "DESSERT        4.25",
        "TAX            1.02",
    )
    y = 196
    for line in lines:
        paper.text((46, y), line, font=body_font, fill=(34, 37, 38))
        y += 48

    paper.line((44, 510, 376, 510), fill=(80, 83, 82), width=2)
    paper.text((46, 540), "TOTAL         $15.52", font=total_font, fill=(20, 23, 24))
    paper.line((44, 598, 376, 598), fill=(80, 83, 82), width=2)
    paper.text((74, 632), "THANK YOU", font=title_font, fill=(25, 29, 30))
    paper.text((65, 684), "Sample image generated", font=body_font, fill=(55, 58, 59))
    paper.text((97, 714), "for documentation", font=body_font, fill=(55, 58, 59))
    background.paste(receipt, receipt_box[:2])
    return background


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    upright = create_upright_demo()
    variants = {
        "upright": upright,
        "tilted_right": upright.rotate(-90, expand=False, fillcolor=(62, 68, 70)),
        "upside_down": upright.rotate(180, expand=False, fillcolor=(62, 68, 70)),
        "tilted_left": upright.rotate(90, expand=False, fillcolor=(62, 68, 70)),
    }
    for label, image in variants.items():
        image.save(OUTPUT_DIR / f"demo_{label}.jpg", quality=92, optimize=True)


if __name__ == "__main__":
    main()
