"""Render the FreakSeek app icon PNGs and Windows .ico from the mark geometry in logo.svg.

Run: python branding/scripts/render_icons.py (needs Pillow). Outputs land in branding/.
"""
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent
SIZES = (16, 32, 48, 64, 128, 256, 512)
# Mark geometry in the 24-unit grid shared with logo.svg and the in-app FreakSeekLogo.
F_GLYPH = [(6.5, 5), (17.5, 5), (17.5, 8), (9.5, 8), (9.5, 10.5), (14.5, 10.5), (14.5, 13.5), (9.5, 13.5), (9.5, 19), (6.5, 19)]
BOLT = [(16.6, 12), (12.8, 17.2), (15.2, 17.2), (13.8, 21), (18.4, 15.4), (16, 15.4), (17.4, 12)]
TOP, BOTTOM = (124, 58, 237), (249, 115, 22)  # violet-600 to orange-500


def render(size: int) -> Image.Image:
    scale = 8  # supersample, then downsample for smooth edges at every size
    px = size * scale
    unit = px / 24
    gradient = Image.new('RGBA', (px, px))
    draw = ImageDraw.Draw(gradient)
    for y in range(px):
        t = y / (px - 1)
        draw.line([(0, y), (px, y)], fill=tuple(round(a + (b - a) * t) for a, b in zip(TOP, BOTTOM)) + (255,))
    mask = Image.new('L', (px, px), 0)
    ImageDraw.Draw(mask).rounded_rectangle([unit * 1, unit * 1, unit * 23, unit * 23], radius=unit * 5, fill=255)
    icon = Image.new('RGBA', (px, px), (0, 0, 0, 0))
    icon.paste(gradient, (0, 0), mask)
    glyphs = ImageDraw.Draw(icon)
    for shape in (F_GLYPH, BOLT):
        glyphs.polygon([(x * unit, y * unit) for x, y in shape], fill=(255, 255, 255, 255))
    return icon.resize((size, size), Image.LANCZOS)


def main() -> None:
    images = {size: render(size) for size in SIZES}
    for size, image in images.items():
        image.save(OUT / f'icon-{size}.png')
    images[512].save(OUT / 'icon.png')
    images[256].save(OUT / 'icon.ico', sizes=[(s, s) for s in SIZES if s <= 256])


if __name__ == '__main__':
    main()
