"""
make_icons.py — generate the PWA app icons.

One-off helper: draws a simple branded mark (dark rounded tile + a teal
"speech/hand" glyph) at the sizes a PWA manifest needs and writes them to
static/. Re-run only if you change the look:  python make_icons.py
"""

from PIL import Image, ImageDraw

BG = (8, 8, 15, 255)        # --bg-primary
TILE = (13, 13, 26, 255)    # --bg-secondary
ACCENT = (0, 212, 184, 255)  # --accent


def rounded(draw, box, radius, fill):
    draw.rounded_rectangle(box, radius=radius, fill=fill)


def make(size, maskable=False):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # maskable icons need their content inside a safe zone (~80%); give a full
    # bleed background and keep the glyph centered and smaller.
    pad = int(size * (0.0 if not maskable else 0.06))
    rounded(d, [pad, pad, size - 1 - pad, size - 1 - pad],
            radius=int(size * 0.22), fill=TILE)

    # inner accent tile
    m = size * (0.30 if not maskable else 0.34)
    rounded(d, [m, m, size - m, size - m], radius=int(size * 0.12), fill=ACCENT)

    # a simple "chat" notch cut from the accent tile's bottom-left corner
    notch = size * 0.12
    cx, cy = size * 0.38, size - m
    d.polygon([(cx, cy - 1), (cx + notch, cy - 1), (cx, cy + notch)], fill=TILE)

    # three "sign/speech" dots on the accent tile
    r = size * 0.045
    yy = size * 0.5
    for i, xx in enumerate((0.40, 0.5, 0.60)):
        x = size * xx
        d.ellipse([x - r, yy - r, x + r, yy + r], fill=TILE)

    return img


if __name__ == "__main__":
    for sz in (192, 512):
        make(sz, maskable=False).save(f"static/icon-{sz}.png")
        make(sz, maskable=True).save(f"static/icon-{sz}-maskable.png")
    # a small favicon too
    make(64).save("static/favicon.png")
    print("wrote static/icon-192.png, icon-512.png, *-maskable.png, favicon.png")
