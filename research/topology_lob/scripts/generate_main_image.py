from __future__ import annotations

from pathlib import Path
import math

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def make_main(path: Path) -> None:
    width, height = 1200, 675
    image = Image.new("RGB", (width, height), (11, 20, 35))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (28, 28, width - 28, height - 28),
        radius=26,
        outline=(70, 100, 128),
        width=3,
        fill=(11, 20, 35),
    )

    for y, bar_width in [(125, 260), (170, 220), (215, 245), (260, 195), (305, 235)]:
        draw.rounded_rectangle(
            (80, y, 80 + bar_width, y + 22),
            radius=10,
            fill=(218, 75, 89),
        )
    for y, bar_width in [(390, 205), (425, 250), (460, 175), (495, 230), (530, 205), (565, 270)]:
        draw.rounded_rectangle(
            (80, y, 80 + bar_width, y + 22),
            radius=10,
            fill=(54, 174, 150),
        )
    draw.line((70, 360, 360, 360), fill=(135, 160, 184), width=3)

    cx, cy = 600, 300
    for radius, count in [(35, 20), (65, 28), (95, 36)]:
        points = []
        for j in range(count):
            angle = 2 * math.pi * j / count
            x = cx + radius * math.cos(angle)
            y = cy + 0.52 * radius * math.sin(angle)
            points.append((x, y))
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(44, 154, 218))
        draw.line(points + [points[0]], fill=(28, 128, 190), width=2)

    for j in range(28):
        angle = 2 * math.pi * j / 28
        x = 875 + 85 * math.cos(angle)
        y = 250 + 50 * math.sin(angle)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(68, 215, 176))

    for y, length in [(155, 90), (190, 130), (225, 65), (260, 115)]:
        draw.line((470, y, 470 + length, y), fill=(125, 153, 181), width=5)

    curve = []
    for x in range(660, 910, 4):
        t = (x - 660) / 250 * 5.5
        y = 330 - 42 * math.sin(t * 2.1) - 18 * math.sin(t * 6.5)
        curve.append((x, y))
    draw.line(curve, fill=(239, 143, 39), width=5, joint="curve")

    nodes = [(1010, 190), (955, 260), (1065, 255), (945, 360), (1055, 350), (1120, 300)]
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 4), (2, 5), (3, 4), (4, 5), (1, 4)]
    for a, b in edges:
        draw.line(
            (nodes[a][0], nodes[a][1], nodes[b][0], nodes[b][1]),
            fill=(97, 124, 151),
            width=4,
        )
    for index, (x, y) in enumerate(nodes):
        fill = (44, 112, 180) if index < 5 else (230, 132, 36)
        draw.ellipse(
            (x - 20, y - 20, x + 20, y + 20),
            fill=fill,
            outline=(150, 190, 220),
            width=2,
        )

    for y in range(515, 610, 13):
        draw.line((520, y, 1115, y), fill=(40, 60, 83), width=2)
    for x in range(520, 1120, 55):
        draw.line((x, 515, x + 80, 610), fill=(38, 59, 82), width=2)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True)


if __name__ == "__main__":
    nested = ASSETS / "main.png"
    root = ROOT.parent.parent / "main.png"
    make_main(nested)
    make_main(root)
    print(nested)
    print(root)
