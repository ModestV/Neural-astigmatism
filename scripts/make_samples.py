"""Генерирует тестовые чертежи с заранее известной разметкой (ground truth).

    backend/.venv/bin/python scripts/make_samples.py     # из корня проекта

В samples/ появятся:
  plan.svg           — лист 1: план этажа с обозначениями электрооборудования;
  plan-2pages.pdf    — лист 1 и лист 2 с другой расстановкой;
  plan.dxf           — лист 1 в DXF, обозначения оформлены блоками (BLOCK + INSERT);
  ground_truth.json  — где и под каким углом стоит каждое обозначение на каждом листе.

Геометрия обозначений «сплющена» (без <g transform>), чтобы алгоритм не мог
подсмотреть ответ в структуре файла. Двери нарисованы дугами — это намеренные
помехи, похожие на розетку.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import ezdxf
import pymupdf

OUT = Path(__file__).resolve().parent.parent / "samples"
PAGE_W, PAGE_H = 420.0, 297.0  # A3 альбомный, мм
FRAME = (20.0, 5.0, 415.0, 292.0)  # рамка листа: x0, y0, x1, y1
MM_TO_PT = 72 / 25.4
# Бэкенд рендерит DXF в масштабе 1:1 с полями 5 мм вокруг содержимого (app/converters.py),
# поэтому левый верхний угол рамки оказывается в точке (5 мм, 5 мм) страницы.
DXF_MARGIN = 5.0

# Примитивы в локальных координатах обозначения (мм, ось Y вниз, (0, 0) — точка вставки):
#   ("line", x1, y1, x2, y2) · ("circle", cx, cy, r) · ("arc", cx, cy, r, a0, a1),
# где дуга идёт по часовой стрелке от угла a0 до a1 (градусы).
SYMBOLS: dict[str, tuple[str, list[tuple]]] = {
    "socket": ("Розетка", [
        ("arc", 0, 0, 3, 180, 360), ("line", -3, 0, 3, 0), ("line", 0, -3, 0, -6),
    ]),
    "switch": ("Выключатель", [
        ("circle", 0, 0, 1.5), ("line", 1.06, -1.06, 4.6, -4.6), ("line", 4.6, -4.6, 6.0, -3.2),
    ]),
    "lamp": ("Светильник", [
        ("circle", 0, 0, 3.5), ("line", -2.47, -2.47, 2.47, 2.47), ("line", -2.47, 2.47, 2.47, -2.47),
    ]),
    "box": ("Распаячная коробка", [
        ("line", -2, -2, 2, -2), ("line", 2, -2, 2, 2), ("line", 2, 2, -2, 2),
        ("line", -2, 2, -2, -2), ("line", -2, 2, 2, -2),
    ]),
}


@dataclass
class Instance:
    symbol: str
    x: float
    y: float
    angle: float = 0.0  # по часовой стрелке, градусы
    legend: bool = False  # экземпляр в таблице условных обозначений


@dataclass
class Sheet:
    number: int
    title: str
    walls: list[tuple[float, float, float, float]]
    doors: list[tuple[float, float, float, float]]  # (x, y, радиус, угол начала дуги)
    instances: list[Instance]


def legend() -> list[Instance]:
    return [Instance(key, 330, 42 + 15 * i, legend=True) for i, key in enumerate(SYMBOLS)]


SHEETS = [
    Sheet(
        number=1,
        title="План расположения электрооборудования",
        walls=[(40, 20, 300, 20), (300, 20, 300, 220), (300, 220, 40, 220), (40, 220, 40, 20),
               (150, 20, 150, 220), (40, 120, 150, 120)],
        doors=[(150, 60, 9, 90), (95, 120, 9, 180), (300, 150, 9, 180)],
        instances=legend() + [
            # помещение 1
            Instance("lamp", 95, 70), Instance("socket", 40, 60, 90), Instance("socket", 40, 95, 90),
            Instance("socket", 100, 20, 180), Instance("switch", 140, 105), Instance("box", 60, 32),
            # помещение 2
            Instance("lamp", 95, 170), Instance("socket", 40, 150, 90), Instance("socket", 150, 190, 270),
            Instance("socket", 75, 220), Instance("socket", 115, 220), Instance("switch", 48, 132),
            Instance("box", 130, 135),
            # зал
            Instance("lamp", 190, 70), Instance("lamp", 260, 70), Instance("lamp", 190, 170),
            Instance("lamp", 260, 170), Instance("socket", 300, 50, 270), Instance("socket", 300, 110, 270),
            Instance("socket", 300, 185, 270), Instance("socket", 200, 20, 180), Instance("socket", 250, 220),
            Instance("socket", 225, 120, 30), Instance("switch", 160, 30, 90), Instance("switch", 290, 210, 180),
            Instance("box", 225, 32), Instance("box", 225, 208),
        ],
    ),
    Sheet(
        number=2,
        title="План расположения электрооборудования. Этаж 2",
        walls=[(40, 20, 300, 20), (300, 20, 300, 220), (300, 220, 40, 220), (40, 220, 40, 20),
               (170, 20, 170, 220)],
        doors=[(170, 110, 9, 90), (40, 180, 9, 0)],
        instances=legend() + [
            Instance("lamp", 80, 60), Instance("lamp", 130, 60), Instance("lamp", 80, 160),
            Instance("lamp", 130, 160), Instance("lamp", 235, 120),
            Instance("socket", 40, 50, 90), Instance("socket", 40, 110, 90), Instance("socket", 105, 20, 180),
            Instance("socket", 170, 70, 270), Instance("socket", 105, 220), Instance("socket", 300, 60, 270),
            Instance("socket", 300, 160, 270), Instance("socket", 235, 220), Instance("socket", 200, 20, 180),
            Instance("socket", 105, 110, 45), Instance("socket", 250, 70, 135),
            Instance("switch", 178, 125, 90), Instance("switch", 48, 190), Instance("switch", 285, 30, 180),
            Instance("box", 150, 30), Instance("box", 270, 200), Instance("box", 60, 205),
        ],
    ),
]


# --- геометрия -------------------------------------------------------------


def transform(inst: Instance, px: float, py: float) -> tuple[float, float]:
    a = math.radians(inst.angle)
    c, s = math.cos(a), math.sin(a)
    return inst.x + px * c - py * s, inst.y + px * s + py * c


def placed_primitives(inst: Instance) -> list[tuple]:
    """Примитивы экземпляра в координатах листа."""
    result = []
    for prim in SYMBOLS[inst.symbol][1]:
        kind = prim[0]
        if kind == "line":
            result.append(("line", *transform(inst, prim[1], prim[2]), *transform(inst, prim[3], prim[4])))
        elif kind == "circle":
            result.append(("circle", *transform(inst, prim[1], prim[2]), prim[3]))
        else:
            cx, cy = transform(inst, prim[1], prim[2])
            result.append(("arc", cx, cy, prim[3], prim[4] + inst.angle, prim[5] + inst.angle))
    return result


def arc_point(cx: float, cy: float, r: float, angle: float) -> tuple[float, float]:
    return cx + r * math.cos(math.radians(angle)), cy + r * math.sin(math.radians(angle))


def bbox(inst: Instance) -> list[float]:
    xs, ys = [], []
    for prim in placed_primitives(inst):
        if prim[0] == "line":
            xs += [prim[1], prim[3]]
            ys += [prim[2], prim[4]]
        elif prim[0] == "circle":
            xs += [prim[1] - prim[3], prim[1] + prim[3]]
            ys += [prim[2] - prim[3], prim[2] + prim[3]]
        else:
            _, cx, cy, r, a0, a1 = prim
            for i in range(65):
                x, y = arc_point(cx, cy, r, a0 + (a1 - a0) * i / 64)
                xs.append(x)
                ys.append(y)
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


# --- SVG -------------------------------------------------------------------


def svg_primitive(prim: tuple) -> str:
    if prim[0] == "line":
        return f'<line x1="{prim[1]:.3f}" y1="{prim[2]:.3f}" x2="{prim[3]:.3f}" y2="{prim[4]:.3f}"/>'
    if prim[0] == "circle":
        return f'<circle cx="{prim[1]:.3f}" cy="{prim[2]:.3f}" r="{prim[3]:g}"/>'
    _, cx, cy, r, a0, a1 = prim
    (sx, sy), (ex, ey) = arc_point(cx, cy, r, a0), arc_point(cx, cy, r, a1)
    large = 1 if (a1 - a0) % 360 > 180 else 0
    return f'<path d="M {sx:.3f} {sy:.3f} A {r:g} {r:g} 0 {large} 1 {ex:.3f} {ey:.3f}"/>'


def text(x: float, y: float, value: str, size: float = 3.5) -> str:
    return f'<text x="{x}" y="{y}" font-size="{size}">{value}</text>'


def sheet_svg(sheet: Sheet) -> str:
    lines = [f'<line x1="{a}" y1="{b}" x2="{c}" y2="{d}"/>' for a, b, c, d in sheet.walls]
    doors = []
    for x, y, r, a0 in sheet.doors:
        doors.append(svg_primitive(("arc", x, y, r, a0, a0 + 90)))
        ex, ey = arc_point(x, y, r, a0)
        doors.append(f'<line x1="{x}" y1="{y}" x2="{ex:.3f}" y2="{ey:.3f}"/>')
    symbols = [svg_primitive(p) for inst in sheet.instances for p in placed_primitives(inst)]
    labels = [text(340, 43.5 + 15 * i, name) for i, (name, _) in enumerate(SYMBOLS.values())]
    return "\n".join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_W:g}mm" height="{PAGE_H:g}mm" '
        f'viewBox="0 0 {PAGE_W:g} {PAGE_H:g}">',
        f'<rect width="{PAGE_W:g}" height="{PAGE_H:g}" fill="white"/>',
        '<g fill="none" stroke="black" stroke-width="0.5">',
        f'<rect x="{FRAME[0]:g}" y="{FRAME[1]:g}" width="{FRAME[2] - FRAME[0]:g}" '
        f'height="{FRAME[3] - FRAME[1]:g}"/>',
        '<rect x="230" y="237" width="185" height="55"/>',
        '<line x1="230" y1="252" x2="415" y2="252"/><line x1="350" y1="252" x2="350" y2="292"/>',
        '<rect x="315" y="20" width="95" height="80"/>',
        "</g>",
        '<g fill="none" stroke="black" stroke-width="0.8">', *lines, "</g>",
        '<g fill="none" stroke="black" stroke-width="0.25">', *doors, "</g>",
        '<g fill="none" stroke="black" stroke-width="0.35">', *symbols, "</g>",
        '<g fill="black" font-family="DejaVu Sans, Arial, sans-serif">',
        text(320, 28, "Условные обозначения", 4),
        *labels,
        text(235, 247, sheet.title, 4.2),
        text(355, 265, f"Лист {sheet.number}"),
        text(355, 275, "Масштаб 1:100"),
        text(235, 265, "Учебный пример для Symbol Search"),
        "</g>",
        "</svg>",
    ])


# --- PDF -------------------------------------------------------------------


def make_pdf(svgs: list[str]) -> bytes:
    out = pymupdf.open()
    for svg in svgs:
        with pymupdf.open(stream=svg.encode(), filetype="svg") as src:
            out.insert_pdf(pymupdf.open("pdf", src.convert_to_pdf()))
    return out.tobytes(garbage=3, deflate=True)


# --- DXF -------------------------------------------------------------------


def flip(x: float, y: float) -> tuple[float, float]:
    """В DXF ось Y направлена вверх."""
    return x, PAGE_H - y


def make_dxf(sheet: Sheet) -> ezdxf.document.Drawing:
    doc = ezdxf.new("R2018", setup=True)
    doc.units = ezdxf.units.MM
    for key, (_, prims) in SYMBOLS.items():
        block = doc.blocks.new(name=key.upper())
        for prim in prims:
            if prim[0] == "line":
                block.add_line((prim[1], -prim[2]), (prim[3], -prim[4]))
            elif prim[0] == "circle":
                block.add_circle((prim[1], -prim[2]), prim[3])
            else:
                block.add_arc((prim[1], -prim[2]), prim[3], -prim[5], -prim[4])

    msp = doc.modelspace()
    fx0, fy0, fx1, fy1 = FRAME
    for x0, y0, x1, y1 in [(fx0, fy0, fx1, fy0), (fx1, fy0, fx1, fy1), (fx1, fy1, fx0, fy1),
                           (fx0, fy1, fx0, fy0),
                           (230, 237, 415, 237), (230, 237, 230, 292), (230, 252, 415, 252),
                           (350, 252, 350, 292), (315, 20, 410, 20), (410, 20, 410, 100),
                           (410, 100, 315, 100), (315, 100, 315, 20), *sheet.walls]:
        msp.add_line(flip(x0, y0), flip(x1, y1))
    for x, y, r, a0 in sheet.doors:
        msp.add_arc(flip(x, y), r, -(a0 + 90), -a0)
        msp.add_line(flip(x, y), flip(*arc_point(x, y, r, a0)))
    for inst in sheet.instances:
        msp.add_blockref(inst.symbol.upper(), flip(inst.x, inst.y), dxfattribs={"rotation": -inst.angle})
    texts = [(320, 28, "Условные обозначения", 4), (235, 247, sheet.title, 4.2),
             (355, 265, f"Лист {sheet.number}", 3.5), (355, 275, "Масштаб 1:100", 3.5),
             (235, 265, "Учебный пример для Symbol Search", 3.5)]
    texts += [(340, 43.5 + 15 * i, name, 3.5) for i, (name, _) in enumerate(SYMBOLS.values())]
    for x, y, value, height in texts:
        msp.add_text(value, height=height * 0.75, dxfattribs={"style": "OpenSans"}).set_placement(flip(x, y))
    return doc


# --- ground truth ----------------------------------------------------------


def annotations(sheet: Sheet, scale: float, dx: float = 0.0, dy: float = 0.0) -> list[dict]:
    result = []
    for inst in sheet.instances:
        x, y, w, h = bbox(inst)
        result.append({
            "symbol": inst.symbol,
            "bbox": [round(v * scale, 3) for v in (x + dx, y + dy, w, h)],
            "angle": inst.angle,
            "legend": inst.legend,
        })
    return result


def main() -> None:
    OUT.mkdir(exist_ok=True)
    svgs = [sheet_svg(sheet) for sheet in SHEETS]
    (OUT / "plan.svg").write_text(svgs[0], encoding="utf-8")
    (OUT / "plan-2pages.pdf").write_bytes(make_pdf(svgs))
    make_dxf(SHEETS[0]).saveas(OUT / "plan.dxf")

    truth = {
        "description": "Эталонная разметка тестовых чертежей. bbox = [x, y, w, h], ось Y вниз.",
        "symbols": {key: name for key, (name, _) in SYMBOLS.items()},
        "files": {
            "plan.svg": {
                "units": "единицы viewBox (мм)",
                "pages": {"1": annotations(SHEETS[0], 1)},
            },
            "plan-2pages.pdf": {
                "units": "pt",
                "pages": {str(s.number): annotations(s, MM_TO_PT) for s in SHEETS},
            },
            "plan.dxf": {
                "units": "pt на странице, которую строит бэкенд (масштаб 1:1, поля 5 мм)",
                "pages": {"1": annotations(SHEETS[0], MM_TO_PT,
                                           DXF_MARGIN - FRAME[0], DXF_MARGIN - FRAME[1])},
            },
        },
    }
    (OUT / "ground_truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    for path in sorted(OUT.iterdir()):
        print(f"{path.relative_to(OUT.parent)}  {path.stat().st_size // 1024} КБ")


if __name__ == "__main__":
    main()
