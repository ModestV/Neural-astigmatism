"""SVG-оверлей. Задача T4."""

from __future__ import annotations

import base64
import math
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import cv2
import numpy as np

from symspot.core.types import Arc, Drawing, Instance, ParentPath, Result, Segment, TextPrim
from symspot.raster.mask import rle_decode
from symspot.render._common import (
    BG_MIN_WIDTH,
    BG_OPACITY,
    DASH_OFF,
    DASH_ON,
    MASK_ALPHA,
    TEXT_FILL_OPACITY,
    arc_is_circle,
    arc_point,
    fmt,
    inst_pieces,
    inst_width,
    instance_ids,
    piece_width,
    size_of,
)
from symspot.render.palette import color_for, color_hex

# Доля высоты рамки текста, занимаемая кеглем фонового текста.
TEXT_FONT_RATIO = 0.8
# Угол (рад), в пределах которого текст считается горизонтальным/вертикальным.
TEXT_AXIS_TOL = 0.3


def _png_b64(arr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", arr)
    if not ok:  # pragma: no cover
        raise RuntimeError("не удалось закодировать PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Фон
# ---------------------------------------------------------------------------


def _parent_d(parent: ParentPath) -> str:
    parts: list[str] = []
    last: tuple[float, float] | None = None
    for it in parent.items:
        pts = it.points
        if it.kind == "text" or not pts:
            continue
        p0 = pts[0]
        if last is None or abs(last[0] - p0[0]) > 1e-9 or abs(last[1] - p0[1]) > 1e-9:
            parts.append(f"M{fmt(p0[0])} {fmt(p0[1])}")
        if it.kind == "line":
            parts.append(f"L{fmt(pts[1][0])} {fmt(pts[1][1])}")
        elif it.kind == "quad":
            parts.append(f"Q{fmt(pts[1][0])} {fmt(pts[1][1])} {fmt(pts[2][0])} {fmt(pts[2][1])}")
        elif it.kind == "cubic":
            parts.append("C" + " ".join(f"{fmt(q[0])} {fmt(q[1])}" for q in pts[1:4]))
        last = pts[-1]
    if parent.closed and len(parts) > 1:
        parts.append("Z")
    return "".join(parts)


def _arc_d(a: Arc) -> str:
    p0, p1 = arc_point(a, a.t0), arc_point(a, a.t1)
    large = 1 if (a.t1 - a.t0) > math.pi else 0
    # Ось Y вниз: рост угла t идет по часовой стрелке на экране = sweep-flag 1.
    return f"M{fmt(p0[0])} {fmt(p0[1])}A{fmt(a.r)} {fmt(a.r)} 0 {large} 1 {fmt(p1[0])} {fmt(p1[1])}"


def _axis_near(a: float, b: float) -> bool:
    return abs(math.remainder(a - b, math.pi)) < TEXT_AXIS_TOL


def _text_element(t: TextPrim) -> str:
    x0, y0, x1, y1 = t.bbox
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ang = (t.angle + math.pi) % (2 * math.pi) - math.pi
    if _axis_near(ang, 0.0):
        height, length = h, w
    elif _axis_near(ang, math.pi / 2):
        height, length = w, h
    else:
        height, length = min(w, h), max(w, h)
    size = max(height * TEXT_FONT_RATIO, 1e-6)
    return (
        f'<text transform="translate({fmt(cx)} {fmt(cy)}) rotate({fmt(math.degrees(ang), 2)})" '
        f'font-size="{fmt(size)}" text-anchor="middle" dominant-baseline="central" '
        f'textLength="{fmt(length)}" lengthAdjust="spacingAndGlyphs" font-family="sans-serif" '
        f'fill="#000" stroke="none">{escape(t.text)}</text>'
    )


def _background_vector(drawing: Drawing) -> str:
    by_width: dict[float, list[str]] = defaultdict(list)
    if drawing.parents:
        for parent in drawing.parents.values():
            d = _parent_d(parent)
            if d:
                by_width[round(max(parent.style.stroke_width, BG_MIN_WIDTH), 3)].append(d)
    else:
        for p in drawing.primitives:
            if isinstance(p, Segment):
                d = f"M{fmt(p.p0[0])} {fmt(p.p0[1])}L{fmt(p.p1[0])} {fmt(p.p1[1])}"
            elif isinstance(p, Arc):
                c = p.center
                d = (
                    f"M{fmt(c[0] - p.r)} {fmt(c[1])}a{fmt(p.r)} {fmt(p.r)} 0 1 0 {fmt(2 * p.r)} 0"
                    f"a{fmt(p.r)} {fmt(p.r)} 0 1 0 {fmt(-2 * p.r)} 0"
                    if arc_is_circle(p)
                    else _arc_d(p)
                )
            else:
                continue
            by_width[round(max(piece_width(p), BG_MIN_WIDTH), 3)].append(d)
    lines = [
        f'<g id="background" opacity="{fmt(BG_OPACITY, 2)}" fill="none" stroke="#000" '
        'stroke-linecap="round" stroke-linejoin="round">'
    ]
    for w, ds in sorted(by_width.items()):
        lines.append(f'<path stroke-width="{fmt(w)}" d="{"".join(ds)}"/>')
    texts = [_text_element(p) for p in drawing.primitives if isinstance(p, TextPrim)]
    if texts:
        lines.append('<g id="background-text" stroke="none">')
        lines.extend(texts)
        lines.append("</g>")
    lines.append("</g>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Экземпляры
# ---------------------------------------------------------------------------


def _inst_attrs(inst: Instance, eff_id: int, label: str) -> str:
    x0, y0, x1, y1 = inst.bbox
    return (
        f'id="inst-{eff_id}" class="inst{" amb" if inst.status == "ambiguous" else ""}" '
        f"data-id={quoteattr(str(eff_id))} data-legend={quoteattr(inst.legend_id)} "
        f"data-label={quoteattr(label)} "
        f'data-status="{inst.status}" data-score="{fmt(inst.score, 4)}" '
        f'data-coverage="{fmt(inst.coverage, 4)}" '
        f'data-bbox="{fmt(x0)} {fmt(y0)} {fmt(x1)} {fmt(y1)}"'
    )


def _instance_vector(inst: Instance, eff_id: int, label: str, prims: dict) -> str:
    color = color_hex(inst.legend_id)
    pieces = list(inst_pieces(prims, inst))
    widths = [inst_width(p) for p in pieces if not isinstance(p, TextPrim)]
    base_w = max(set(widths), key=widths.count) if widths else 0.8
    dash = ""
    if inst.status == "ambiguous":
        dash = f' stroke-dasharray="{fmt(DASH_ON * base_w, 3)} {fmt(DASH_OFF * base_w, 3)}"'
    out = [
        f'<g {_inst_attrs(inst, eff_id, label)} fill="none" stroke="{color}" '
        f'stroke-width="{fmt(base_w, 3)}" stroke-linecap="round"{dash}>'
    ]
    for p in pieces:
        w_attr = ""
        if not isinstance(p, TextPrim) and abs(inst_width(p) - base_w) > 1e-6:
            w_attr = f' stroke-width="{fmt(inst_width(p), 3)}"'
        if isinstance(p, Segment):
            out.append(
                f'<line x1="{fmt(p.p0[0])}" y1="{fmt(p.p0[1])}" x2="{fmt(p.p1[0])}" '
                f'y2="{fmt(p.p1[1])}"{w_attr}/>'
            )
        elif isinstance(p, Arc):
            if arc_is_circle(p):
                out.append(
                    f'<circle cx="{fmt(p.center[0])}" cy="{fmt(p.center[1])}" '
                    f'r="{fmt(p.r)}"{w_attr}/>'
                )
            else:
                out.append(f'<path d="{_arc_d(p)}"{w_attr}/>')
        elif isinstance(p, TextPrim):
            x0, y0, x1, y1 = p.bbox
            out.append(
                f'<rect x="{fmt(x0)}" y="{fmt(y0)}" width="{fmt(x1 - x0)}" '
                f'height="{fmt(y1 - y0)}" fill="{color}" '
                f'fill-opacity="{fmt(TEXT_FILL_OPACITY, 2)}"/>'
            )
    out.append("</g>")
    return "\n".join(out)


def _instance_raster(inst: Instance, eff_id: int, label: str) -> str:
    color = color_hex(inst.legend_id)
    r, g, b = color_for(inst.legend_id)
    out = [f"<g {_inst_attrs(inst, eff_id, label)}>"]
    if inst.mask_rle is not None:
        mask, (ox, oy) = rle_decode(inst.mask_rle)
        h, w = mask.shape
        if h > 0 and w > 0:
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[..., 0], rgba[..., 1], rgba[..., 2] = b, g, r  # cv2: BGRA
            rgba[..., 3] = np.where(mask, MASK_ALPHA, 0).astype(np.uint8)
            out.append(
                f'<image x="{ox}" y="{oy}" width="{w}" height="{h}" '
                f'xlink:href="data:image/png;base64,{_png_b64(rgba)}"/>'
            )
    if inst.mask_rle is None or inst.status == "ambiguous":
        x0, y0, x1, y1 = inst.bbox
        dash = ' stroke-dasharray="6 4"' if inst.status == "ambiguous" else ""
        out.append(
            f'<rect x="{fmt(x0)}" y="{fmt(y0)}" width="{fmt(x1 - x0)}" height="{fmt(y1 - y0)}" '
            f'fill="none" stroke="{color}" stroke-width="1.5"{dash}/>'
        )
    out.append("</g>")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Сборка
# ---------------------------------------------------------------------------


def build_svg(
    drawing: Drawing, result: Result, *, xml_header: bool = True, responsive: bool = False
) -> str:
    """Собрать SVG-оверлей строкой (общая часть для SVG, PDF, PNG и HTML)."""
    width, height = size_of(drawing)
    labels = {q.legend_id: (q.label or "") for q in result.queries}
    ids = instance_ids(result)
    size_attrs = (
        'width="100%" height="100%"'
        if responsive
        else f'width="{fmt(width)}" height="{fmt(height)}"'
    )
    head = '<?xml version="1.0" encoding="UTF-8"?>\n' if xml_header else ""
    parts = [
        head + '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'{size_attrs} viewBox="0 0 {fmt(width)} {fmt(height)}">',
        f'<rect id="paper" x="0" y="0" width="{fmt(width)}" height="{fmt(height)}" fill="#fff"/>',
        '<g id="viewport">',
    ]
    if drawing.kind == "raster":
        if drawing.image is not None:
            img = np.asarray(drawing.image)
            parts.append(
                '<g id="background">'
                f'<image x="0" y="0" width="{fmt(width)}" height="{fmt(height)}" '
                f'xlink:href="data:image/png;base64,{_png_b64(img)}"/></g>'
            )
        else:
            parts.append('<g id="background"></g>')
    else:
        parts.append(_background_vector(drawing))

    parts.append('<g id="instances">')
    prims = drawing.prim_by_id() if drawing.kind == "vector" else {}
    for inst, eff_id in zip(result.instances, ids, strict=True):
        label = labels.get(inst.legend_id, "")
        if drawing.kind == "raster":
            parts.append(_instance_raster(inst, eff_id, label))
        else:
            parts.append(_instance_vector(inst, eff_id, label, prims))
    parts.append("</g>")
    parts.append("</g>")
    parts.append("</svg>")
    return "\n".join(parts)


def render_svg(drawing: Drawing, result: Result, out_path: str | Path) -> Path:
    """Исходный чертеж серым (``drawing.parents``, opacity 0.35) + цветной слой: точная
    геометрия кусков ``instance.piece_ids`` (Segment -> line, Arc -> path A, TextPrim ->
    рамка/текст) цветом ``color_for(legend_id)``; ambiguous — пунктир. Для растрового
    чертежа фон = встроенное PNG, слой = маски ``mask_rle``. Размер = размер чертежа,
    viewBox в единицах чертежа. У каждого экземпляра группа ``<g id="inst-<id>">``.
    ``drawing`` должен быть нормализован (куски с теми же id, что в result).
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_svg(drawing, result), encoding="utf-8")
    return out
