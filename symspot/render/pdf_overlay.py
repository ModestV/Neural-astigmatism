"""PDF-оверлей. Задача T4."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import pymupdf

from symspot.core.types import Arc, Drawing, Point, Result, Segment, TextPrim
from symspot.render._cairo import load_cairosvg
from symspot.render._common import (
    BG_OPACITY,
    DASH_OFF,
    DASH_ON,
    TEXT_FILL_OPACITY,
    arc_beziers,
    arc_is_circle,
    inst_pieces,
    inst_width,
)
from symspot.render.palette import color_for
from symspot.render.svg_overlay import build_svg

# Белая вуаль поверх исходной страницы: фон выглядит бледнее, цвет экземпляров заметнее.
VEIL_OPACITY = 1.0 - BG_OPACITY
# Округление толщины для группировки кусков (pt).
WIDTH_ROUND = 2
# 1 единица SVG = 1 pt в PDF (по умолчанию cairosvg считает 96 dpi).
PDF_DPI = 72


def _is_pdf_source(drawing: Drawing) -> bool:
    return (
        drawing.kind == "vector"
        and bool(drawing.source_path)
        and drawing.source_path.lower().endswith(".pdf")
        and Path(drawing.source_path).is_file()
    )


def _overlay_on_page(page: pymupdf.Page, drawing: Drawing, result: Result) -> None:
    """Нарисовать вуаль и куски экземпляров; координаты чертежа = отображаемая страница."""
    dm = page.derotation_matrix
    rotated = page.rotation != 0

    def P(p: Point) -> pymupdf.Point:
        q = pymupdf.Point(p[0], p[1])
        return q * dm if rotated else q

    # Вуаль на всю страницу (в неповернутых координатах).
    veil = page.new_shape()
    veil.draw_rect(page.rect * dm if rotated else page.rect)
    veil.finish(color=None, fill=(1, 1, 1), fill_opacity=VEIL_OPACITY)
    veil.commit()

    prims = drawing.prim_by_id()
    for inst in result.instances:
        rgb = tuple(c / 255.0 for c in color_for(inst.legend_id))
        groups: dict[float, list] = defaultdict(list)
        rects: list[TextPrim] = []
        for p in inst_pieces(prims, inst):
            if isinstance(p, TextPrim):
                rects.append(p)
            else:
                groups[round(inst_width(p), WIDTH_ROUND)].append(p)
        for width, pieces in groups.items():
            sh = page.new_shape()
            for p in pieces:
                if isinstance(p, Segment):
                    sh.draw_line(P(p.p0), P(p.p1))
                elif isinstance(p, Arc):
                    if arc_is_circle(p):
                        # Полный круг как 4 кривые: draw_circle не поддерживает поворот.
                        for b in arc_beziers(Arc(p.id, p.center, p.r, 0.0, math.tau, 0)):
                            sh.draw_bezier(P(b[0]), P(b[1]), P(b[2]), P(b[3]))
                    else:
                        for b in arc_beziers(p):
                            sh.draw_bezier(P(b[0]), P(b[1]), P(b[2]), P(b[3]))
            dashes = (
                f"[{DASH_ON * width:.2f} {DASH_OFF * width:.2f}] 0"
                if inst.status == "ambiguous"
                else None
            )
            sh.finish(color=rgb, fill=None, width=width, dashes=dashes, lineCap=1, closePath=False)
            sh.commit()
        for t in rects:
            sh = page.new_shape()
            x0, y0, x1, y1 = t.bbox
            sh.draw_polyline([P((x0, y0)), P((x1, y0)), P((x1, y1)), P((x0, y1))])
            sh.finish(
                color=rgb, fill=rgb, fill_opacity=TEXT_FILL_OPACITY, width=0.5, closePath=True
            )
            sh.commit()


def render_pdf(drawing: Drawing, result: Result, out_path: str | Path) -> Path:
    """Копия исходной страницы PDF (``drawing.source_path``, ``drawing.page``) + куски
    экземпляров поверх через ``page.new_shape()`` (с учетом rotation страницы:
    координаты чертежа -> координаты страницы через ``page.derotation_matrix``).
    Для не-PDF источника — PDF из SVG-оверлея (cairosvg).
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if _is_pdf_source(drawing):
        with pymupdf.open(drawing.source_path) as src, pymupdf.open() as doc:
            doc.insert_pdf(src, from_page=drawing.page, to_page=drawing.page)
            _overlay_on_page(doc[0], drawing, result)
            doc.save(str(out), garbage=3, deflate=True)
        return out
    svg = build_svg(drawing, result)
    load_cairosvg().svg2pdf(bytestring=svg.encode("utf-8"), write_to=str(out), dpi=PDF_DPI)
    return out
