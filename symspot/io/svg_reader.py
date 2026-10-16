"""Чтение SVG (svgelements). Задача T2."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from svgelements import (
    SVG,
    Arc,
    Close,
    CubicBezier,
    Line,
    Matrix,
    Move,
    QuadraticBezier,
    Shape,
    Text,
)

from symspot.core.types import Drawing, ParentPath, PathItem, Point, Style, TextPrim

# Число точек сэмплирования одной кривой (с концами) в ``ParentPath.polyline``.
CURVE_SAMPLES = 16
# Допуск стыковки сегментов в подпуть: доля большей стороны viewBox.
JOIN_TOL_REL = 1e-6
# Оценка рамки текста, когда метрик шрифта нет: ширина = k * font_size * len(text),
# от базовой линии вверх на ASCENT * font_size, вниз на DESCENT * font_size.
TEXT_WIDTH_PER_CHAR = 0.6
TEXT_ASCENT = 0.8
TEXT_DESCENT = 0.2
DEFAULT_FONT_SIZE = 16.0


def _bernstein(n_ctrl: int, n_samples: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n_samples)
    deg = n_ctrl - 1
    return np.stack(
        [math.comb(deg, k) * t**k * (1.0 - t) ** (deg - k) for k in range(n_ctrl)], axis=1
    )


_B_CUBIC = _bernstein(4, CURVE_SAMPLES)
_B_QUAD = _bernstein(3, CURVE_SAMPLES)


def _sample_items(items: list[PathItem]) -> np.ndarray:
    """Сэмплированная полилиния (N, 2) для связной цепочки items."""
    chunks: list[np.ndarray] = []
    for k, it in enumerate(items):
        pts = np.asarray(it.points, dtype=float)
        if it.kind == "cubic":
            pts = _B_CUBIC @ pts
        elif it.kind == "quad":
            pts = _B_QUAD @ pts
        chunks.append(pts if k == 0 else pts[1:])
    if not chunks:
        return np.zeros((0, 2), dtype=float)
    return np.vstack(chunks)


def _close(a: Point, b: Point, tol: float) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def _mat3(m: Matrix) -> np.ndarray:
    return np.array([[m.a, m.c, m.e], [m.b, m.d, m.f], [0.0, 0.0, 1.0]], dtype=float)


def _color(c: object) -> tuple[float, float, float] | None:
    try:
        r, g, b = c.red, c.green, c.blue  # type: ignore[attr-defined]
    except AttributeError:
        return None
    if r is None or g is None or b is None:
        return None
    return (r / 255.0, g / 255.0, b / 255.0)


def read_svg(path: str | Path) -> Drawing:
    """Прочитать SVG в ``Drawing(kind="vector")``.

    - Координаты в пользовательских единицах корневого ``viewBox`` (если он есть), иначе в px;
      все transform применены, ``<use>`` развернут, ось Y вниз.
    - Каждый геометрический элемент разбивается на связные подпути -> ``ParentPath``
      (line/quad/cubic; дуги SVG переводятся в cubic), ``group`` = порядковый номер элемента,
      ``source_ref`` = ``"svg:#<id>"`` если у элемента есть id, иначе ``"svg:<номер>"``.
    - ``<text>`` -> ``TextPrim`` (bbox по позиции и font-size, если точный недоступен)
      + ``ParentPath`` с одним item ``text``.
    - ``width``/``height`` = размер viewBox, ``units_per_px`` = 1.0.

    Номер элемента (``group``) сквозной по всем геометрическим элементам и ``<text>`` в порядке
    обхода документа. Без ``viewBox`` и без ``width``/``height`` svgelements подставляет
    1000 x 1000.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"SVG не найден: {p}")
    svg = SVG.parse(str(p), reify=True)

    vb = svg.viewbox
    if vb is not None:
        width, height = float(vb.width), float(vb.height)
        to_user = np.linalg.inv(_mat3(Matrix(svg.viewbox_transform)))
    else:
        width, height = float(svg.width), float(svg.height)
        to_user = np.eye(3)
    tol = JOIN_TOL_REL * max(width, height)
    ua, uc, ue = float(to_user[0, 0]), float(to_user[0, 1]), float(to_user[0, 2])
    ub, ud, uf = float(to_user[1, 0]), float(to_user[1, 1]), float(to_user[1, 2])
    len_scale = math.sqrt(abs(ua * ud - uc * ub))

    def pt(q: object) -> Point:
        x, y = float(q.x), float(q.y)  # type: ignore[attr-defined]
        return (ua * x + uc * y + ue, ub * x + ud * y + uf)

    parents: dict[int, ParentPath] = {}
    primitives: list[TextPrim] = []
    next_id = 0
    group = -1

    for el in svg.elements():
        if isinstance(el, Shape):
            group += 1
            next_id = _read_shape(el, group, next_id, parents, pt, tol, len_scale)
        elif isinstance(el, Text):
            group += 1
            text = el.text
            if text is None or not text.strip():
                continue
            fs = float(el.font_size) if el.font_size else DEFAULT_FONT_SIZE
            w = TEXT_WIDTH_PER_CHAR * fs * len(text)
            anchor = getattr(el, "anchor", "start")
            x = float(el.x or 0.0)
            y = float(el.y or 0.0)
            x0 = x - w / 2 if anchor == "middle" else x - w if anchor == "end" else x
            local = np.array(
                [
                    [x0, y - TEXT_ASCENT * fs, 1.0],
                    [x0 + w, y - TEXT_ASCENT * fs, 1.0],
                    [x0 + w, y + TEXT_DESCENT * fs, 1.0],
                    [x0, y + TEXT_DESCENT * fs, 1.0],
                ]
            )
            full = to_user @ _mat3(el.transform)
            corners = local @ full.T
            bbox = (
                float(corners[:, 0].min()),
                float(corners[:, 1].min()),
                float(corners[:, 0].max()),
                float(corners[:, 1].max()),
            )
            angle = math.atan2(full[1, 0], full[0, 0])
            ref = f"svg:#{el.id}" if el.id else f"svg:{group}"
            parents[next_id] = ParentPath(
                id=next_id,
                items=[PathItem("text", ())],
                polyline=np.zeros((0, 2), dtype=float),
                group=group,
                source_ref=ref,
            )
            primitives.append(
                TextPrim(id=next_id + 1, text=text, bbox=bbox, parent_id=next_id, angle=angle)
            )
            next_id += 2

    return Drawing(
        kind="vector",
        width=width,
        height=height,
        units_per_px=1.0,
        primitives=list(primitives),
        parents=parents,
        source_path=str(p),
        page=0,
        normalized=False,
    )


def _read_shape(
    el: Shape,
    group: int,
    next_id: int,
    parents: dict[int, ParentPath],
    pt,
    tol: float,
    len_scale: float,
) -> int:
    """Разбить геометрический элемент на подпути и добавить в ``parents``; вернуть next_id."""
    stroke = _color(el.stroke) if el.stroke is not None and el.stroke.value is not None else None
    fill_c = _color(el.fill) if el.fill is not None and el.fill.value is not None else None
    sw = float(el.stroke_width) if (stroke is not None and el.stroke_width) else 0.0
    style = Style(stroke_width=sw * len_scale, color=stroke if stroke is not None else fill_c)
    fill_only = stroke is None and fill_c is not None
    ref = f"svg:#{el.id}" if el.id else f"svg:{group}"

    items: list[PathItem] = []
    closed_flag = False

    def flush() -> None:
        nonlocal items, closed_flag, next_id
        if items:
            first, last = items[0].points[0], items[-1].points[-1]
            degenerate = len(items) == 1 and items[0].kind == "line"
            closed = closed_flag or (not degenerate and _close(first, last, tol))
            parents[next_id] = ParentPath(
                id=next_id,
                items=items,
                polyline=_sample_items(items),
                style=style,
                closed=closed,
                group=group,
                source_ref=ref,
                fill=fill_only,
            )
            next_id += 1
        items = []
        closed_flag = False

    for seg in el.segments(transformed=True):
        if isinstance(seg, Move):
            flush()
            continue
        if isinstance(seg, Close):
            if items:
                start, cur = items[0].points[0], items[-1].points[-1]
                if not _close(start, cur, tol):
                    items.append(PathItem("line", (cur, start)))
                closed_flag = True
                flush()
            continue
        if seg.start is None or seg.end is None:
            continue
        s = pt(seg.start)
        if items and not _close(items[-1].points[-1], s, tol):
            flush()
        if isinstance(seg, Line):
            e = pt(seg.end)
            if not _close(s, e, tol):
                items.append(PathItem("line", (s, e)))
        elif isinstance(seg, QuadraticBezier):
            items.append(PathItem("quad", (s, pt(seg.control), pt(seg.end))))
        elif isinstance(seg, CubicBezier):
            items.append(PathItem("cubic", (s, pt(seg.control1), pt(seg.control2), pt(seg.end))))
        elif isinstance(seg, Arc):
            curves = list(seg.as_cubic_curves())
            if not curves:
                e = pt(seg.end)
                if not _close(s, e, tol):
                    items.append(PathItem("line", (s, e)))
                continue
            prev = s
            for cv in curves:
                c1, c2, e = pt(cv.control1), pt(cv.control2), pt(cv.end)
                items.append(PathItem("cubic", (prev, c1, c2, e)))
                prev = e
    flush()
    return next_id
