"""Чтение векторного PDF (PyMuPDF). Задача T2."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pymupdf

from symspot.core.types import Drawing, ParentPath, PathItem, Point, Style, TextPrim

# Число точек сэмплирования одной кривой (с концами) в ``ParentPath.polyline``.
CURVE_SAMPLES = 16
# Допуск стыковки items в подпуть: доля большей стороны страницы.
JOIN_TOL_REL = 1e-6


def _bernstein(n_ctrl: int, n_samples: int) -> np.ndarray:
    """Матрица (n_samples, n_ctrl) базисных многочленов Бернштейна степени n_ctrl-1."""
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


def _rect_items(corners: list[Point]) -> list[PathItem]:
    """Замкнутый контур из 4 line по углам."""
    n = len(corners)
    return [PathItem("line", (corners[i], corners[(i + 1) % n])) for i in range(n)]


def _split_subpaths(chains: list[list[PathItem]], tol: float) -> list[list[PathItem]]:
    """Разбить последовательность item-цепочек на связные подпути.

    ``chains`` - список групп items; группа (``re``/``qu``) всегда образует отдельный подпуть,
    одиночные items (``l``/``c``) склеиваются, пока начало очередного совпадает с концом
    предыдущего.
    """
    out: list[list[PathItem]] = []
    cur: list[PathItem] = []
    for chain in chains:
        if len(chain) > 1:  # прямоугольник/квад: свой подпуть
            if cur:
                out.append(cur)
                cur = []
            out.append(chain)
            continue
        it = chain[0]
        if cur and not _close(cur[-1].points[-1], it.points[0], tol):
            out.append(cur)
            cur = []
        cur.append(it)
    if cur:
        out.append(cur)
    return out


def _rgb(color: object) -> tuple[float, float, float] | None:
    if color is None:
        return None
    try:
        c = tuple(float(x) for x in color)  # type: ignore[attr-defined]
    except TypeError:
        return None
    if len(c) == 3:
        return c  # type: ignore[return-value]
    if len(c) == 1:
        return (c[0], c[0], c[0])
    return None


def read_pdf(path: str | Path, page: int = 0) -> Drawing:
    """Прочитать страницу PDF в ``Drawing(kind="vector")``.

    - ``page.get_drawings()``: каждый drawing разбивается на связные подпути (по разрывам
      между концом предыдущего item и началом следующего); каждый подпуть -> ``ParentPath``
      с items ``line``/``cubic``/``quad`` (``re`` и ``qu`` раскладываются на 4 line),
      ``group`` = индекс drawing, ``source_ref`` = ``"pdf:<индекс>"``, ``style`` из width/color,
      ``fill`` = есть заливка без обводки.
    - ``page.get_text("dict")``: каждый span -> ``TextPrim`` (+ ``ParentPath`` с одним item
      ``text`` и пустой полилинией); ``angle`` из ``line["dir"]``.
    - Координаты в pt, ось Y вниз, с учетом ``page.rotation`` (как на отображаемой странице,
      через ``page.rotation_matrix``). ``width``/``height`` = размер отображаемой страницы.
    - ``units_per_px`` = 1.0, ``source_path``, ``page`` заполнены, ``normalized=False``.
    - id родителей и TextPrim уникальны, начинаются с 0, идут подряд.

    ``group`` текстовых родителей продолжает нумерацию drawing'ов (n_drawings + номер span'а),
    ``source_ref`` = ``"pdf:t<номер span'а>"``.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"PDF не найден: {p}")
    with pymupdf.open(p) as doc:
        pg = doc.load_page(page)
        rect = pg.rect
        width, height = float(rect.width), float(rect.height)
        tol = JOIN_TOL_REL * max(width, height)

        if pg.rotation != 0:
            m = pg.rotation_matrix
            ma, mb, mc, md, me, mf = m.a, m.b, m.c, m.d, m.e, m.f
        else:
            ma, mb, mc, md, me, mf = 1.0, 0.0, 0.0, 1.0, 0.0, 0.0
        rotated = pg.rotation != 0

        def tp(x: float, y: float) -> Point:
            if not rotated:
                return (float(x), float(y))
            return (ma * x + mc * y + me, mb * x + md * y + mf)

        parents: dict[int, ParentPath] = {}
        next_id = 0

        drawings = pg.get_drawings()
        for gi, d in enumerate(drawings):
            chains: list[list[PathItem]] = []
            for item in d["items"]:
                op = item[0]
                if op == "l":
                    chains.append(
                        [PathItem("line", (tp(item[1].x, item[1].y), tp(item[2].x, item[2].y)))]
                    )
                elif op == "c":
                    pts = tuple(tp(q.x, q.y) for q in item[1:5])
                    chains.append([PathItem("cubic", pts)])
                elif op == "re":
                    r = item[1]
                    corners = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
                    chains.append(_rect_items([tp(x, y) for x, y in corners]))
                elif op == "qu":
                    q = item[1]
                    corners = [q.ul, q.ur, q.lr, q.ll]
                    chains.append(_rect_items([tp(c.x, c.y) for c in corners]))
                # прочие операторы (в т.ч. неизвестные) пропускаются

            if not chains:
                continue
            subpaths = _split_subpaths(chains, tol)

            stroke = _rgb(d.get("color"))
            fill_c = _rgb(d.get("fill"))
            fill_only = stroke is None and fill_c is not None
            width_pt = d.get("width")
            style = Style(
                stroke_width=float(width_pt) if width_pt is not None else 0.0,
                color=stroke if stroke is not None else fill_c,
            )
            explicit_close = bool(d.get("closePath"))

            for si, items in enumerate(subpaths):
                first, last = items[0].points[0], items[-1].points[-1]
                closed = _close(first, last, tol)
                degenerate = len(items) == 1 and items[0].kind == "line"
                if not closed and not degenerate and explicit_close and si == len(subpaths) - 1:
                    items = [*items, PathItem("line", (last, first))]
                    closed = True
                parents[next_id] = ParentPath(
                    id=next_id,
                    items=items,
                    polyline=_sample_items(items),
                    style=style,
                    closed=closed,
                    group=gi,
                    source_ref=f"pdf:{gi}",
                    fill=fill_only,
                )
                next_id += 1

        primitives: list[TextPrim] = []
        n_groups = len(drawings)
        span_no = 0
        for block in pg.get_text("dict").get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                dx, dy = line.get("dir", (1.0, 0.0))
                rdx = ma * dx + mc * dy
                rdy = mb * dx + md * dy
                angle = math.atan2(rdy, rdx)
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if not text.strip():
                        continue
                    x0, y0, x1, y1 = span["bbox"]
                    cs = [tp(x0, y0), tp(x1, y0), tp(x1, y1), tp(x0, y1)]
                    xs = [c[0] for c in cs]
                    ys = [c[1] for c in cs]
                    bbox = (min(xs), min(ys), max(xs), max(ys))
                    pid = next_id
                    parents[pid] = ParentPath(
                        id=pid,
                        items=[PathItem("text", ())],
                        polyline=np.zeros((0, 2), dtype=float),
                        group=n_groups + span_no,
                        source_ref=f"pdf:t{span_no}",
                    )
                    primitives.append(
                        TextPrim(id=pid + 1, text=text, bbox=bbox, parent_id=pid, angle=angle)
                    )
                    next_id += 2
                    span_no += 1

    return Drawing(
        kind="vector",
        width=width,
        height=height,
        units_per_px=1.0,
        primitives=list(primitives),
        parents=parents,
        source_path=str(p),
        page=page,
        normalized=False,
    )
