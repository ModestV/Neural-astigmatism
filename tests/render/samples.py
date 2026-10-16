"""Синтетические чертежи и результаты для тестов и примеров рендеринга."""

from __future__ import annotations

import math

import numpy as np

from symspot.core.types import (
    Arc,
    Drawing,
    Instance,
    ParentPath,
    PathItem,
    QuerySpec,
    Result,
    Segment,
    Similarity,
    Style,
    TextPrim,
)
from symspot.raster.mask import rle_encode

W, H = 220.0, 160.0
KAPPA = 4.0 / 3.0 * math.tan(math.pi / 8)


def _circle_items(cx: float, cy: float, r: float) -> list[PathItem]:
    k = KAPPA * r
    pts = [
        ((cx + r, cy), (cx + r, cy + k), (cx + k, cy + r), (cx, cy + r)),
        ((cx, cy + r), (cx - k, cy + r), (cx - r, cy + k), (cx - r, cy)),
        ((cx - r, cy), (cx - r, cy - k), (cx - k, cy - r), (cx, cy - r)),
        ((cx, cy - r), (cx + k, cy - r), (cx + r, cy - k), (cx + r, cy)),
    ]
    return [PathItem("cubic", p) for p in pts]


class _Builder:
    def __init__(self) -> None:
        self.prims: list = []
        self.parents: dict[int, ParentPath] = {}
        self.next_parent = 0
        self.next_prim = 100

    def parent(self, items: list[PathItem], width: float = 0.5, closed: bool = False) -> int:
        pid = self.next_parent
        self.next_parent += 1
        pts = np.array([p for it in items for p in it.points], dtype=float).reshape(-1, 2)
        self.parents[pid] = ParentPath(
            id=pid, items=items, polyline=pts, style=Style(stroke_width=width), closed=closed
        )
        return pid

    def seg(self, p0, p1, parent: int, width: float = 0.5) -> int:
        i = self.next_prim
        self.next_prim += 1
        self.prims.append(Segment(i, p0, p1, parent, Style(stroke_width=width)))
        return i

    def arc(self, c, r, t0, t1, parent: int, width: float = 0.5) -> int:
        i = self.next_prim
        self.next_prim += 1
        self.prims.append(Arc(i, c, r, t0, t1, parent, Style(stroke_width=width)))
        return i

    def text(self, s: str, bbox, parent: int) -> int:
        i = self.next_prim
        self.next_prim += 1
        self.prims.append(TextPrim(i, s, bbox, parent))
        return i


def _inst(legend: str, ids: list[int], bbox, status: str = "ok", iid: int = -1, **kw) -> Instance:
    return Instance(
        legend_id=legend,
        T=Similarity.identity(),
        score=kw.pop("score", 0.9),
        coverage=kw.pop("coverage", 0.95),
        piece_ids=ids,
        parent_ids=[],
        bbox=bbox,
        status=status,  # type: ignore[arg-type]
        id=iid,
        **kw,
    )


def make_vector_sample() -> tuple[Drawing, Result]:
    """Окружность с диагональю (x2), квадрат с крестом (ambiguous), дверь (дуга 90 град.),
    стена через окружность, текст."""
    b = _Builder()
    wall_y = 50.0
    # символы "окружность с диагональю"
    circ_ids: list[list[int]] = []
    for cx in (50.0, 150.0):
        pc = b.parent(_circle_items(cx, wall_y, 12.0), closed=True)
        pd = b.parent(
            [PathItem("line", ((cx - 8.5, wall_y + 8.5), (cx + 8.5, wall_y - 8.5)))], width=0.5
        )
        circ_ids.append(
            [
                b.arc((cx, wall_y), 12.0, 0.0, 2 * math.pi, pc),
                b.seg((cx - 8.5, wall_y + 8.5), (cx + 8.5, wall_y - 8.5), pd),
            ]
        )
    # стена через оба символа: куски вне и внутри кругов (внутренние не назначены)
    pw = b.parent([PathItem("line", ((0.0, wall_y), (W, wall_y)))], width=1.2)
    for xa, xb in [(0, 38), (38, 62), (62, 138), (138, 162), (162, W)]:
        b.seg((float(xa), wall_y), (float(xb), wall_y), pw, width=1.2)
    # квадрат с крестом (ambiguous)
    sq = []
    x0, y0, s = 40.0, 100.0, 24.0
    corners = [(x0, y0), (x0 + s, y0), (x0 + s, y0 + s), (x0, y0 + s)]
    for k in range(4):
        pa, pb = corners[k], corners[(k + 1) % 4]
        pid = b.parent([PathItem("line", (pa, pb))])
        sq.append(b.seg(pa, pb, pid))
    pcross = b.parent([PathItem("line", (corners[0], corners[2]))])
    sq.append(b.seg(corners[0], corners[2], pcross))
    # дверь: стойка + четверть окружности
    dx, dy, r = 120.0, 130.0, 22.0
    pdoor = b.parent([PathItem("line", ((dx, dy), (dx + r, dy)))], width=0.8)
    door = [
        b.seg((dx, dy), (dx + r, dy), pdoor, width=0.8),
        b.arc((dx, dy), r, 0.0, math.pi / 2, pdoor, width=0.5),
    ]
    # дуга больше полукруга (large-arc)
    pbig = b.parent([PathItem("line", ((190.0, 120.0), (191.0, 120.0)))])
    big = b.arc((190.0, 125.0), 12.0, math.pi / 2, math.pi / 2 + 1.5 * math.pi, pbig)
    # подпись
    pt = b.parent([PathItem("text", ())])
    txt = b.text("P-101", (60.0, 70.0, 90.0, 80.0), pt)

    drawing = Drawing(
        kind="vector",
        width=W,
        height=H,
        units_per_px=1.0,
        primitives=b.prims,
        parents=b.parents,
        source_path="",
        normalized=True,
    )
    insts = [
        _inst("circ_diag", circ_ids[0], (38, 38, 62, 62), iid=0),
        _inst("circ_diag", circ_ids[1], (138, 38, 162, 62), iid=1, score=0.8),
        _inst(
            "square_x",
            sq,
            (40, 100, 64, 124),
            status="ambiguous",
            iid=2,
            rivals=["circ_diag"],
            score=0.55,
        ),
        _inst("door", door, (120, 130, 142, 152), iid=3, source="exact"),
        _inst("tail", [big], (178, 113, 202, 137), iid=4),
        _inst("label", [txt], (60, 70, 90, 80), iid=5),
    ]
    result = Result(
        drawing="sample.svg",
        instances=insts,
        timings={"read": 0.01, "search": 0.25, "verify": 0.1},
        config={},
        queries=[
            QuerySpec("circ_diag", (38, 38, 62, 62), "Окружность с диагональю"),
            QuerySpec("square_x", (40, 100, 64, 124), "Квадрат"),
        ],
    )
    return drawing, result


def make_raster_sample() -> tuple[Drawing, Result]:
    """Растр 80x120: линия + круг; у экземпляров маски; один ambiguous без маски."""
    img = np.full((80, 120), 255, dtype=np.uint8)
    img[40, :] = 0
    yy, xx = np.ogrid[:80, :120]
    ring = np.abs(np.hypot(xx - 30, yy - 40) - 10) < 0.8
    img[ring] = 0
    mask = ring[28:53, 18:43]
    rle = rle_encode(mask, origin=(18, 28))
    drawing = Drawing(
        kind="raster",
        width=120.0,
        height=80.0,
        units_per_px=1.0,
        image=img,
        source_path="img.png",
    )
    insts = [
        Instance(
            legend_id="ring",
            T=Similarity.identity(),
            score=0.9,
            coverage=0.9,
            piece_ids=[],
            parent_ids=[],
            bbox=(18, 28, 43, 53),
            mask_rle=rle,
            id=0,
        ),
        Instance(
            legend_id="other",
            T=Similarity.identity(),
            score=0.5,
            coverage=0.5,
            piece_ids=[],
            parent_ids=[],
            bbox=(70, 20, 100, 60),
            mask_rle=None,
            status="ambiguous",
            id=1,
        ),
    ]
    result = Result(
        drawing="img.png",
        instances=insts,
        timings={"raster": 0.5},
        config={},
        kind="raster",
        width=120.0,
        height=80.0,
    )
    return drawing, result
