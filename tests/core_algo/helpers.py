"""Ручные сцены для тестов ядра: символы в локальной системе, копии, стены, штриховка.

Все сцены собираются через ``normalize_drawing`` (как в боевом пайплайне): символ задается
примитивами, окружности уходят в ``ParentPath`` четырьмя кубиками Безье, отрезки линиями.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np

from symspot.core.config import Config
from symspot.core.geometry_utils import (
    prims_bbox,
    transform_prim,
)
from symspot.core.types import (
    Arc,
    BBox,
    Drawing,
    ParentPath,
    PathItem,
    Primitive,
    QuerySpec,
    Segment,
    Similarity,
    TextPrim,
)
from symspot.normalize import normalize_drawing
from symspot.normalize.index import PrimIndex

TWO_PI = 2 * math.pi


# ---------------------------------------------------------------------------
# Символы в локальной системе (радиус круга = 10, центр в нуле)
# ---------------------------------------------------------------------------


def _circle(r: float = 10.0) -> Arc:
    return Arc(0, (0.0, 0.0), r, 0.0, TWO_PI, 0)


def sym_slash() -> list[Primitive]:
    """⊘: круг и диагональ, выходящая за круг (диагональ режется кругом при планаризации)."""
    return [_circle(), Segment(1, (-8.5, -8.5), (8.5, 8.5), 1)]


def sym_xcircle() -> list[Primitive]:
    """Круг с крестом: ⊘ плюс вторая диагональ."""
    return [*sym_slash(), Segment(2, (-8.5, 8.5), (8.5, -8.5), 2)]


def sym_circle() -> list[Primitive]:
    return [_circle()]


def sym_square(a: float = 14.0) -> list[Primitive]:
    pts = [(-a, -a), (a, -a), (a, a), (-a, a)]
    return [Segment(i, pts[i], pts[(i + 1) % 4], i) for i in range(4)]


def sym_tri() -> list[Primitive]:
    """Хиральный символ: разносторонний треугольник с косым отводом (не имеет симметрий)."""
    a, b, c = (-8.0, 8.0), (10.0, 8.0), (0.0, -6.0)
    return [
        Segment(0, a, b, 0),
        Segment(1, b, c, 1),
        Segment(2, c, a, 2),
        Segment(3, c, (6.0, -13.0), 3),
    ]


def sym_text_circle(text: str) -> list[Primitive]:
    """Детектор: круг с текстом внутри."""
    return [_circle(), TextPrim(1, text, (-5.0, -3.0, 5.0, 3.0), 1)]


def sym_duplex() -> list[Primitive]:
    return [
        _circle(),
        Segment(1, (-6.0, -3.0), (6.0, -3.0), 1),
        Segment(2, (-6.0, 3.0), (6.0, 3.0), 2),
    ]


def sym_gfci() -> list[Primitive]:
    """duplex плюс подпись под кругом."""
    return [*sym_duplex(), TextPrim(3, "GFCI", (-6.0, 12.0, 6.0, 18.0), 3)]


def local_bbox(prims: list[Primitive], pad: float = 1.0) -> BBox:
    b = prims_bbox([p for p in prims if not isinstance(p, TextPrim)])
    tb = [p.bbox for p in prims if isinstance(p, TextPrim)]
    xs0 = [b[0]] + [t[0] for t in tb]
    ys0 = [b[1]] + [t[1] for t in tb]
    xs1 = [b[2]] + [t[2] for t in tb]
    ys1 = [b[3]] + [t[3] for t in tb]
    return (min(xs0) - pad, min(ys0) - pad, max(xs1) + pad, max(ys1) + pad)


# ---------------------------------------------------------------------------
# Сцена
# ---------------------------------------------------------------------------


@dataclass
class Placement:
    name: str
    local: list[Primitive]
    S: Similarity  # локальная система символа -> чертеж
    parents: list[int]  # parent_id исходных путей этой копии
    box: BBox  # рамка копии в чертеже (для шаблона: рамка эталона)


@dataclass
class Scene:
    prims: list[Primitive] = field(default_factory=list)  # до нормализации
    placements: list[Placement] = field(default_factory=list)
    next_id: int = 1000
    width: float = 600.0
    height: float = 600.0
    ref_size: float = 30.0
    drawing: Drawing | None = None
    index: PrimIndex | None = None
    cfg: Config = field(default_factory=Config)
    wall_parents: list[int] = field(default_factory=list)

    # --- наполнение ---
    def _fresh(self) -> int:
        self.next_id += 1
        return self.next_id

    def place(
        self,
        name: str,
        local: list[Primitive],
        S: Similarity,
        pad: float = 1.0,
    ) -> Placement:
        parents: list[int] = []
        for p in local:
            q = transform_prim(p, S)
            i = self._fresh()
            q = dataclasses.replace(q, id=i, parent_id=i, stroke_id=i)
            self.prims.append(q)
            parents.append(i)
        lb = local_bbox(local, pad)
        corners = np.array([(lb[0], lb[1]), (lb[2], lb[1]), (lb[2], lb[3]), (lb[0], lb[3])])
        tc = S.apply(corners)
        box = (
            float(tc[:, 0].min()),
            float(tc[:, 1].min()),
            float(tc[:, 0].max()),
            float(tc[:, 1].max()),
        )
        pl = Placement(name, local, S, parents, box)
        self.placements.append(pl)
        return pl

    def add_segment(self, p0, p1) -> int:
        i = self._fresh()
        self.prims.append(Segment(i, tuple(p0), tuple(p1), i, stroke_id=i))
        return i

    def add_wall(self, p0, p1) -> int:
        i = self.add_segment(p0, p1)
        self.wall_parents.append(i)
        return i

    def add_text(self, text: str, bbox: BBox) -> int:
        i = self._fresh()
        self.prims.append(TextPrim(i, text, bbox, i, stroke_id=i))
        return i

    def add_hatch(self, x0, y0, w, h, spacing=1.5, seg_len=8.0) -> list[int]:
        """Штриховка под 45 градусов: много коротких параллельных отрезков."""
        ids = []
        d = seg_len / math.sqrt(2)
        y = y0
        while y < y0 + h:
            x = x0
            while x < x0 + w:
                ids.append(self.add_segment((x, y + d), (x + d, y)))
                x += seg_len * 1.3
            y += spacing
        return ids

    # --- сборка ---
    def build(self, ref_size: float | None = None, cfg: Config | None = None) -> Scene:
        if cfg is not None:
            self.cfg = cfg
        if ref_size is not None:
            self.ref_size = ref_size
        parents: dict[int, ParentPath] = {}
        texts: list[Primitive] = []
        for p in self.prims:
            if isinstance(p, TextPrim):
                texts.append(p)
            elif isinstance(p, Segment):
                parents[p.id] = ParentPath(
                    p.id, [PathItem("line", (p.p0, p.p1))], np.array([p.p0, p.p1], dtype=float)
                )
            else:
                items = _arc_to_cubics(p)
                parents[p.id] = ParentPath(p.id, items, np.zeros((0, 2)))
        d = Drawing(
            kind="vector",
            width=self.width,
            height=self.height,
            units_per_px=1.0,
            primitives=texts,
            parents=parents,
        )
        self.drawing = normalize_drawing(d, self.ref_size, self.cfg)
        self.index = PrimIndex(self.drawing.primitives, self.cfg.sample_step * self.ref_size)
        return self

    # --- удобства ---
    def spec(self, pl: Placement, legend_id: str | None = None, box: BBox | None = None):
        return QuerySpec(legend_id or pl.name, box or pl.box)

    def pieces_of(self, parents) -> list[int]:
        ps = set(parents)
        assert self.drawing is not None
        return [p.id for p in self.drawing.primitives if p.parent_id in ps]

    def wall_pieces(self) -> list[int]:
        return self.pieces_of(self.wall_parents)


def _arc_to_cubics(a: Arc) -> list[PathItem]:
    n = max(1, math.ceil((a.t1 - a.t0) / (math.pi / 2) - 1e-9))
    dt = (a.t1 - a.t0) / n
    k = 4.0 / 3.0 * math.tan(dt / 4.0)
    cx, cy = a.center
    items = []
    for i in range(n):
        t0 = a.t0 + i * dt
        t1 = t0 + dt
        p0 = (cx + a.r * math.cos(t0), cy + a.r * math.sin(t0))
        p3 = (cx + a.r * math.cos(t1), cy + a.r * math.sin(t1))
        c1 = (p0[0] - k * a.r * math.sin(t0), p0[1] + k * a.r * math.cos(t0))
        c2 = (p3[0] + k * a.r * math.sin(t1), p3[1] - k * a.r * math.cos(t1))
        items.append(PathItem("cubic", (p0, c1, c2, p3)))
    return items


def sim(s=1.0, deg=0.0, tx=0.0, ty=0.0, mirror=False) -> Similarity:
    return Similarity(s, math.radians(deg), tx, ty, mirror)


def true_T(query, pl_template: Placement, pl_copy: Placement) -> Similarity:
    """T: нормализованный эталон -> чертеж, для копии ``pl_copy`` (эталон снят с шаблона)."""
    return pl_copy.S.compose(pl_template.S.inverse()).compose(query.frame)


def angle_diff_deg(a: float, b: float) -> float:
    d = (a - b + math.pi) % TWO_PI - math.pi
    return abs(math.degrees(d))


def instance_diag(T: Similarity) -> float:
    """Диагональ рамки эталона в чертеже при преобразовании T (нормализованная диагональ = 1)."""
    return T.s


def make_hyp(legend_id: str, T: Similarity, votes: float = 10.0, source: str = "vote"):
    from symspot.core.types import Hypothesis

    return Hypothesis(legend_id, T, votes, source)  # type: ignore[arg-type]


def hyp_matches(
    hyp, T_true: Similarity, query, *, ang_tol_deg: float = 10.0, symmetric: bool = False
) -> bool:
    """Гипотеза совпадает с истиной: центр ближе 0.25 диагонали, угол ближе 10 градусов.

    Для симметричных символов (``symmetric=True``) вместо сравнения угла и зеркала сравнивается
    геометрия: средний путь от образов сэмплов эталона по гипотезе до образов по истине.
    """
    from scipy.spatial import cKDTree

    diag = instance_diag(T_true)
    c = np.hypot(hyp.T.tx - T_true.tx, hyp.T.ty - T_true.ty)
    if c >= 0.25 * diag:
        return False
    if abs(np.log(hyp.T.s / T_true.s)) > 0.2:
        return False
    if not symmetric:
        return hyp.T.mirror == T_true.mirror and angle_diff_deg(hyp.T.theta, T_true.theta) < (
            ang_tol_deg
        )
    a = hyp.T.apply(query.samples)
    b = T_true.apply(query.samples)
    d, _ = cKDTree(b).query(a)
    return float(np.mean(d)) < 0.05 * diag
