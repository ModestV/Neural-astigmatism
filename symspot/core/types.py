"""Контракты SymSpot. Меняет только главная сессия.

Соглашения:
- координаты в единицах чертежа, ось Y вниз (как в PDF/SVG после чтения);
- рамка ``BBox`` = (x0, y0, x1, y1), x0 <= x1, y0 <= y1;
- дуга: точка(t) = center + r * (cos t, sin t) в координатах чертежа, t идет от t0 к t1,
  t0 < t1, t1 - t0 <= 2π; полный круг: t1 - t0 = 2π. Из-за оси Y вниз на экране это
  обход по часовой стрелке, но формулы те же, что в обычной математике;
- преобразование подобия ``Similarity``: p' = s * R(theta) * M * p + (tx, ty),
  где M = diag(1, -1) при mirror (отражение по Y до поворота), иначе единичная.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Point = tuple[float, float]
BBox = tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Геометрия
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Style:
    """Стиль линии из исходного файла."""

    stroke_width: float = 0.0
    color: tuple[float, float, float] | None = None  # RGB 0..1, None = не задан


DEFAULT_STYLE = Style()


@dataclass(frozen=True)
class Segment:
    """Отрезок. ``parent_id`` = исходный путь файла, ``stroke_id`` = примитив до планаризации."""

    id: int
    p0: Point
    p1: Point
    parent_id: int
    style: Style = DEFAULT_STYLE
    stroke_id: int = -1


@dataclass(frozen=True)
class Arc:
    """Дуга окружности (см. соглашение об углах в шапке модуля)."""

    id: int
    center: Point
    r: float
    t0: float
    t1: float
    parent_id: int
    style: Style = DEFAULT_STYLE
    stroke_id: int = -1

    @property
    def is_full(self) -> bool:
        return self.t1 - self.t0 >= 2 * math.pi - 1e-9


@dataclass(frozen=True)
class TextPrim:
    """Текст: строка и рамка (выровненная по осям) в единицах чертежа."""

    id: int
    text: str
    bbox: BBox
    parent_id: int
    stroke_id: int = -1
    angle: float = 0.0  # направление базовой линии текста, радианы


Primitive = Segment | Arc | TextPrim


# ---------------------------------------------------------------------------
# Исходные пути файла
# ---------------------------------------------------------------------------

PathItemKind = Literal["line", "cubic", "quad", "text"]


@dataclass(frozen=True)
class PathItem:
    """Элемент исходного пути в координатах чертежа.

    line: 2 точки; quad: 3 точки (p0, c, p1); cubic: 4 точки (p0, c1, c2, p1);
    text: 0 точек (геометрия в TextPrim).
    Прямоугольники и эллипсы ридеры раскладывают на line/cubic.
    """

    kind: PathItemKind
    points: tuple[Point, ...]


@dataclass
class ParentPath:
    """Один связный подпуть исходного файла (разбиение по moveto).

    ``polyline`` — сэмплированная полилиния всех items (N, 2) для отладки и грубых оценок.
    ``group`` — индекс исходного объекта файла (drawing в PDF, элемент в SVG), к которому
    относится подпуть; ``source_ref`` — человекочитаемая ссылка (``svg:#id``, ``pdf:12``).
    """

    id: int
    items: list[PathItem]
    polyline: np.ndarray
    style: Style = DEFAULT_STYLE
    closed: bool = False
    group: int = -1
    source_ref: str = ""
    fill: bool = False

    @property
    def bbox(self) -> BBox:
        if self.polyline.size == 0:
            return (0.0, 0.0, 0.0, 0.0)
        mn = self.polyline.min(axis=0)
        mx = self.polyline.max(axis=0)
        return (float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1]))


@dataclass
class Drawing:
    """Чертеж.

    vector: после ридера ``parents`` заполнены, ``primitives`` содержит только TextPrim;
    после нормализации ``primitives`` = куски планаризации (Segment/Arc) + TextPrim,
    ``normalized=True``.
    raster: ``image`` — grayscale uint8 (H, W), единицы = пиксели, ``primitives`` пуст.
    ``units_per_px`` — сколько единиц чертежа в одном пикселе растеризации по умолчанию
    (для PDF при 72 dpi = 1.0; для растра 1.0).
    """

    kind: Literal["vector", "raster"]
    width: float
    height: float
    units_per_px: float
    primitives: list[Primitive] = field(default_factory=list)
    parents: dict[int, ParentPath] = field(default_factory=dict)
    image: np.ndarray | None = None
    source_path: str = ""
    page: int = 0
    normalized: bool = False

    def prim_by_id(self) -> dict[int, Primitive]:
        return {p.id: p for p in self.primitives}


# ---------------------------------------------------------------------------
# Преобразования
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Similarity:
    """p' = s * R(theta) * M * p + t, M = diag(1, -1) при mirror."""

    s: float = 1.0
    theta: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    mirror: bool = False

    @staticmethod
    def identity() -> Similarity:
        return Similarity()

    def linear(self) -> np.ndarray:
        """Матрица 2x2 линейной части: s * R(theta) * M."""
        c, si = math.cos(self.theta), math.sin(self.theta)
        R = np.array([[c, -si], [si, c]], dtype=float)
        if self.mirror:
            R = R @ np.array([[1.0, 0.0], [0.0, -1.0]])
        return self.s * R

    def matrix(self) -> np.ndarray:
        """Однородная матрица 3x3."""
        m = np.eye(3)
        m[:2, :2] = self.linear()
        m[0, 2] = self.tx
        m[1, 2] = self.ty
        return m

    def apply(self, pts: np.ndarray) -> np.ndarray:
        """Применить к точкам (N, 2) или (2,)."""
        p = np.asarray(pts, dtype=float)
        return p @ self.linear().T + np.array([self.tx, self.ty])

    def apply_vec(self, v: np.ndarray) -> np.ndarray:
        """Применить линейную часть к векторам (N, 2) без сдвига и без масштаба."""
        L = self.linear() / self.s
        return np.asarray(v, dtype=float) @ L.T

    def apply_angle(self, t: float) -> float:
        """Направление (радианы) после преобразования."""
        return (-t if self.mirror else t) + self.theta

    def compose(self, other: Similarity) -> Similarity:
        """self ∘ other: сначала other, потом self."""
        s = self.s * other.s
        if self.mirror:
            theta = self.theta - other.theta
        else:
            theta = self.theta + other.theta
        mirror = self.mirror != other.mirror
        t = self.apply(np.array([other.tx, other.ty]))
        return Similarity(s, _wrap(theta), float(t[0]), float(t[1]), mirror)

    def inverse(self) -> Similarity:
        s = 1.0 / self.s
        theta = self.theta if self.mirror else -self.theta
        inv_lin = Similarity(s, theta, 0.0, 0.0, self.mirror)
        t = -inv_lin.apply(np.array([self.tx, self.ty]))
        return Similarity(s, _wrap(theta), float(t[0]), float(t[1]), self.mirror)

    def to_dict(self) -> dict:
        return {
            "s": self.s,
            "theta": self.theta,
            "tx": self.tx,
            "ty": self.ty,
            "mirror": self.mirror,
        }

    @staticmethod
    def from_dict(d: dict) -> Similarity:
        return Similarity(
            float(d["s"]),
            float(d["theta"]),
            float(d["tx"]),
            float(d["ty"]),
            bool(d.get("mirror", False)),
        )


def _wrap(a: float) -> float:
    """Угол в [-π, π)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Эталоны, гипотезы, экземпляры
# ---------------------------------------------------------------------------


@dataclass
class QuerySpec:
    """Эталон, размеченный рамкой на чертеже (в единицах чертежа)."""

    legend_id: str
    bbox: BBox
    label: str | None = None


@dataclass
class Query:
    """Нормализованный эталон.

    ``prims`` в нормализованных координатах: центр рамки -> 0, диагональ рамки -> 1.
    ``frame`` переводит нормализованные координаты в координаты чертежа (место эталона).
    ``samples`` (N, 2), ``tangents`` (N, 2, единичные, неориентированные), ``weights`` (N,),
    ``sample_prim`` (N,) — индекс примитива в ``prims`` для каждого сэмпла.
    ``diag`` — диагональ рамки в единицах чертежа. ``variant`` — 0 для исходного эталона,
    >0 для эталонов из расширения запроса.
    """

    legend_id: str
    prims: list[Primitive]
    frame: Similarity
    samples: np.ndarray
    tangents: np.ndarray
    weights: np.ndarray
    sample_prim: np.ndarray
    diag: float
    label: str | None = None
    piece_ids: list[int] = field(default_factory=list)  # куски чертежа, из которых взят эталон
    variant: int = 0
    box: BBox | None = None  # рамка эталона в нормализованных координатах

    @property
    def texts(self) -> list[str]:
        return [p.text for p in self.prims if isinstance(p, TextPrim)]

    @property
    def complexity(self) -> int:
        return int(len(self.samples))


@dataclass
class Hypothesis:
    """Кандидат: T переводит нормализованный эталон в координаты чертежа."""

    legend_id: str
    T: Similarity
    votes: float
    source: Literal["exact", "vote", "expand", "chamfer", "text"]
    variant: int = 0


@dataclass
class Instance:
    """Найденный экземпляр символа."""

    legend_id: str
    T: Similarity
    score: float
    coverage: float
    piece_ids: list[int]
    parent_ids: list[int]
    bbox: BBox
    mask_rle: dict | None = None
    status: Literal["ok", "ambiguous"] = "ok"
    id: int = -1
    variant: int = 0
    source: str = "vote"
    prim_coverage: list[float] = field(default_factory=list)
    residual: float = 0.0  # средний остаток ICP в долях диагонали эталона
    rivals: list[str] = field(default_factory=list)  # legend_id конкурентов при ambiguous
    is_query: bool = False  # экземпляр совпадает с самим эталоном

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "legend_id": self.legend_id,
            "T": self.T.to_dict(),
            "score": self.score,
            "coverage": self.coverage,
            "piece_ids": list(self.piece_ids),
            "parent_ids": list(self.parent_ids),
            "bbox": list(self.bbox),
            "mask_rle": self.mask_rle,
            "status": self.status,
            "variant": self.variant,
            "source": self.source,
            "prim_coverage": list(self.prim_coverage),
            "residual": self.residual,
            "rivals": list(self.rivals),
            "is_query": self.is_query,
        }

    @staticmethod
    def from_dict(d: dict) -> Instance:
        return Instance(
            legend_id=str(d["legend_id"]),
            T=Similarity.from_dict(d["T"]),
            score=float(d["score"]),
            coverage=float(d["coverage"]),
            piece_ids=[int(x) for x in d.get("piece_ids", [])],
            parent_ids=[int(x) for x in d.get("parent_ids", [])],
            bbox=tuple(float(x) for x in d["bbox"]),  # type: ignore[arg-type]
            mask_rle=d.get("mask_rle"),
            status=d.get("status", "ok"),
            id=int(d.get("id", -1)),
            variant=int(d.get("variant", 0)),
            source=str(d.get("source", "vote")),
            prim_coverage=[float(x) for x in d.get("prim_coverage", [])],
            residual=float(d.get("residual", 0.0)),
            rivals=[str(x) for x in d.get("rivals", [])],
            is_query=bool(d.get("is_query", False)),
        )


@dataclass
class GroundTruth:
    """Эталонная разметка экземпляра (из COCO ``object`` или GT синтетики)."""

    legend_id: str
    bbox: BBox
    parent_ids: list[int] = field(default_factory=list)
    is_query: bool = False


@dataclass
class Result:
    """Результат прогона на одном чертеже."""

    drawing: str
    instances: list[Instance]
    timings: dict[str, float]
    config: dict
    queries: list[QuerySpec] = field(default_factory=list)
    kind: Literal["vector", "raster"] = "vector"
    width: float = 0.0
    height: float = 0.0
    page: int = 0

    def to_dict(self) -> dict:
        return {
            "drawing": self.drawing,
            "kind": self.kind,
            "width": self.width,
            "height": self.height,
            "page": self.page,
            "queries": [
                {"legend_id": q.legend_id, "bbox": list(q.bbox), "label": q.label}
                for q in self.queries
            ],
            "instances": [i.to_dict() for i in self.instances],
            "timings": dict(self.timings),
            "config": dict(self.config),
        }

    @staticmethod
    def from_dict(d: dict) -> Result:
        return Result(
            drawing=str(d["drawing"]),
            instances=[Instance.from_dict(x) for x in d.get("instances", [])],
            timings=dict(d.get("timings", {})),
            config=dict(d.get("config", {})),
            queries=[
                QuerySpec(str(q["legend_id"]), tuple(q["bbox"]), q.get("label"))  # type: ignore[arg-type]
                for q in d.get("queries", [])
            ],
            kind=d.get("kind", "vector"),
            width=float(d.get("width", 0.0)),
            height=float(d.get("height", 0.0)),
            page=int(d.get("page", 0)),
        )
