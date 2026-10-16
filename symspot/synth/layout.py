"""Фон плана: комнаты, стены, двери, штриховки, размерные линии."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from symspot.core.types import BBox

Rect = tuple[float, float, float, float]

ROOM_NAMES = (
    "BEDROOM",
    "KITCHEN",
    "LIVING",
    "BATH",
    "OFFICE",
    "STORAGE",
    "HALL",
    "LAB",
    "CORRIDOR",
    "DINING",
    "UTILITY",
    "GARAGE",
    "LOBBY",
    "MEETING",
)
NOTE_TEXTS = (
    "NOTE 1",
    "TYP.",
    "VERIFY IN FIELD",
    "SEE DETAIL A",
    "N.T.S.",
    "EQ",
    "CLG 2700",
    "A-101",
    "SCALE 1:50",
    "FIN. FLR",
    "REF. SHEET E-2",
    "H=2400",
)


@dataclass
class Wall:
    """Стена: две параллельные линии c ± t/2, протяженность a0..a1, проемы для дверей."""

    orient: str  # "h": линии горизонтальны (y = const); "v": вертикальны
    c: float
    a0: float
    a1: float
    t: float
    interior: bool
    doors: list[tuple[float, float]] = field(default_factory=list)
    inward: int = 1  # для внешних стен: знак нормали, направленной внутрь здания

    def line_pieces(self) -> list[tuple[float, float]]:
        """Куски линий вдоль стены (между проемами)."""
        pts = [self.a0]
        for g0, g1 in sorted(self.doors):
            pts.extend([g0, g1])
        pts.append(self.a1)
        return [(pts[i], pts[i + 1]) for i in range(0, len(pts), 2) if pts[i + 1] > pts[i]]

    def lines(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        out = []
        for off in (-self.t / 2, self.t / 2):
            for p0, p1 in self.line_pieces():
                if self.orient == "h":
                    out.append(((p0, self.c + off), (p1, self.c + off)))
                else:
                    out.append(((self.c + off, p0), (self.c + off, p1)))
        return out


@dataclass
class Door:
    hinge: tuple[float, float]
    u: tuple[float, float]  # единичный вектор вдоль стены от петли к другому косяку
    n: tuple[float, float]  # единичный вектор в сторону распахивания
    w: float

    def leaf(self) -> tuple[tuple[float, float], tuple[float, float]]:
        h = self.hinge
        return h, (h[0] + self.n[0] * self.w, h[1] + self.n[1] * self.w)

    def arc(self) -> tuple[tuple[float, float], float, float, float]:
        """(центр, радиус, t0, t1) четверти круга от полотна до закрытого положения."""
        an = math.atan2(self.n[1], self.n[0])
        au = math.atan2(self.u[1], self.u[0])
        d = (au - an + math.pi) % (2 * math.pi) - math.pi
        t0 = an if d > 0 else au
        t0 %= 2 * math.pi
        return self.hinge, self.w, t0, t0 + math.pi / 2

    def bbox(self) -> BBox:
        h = self.hinge
        pts = [
            h,
            (h[0] + self.u[0] * self.w, h[1] + self.u[1] * self.w),
            (h[0] + self.n[0] * self.w, h[1] + self.n[1] * self.w),
            (
                h[0] + (self.u[0] + self.n[0]) * self.w,
                h[1] + (self.u[1] + self.n[1]) * self.w,
            ),
        ]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class Layout:
    building: Rect
    rooms: list[Rect]
    walls: list[Wall]
    doors: list[Door]


def _partition(
    rect: Rect, rng: np.random.Generator, minr: float, t: float, stop_p: float, walls: list[Wall]
) -> list[Rect]:
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    can_v = w >= 2 * minr + t
    can_h = h >= 2 * minr + t
    if not (can_v or can_h):
        return [rect]
    if max(w, h) < 2.4 * minr and rng.random() < stop_p:
        return [rect]
    if can_v and can_h:
        vertical = rng.random() < w / (w + h)
    else:
        vertical = can_v
    if vertical:
        c = float(rng.uniform(x0 + minr + t / 2, x1 - minr - t / 2))
        walls.append(Wall("v", c, y0, y1, t, True))
        a = (x0, y0, c - t / 2, y1)
        b = (c + t / 2, y0, x1, y1)
    else:
        c = float(rng.uniform(y0 + minr + t / 2, y1 - minr - t / 2))
        walls.append(Wall("h", c, x0, x1, t, True))
        a = (x0, y0, x1, c - t / 2)
        b = (x0, c + t / 2, x1, y1)
    return _partition(a, rng, minr, t, stop_p, walls) + _partition(b, rng, minr, t, stop_p, walls)


def _door_for(wall: Wall, g0: float, w: float, rng: np.random.Generator) -> Door:
    """Дверь в проеме [g0, g0 + w] стены; сторона и петля выбираются случайно/по inward."""
    g1 = g0 + w
    flip = rng.random() < 0.5
    if wall.interior:
        side = 1 if rng.random() < 0.5 else -1
    else:
        side = wall.inward
    start, end = (g1, g0) if flip else (g0, g1)
    sgn = 1.0 if end > start else -1.0
    if wall.orient == "h":
        hinge = (start, wall.c)
        u = (sgn, 0.0)
        n = (0.0, float(side))
    else:
        hinge = (wall.c, start)
        u = (0.0, sgn)
        n = (float(side), 0.0)
    return Door(hinge, u, n, w)


def build_layout(
    rng: np.random.Generator,
    building: Rect,
    *,
    min_room: float,
    stop_p: float,
    t_out: float,
    t_in: float,
) -> Layout:
    x0, y0, x1, y1 = building
    walls: list[Wall] = []
    # внешние стены: горизонтальные на всю ширину, вертикальные между их внутренними линиями
    walls.append(Wall("h", y0 + t_out / 2, x0, x1, t_out, False, inward=1))
    walls.append(Wall("h", y1 - t_out / 2, x0, x1, t_out, False, inward=-1))
    walls.append(Wall("v", x0 + t_out / 2, y0 + t_out, y1 - t_out, t_out, False, inward=1))
    walls.append(Wall("v", x1 - t_out / 2, y0 + t_out, y1 - t_out, t_out, False, inward=-1))
    inner = (x0 + t_out, y0 + t_out, x1 - t_out, y1 - t_out)
    interior: list[Wall] = []
    rooms = _partition(inner, rng, min_room, t_in, stop_p, interior)

    doors: list[Door] = []
    for wall in interior:
        length = wall.a1 - wall.a0
        n_doors = 1 + (1 if length > 2.2 * min_room else 0)
        seg = (length - 12.0) / n_doors
        for k in range(n_doors):
            w = float(rng.uniform(12.0, 16.0))
            lo = wall.a0 + 6.0 + k * seg
            hi = lo + seg - w
            if hi <= lo:
                continue
            g0 = float(rng.uniform(lo, hi))
            wall.doors.append((g0, g0 + w))
            doors.append(_door_for(wall, g0, w, rng))
    for wall in walls:
        if rng.random() < 0.6:
            w = float(rng.uniform(14.0, 18.0))
            lo = wall.a0 + 12.0
            hi = wall.a1 - 12.0 - w
            if hi > lo:
                g0 = float(rng.uniform(lo, hi))
                wall.doors.append((g0, g0 + w))
                doors.append(_door_for(wall, g0, w, rng))
    return Layout(building, rooms, walls + interior, doors)


def hatch_segments(
    rect: Rect, spacing: float, angle: float
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Параллельные линии под углом ``angle`` с шагом ``spacing``, обрезанные по прямоугольнику."""
    x0, y0, x1, y1 = rect
    d = np.array([math.cos(angle), math.sin(angle)])
    nrm = np.array([-d[1], d[0]])
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    proj = corners @ nrm
    out = []
    o = proj.min() + spacing / 2
    eps = 1e-9
    while o < proj.max():
        p = nrm * o
        lo, hi = -math.inf, math.inf
        ok = True
        for axis, (a, b) in enumerate(((x0, x1), (y0, y1))):
            if abs(d[axis]) < eps:
                if not (a <= p[axis] <= b):
                    ok = False
                    break
            else:
                s1 = (a - p[axis]) / d[axis]
                s2 = (b - p[axis]) / d[axis]
                lo = max(lo, min(s1, s2))
                hi = min(hi, max(s1, s2))
        if ok and hi - lo > 0.5:
            out.append(
                (
                    (float(p[0] + d[0] * lo), float(p[1] + d[1] * lo)),
                    (float(p[0] + d[0] * hi), float(p[1] + d[1] * hi)),
                )
            )
        o += spacing
    return out
