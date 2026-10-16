"""Общие помощники рендеринга: id экземпляров, куски, стили, дуги."""

from __future__ import annotations

import math
from collections.abc import Iterator

from symspot.core.types import Arc, Drawing, Instance, Point, Primitive, Result, Segment, TextPrim

# Фон: исходные линии серым.
BG_OPACITY = 0.35
BG_MIN_WIDTH = 0.3
# Экземпляры: толщина = множитель * исходная, но не меньше минимума.
INST_WIDTH_FACTOR = 2.0
INST_MIN_WIDTH = 0.8
# Прозрачность заливки рамки текста экземпляра и масок растра (0..1 / 0..255).
TEXT_FILL_OPACITY = 0.25
MASK_ALPHA = 170
# Пунктир ambiguous в единицах толщины линии.
DASH_ON = 4.0
DASH_OFF = 3.0
# Дуга, у которой не хватает до полного круга меньше этого, рисуется кругом.
FULL_ARC_EPS = 1e-4
# Максимальный угол одного отрезка Безье при аппроксимации дуги.
BEZIER_MAX_ANGLE = math.pi / 2
# Осветление фона растра: out = k * img + b.
RASTER_LIGHTEN_K = 0.5
RASTER_LIGHTEN_B = 127


def fmt(v: float, digits: int = 4) -> str:
    """Короткая запись числа для SVG."""
    s = f"{v:.{digits}f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def instance_ids(result: Result) -> list[int]:
    """Уникальные id экземпляров: свой ``id``, если он >= 0 и не повторяется, иначе новый."""
    used: set[int] = set()
    out: list[int] = []
    pending: list[int] = []
    for k, inst in enumerate(result.instances):
        if inst.id >= 0 and inst.id not in used:
            used.add(inst.id)
            out.append(inst.id)
        else:
            out.append(-1)
            pending.append(k)
    nxt = (max(used) + 1) if used else 0
    for k in pending:
        out[k] = nxt
        nxt += 1
    return out


def size_of(drawing: Drawing) -> tuple[float, float]:
    """Размер чертежа (для растра точно по изображению)."""
    if drawing.kind == "raster" and drawing.image is not None:
        h, w = drawing.image.shape[:2]
        return float(w), float(h)
    return float(drawing.width), float(drawing.height)


def inst_pieces(drawing_prims: dict[int, Primitive], inst: Instance) -> Iterator[Primitive]:
    """Куски экземпляра, найденные в чертеже (неизвестные id пропускаются)."""
    for pid in inst.piece_ids:
        p = drawing_prims.get(pid)
        if p is not None:
            yield p


def piece_width(p: Primitive) -> float:
    """Исходная толщина линии куска (0, если не задана)."""
    if isinstance(p, Segment | Arc):
        return float(p.style.stroke_width)
    return 0.0


def inst_width(p: Primitive) -> float:
    """Толщина цветной линии куска экземпляра."""
    return max(INST_MIN_WIDTH, INST_WIDTH_FACTOR * piece_width(p))


def arc_is_circle(a: Arc) -> bool:
    return (a.t1 - a.t0) >= 2 * math.pi - FULL_ARC_EPS


def arc_point(a: Arc, t: float) -> Point:
    return (a.center[0] + a.r * math.cos(t), a.center[1] + a.r * math.sin(t))


def arc_beziers(a: Arc) -> list[tuple[Point, Point, Point, Point]]:
    """Дуга как набор кубических Безье (каждая не длиннее ``BEZIER_MAX_ANGLE``)."""
    total = min(a.t1 - a.t0, 2 * math.pi)
    n = max(1, math.ceil(total / BEZIER_MAX_ANGLE - 1e-9))
    step = total / n
    k = 4.0 / 3.0 * math.tan(step / 4.0)
    out = []
    for i in range(n):
        ta, tb = a.t0 + i * step, a.t0 + (i + 1) * step
        p0, p3 = arc_point(a, ta), arc_point(a, tb)
        p1 = (p0[0] - k * a.r * math.sin(ta), p0[1] + k * a.r * math.cos(ta))
        p2 = (p3[0] + k * a.r * math.sin(tb), p3[1] - k * a.r * math.cos(tb))
        out.append((p0, p1, p2, p3))
    return out


def is_text(p: Primitive) -> bool:
    return isinstance(p, TextPrim)
