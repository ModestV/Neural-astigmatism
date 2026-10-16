"""Безье -> полилинии, распознавание окружностей и дуг. Задача T5."""

from __future__ import annotations

import math

import numpy as np

from symspot.core.types import Arc, ParentPath, PathItem, Segment, Style

_TWO_PI = 2.0 * math.pi
_MAX_DEPTH = 18
_CURVE_SAMPLES = 8  # интервалов на одну кривую при фите окружности
_GN_ITERS = 10
_MAX_R_FACTOR = 1e3  # радиус больше этого множителя размера цепочки считается прямой
_DEGENERATE = 1e-12


# ---------------------------------------------------------------------------
# Безье
# ---------------------------------------------------------------------------


def _dist_point_seg(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx = bx - ax
    dy = by - ay
    ll = dx * dx + dy * dy
    if ll < 1e-30:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / ll
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def flatten_bezier(ctrl: np.ndarray, tol: float) -> np.ndarray:
    """Квадратичная (3 точки) или кубическая (4 точки) Безье -> полилиния (K, 2), K >= 2.

    Отклонение полилинии от кривой не больше ``tol`` (адаптивное деление).
    Первая и последняя точки совпадают с концами кривой.
    """
    c = np.asarray(ctrl, dtype=float)
    if c.ndim != 2 or c.shape[1] != 2 or len(c) not in (3, 4):
        raise ValueError(f"flatten_bezier: ожидалось 3 или 4 точки (N, 2), получено {c.shape}")
    if tol <= 0:
        raise ValueError("flatten_bezier: tol должен быть > 0")
    if len(c) == 3:
        p0, q, p3 = c
        c1 = p0 + (2.0 / 3.0) * (q - p0)
        c2 = p3 + (2.0 / 3.0) * (q - p3)
    else:
        p0, c1, c2, p3 = c
    start = (float(p0[0]), float(p0[1]))
    out: list[tuple[float, float]] = [start]
    stack = [
        (
            start,
            (float(c1[0]), float(c1[1])),
            (float(c2[0]), float(c2[1])),
            (float(p3[0]), float(p3[1])),
            0,
        )
    ]
    while stack:
        a, b, cc, d, depth = stack.pop()
        flat = (
            _dist_point_seg(b[0], b[1], a[0], a[1], d[0], d[1]) <= tol
            and _dist_point_seg(cc[0], cc[1], a[0], a[1], d[0], d[1]) <= tol
        )
        if flat or depth >= _MAX_DEPTH:
            out.append(d)
            continue
        ab = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)
        bc = ((b[0] + cc[0]) * 0.5, (b[1] + cc[1]) * 0.5)
        cd = ((cc[0] + d[0]) * 0.5, (cc[1] + d[1]) * 0.5)
        abc = ((ab[0] + bc[0]) * 0.5, (ab[1] + bc[1]) * 0.5)
        bcd = ((bc[0] + cd[0]) * 0.5, (bc[1] + cd[1]) * 0.5)
        mid = ((abc[0] + bcd[0]) * 0.5, (abc[1] + bcd[1]) * 0.5)
        stack.append((mid, bcd, cd, d, depth + 1))
        stack.append((a, ab, abc, mid, depth + 1))
    return np.array(out, dtype=float)


def _sample_curve(ctrl: np.ndarray, n: int) -> np.ndarray:
    """n + 1 точек кривой Безье (включая концы), (n + 1, 2)."""
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    u = 1.0 - t
    if len(ctrl) == 3:
        return u * u * ctrl[0] + 2.0 * u * t * ctrl[1] + t * t * ctrl[2]
    return u**3 * ctrl[0] + 3.0 * u * u * t * ctrl[1] + 3.0 * u * t * t * ctrl[2] + t**3 * ctrl[3]


# ---------------------------------------------------------------------------
# Окружность
# ---------------------------------------------------------------------------

_INF_CIRCLE = ((math.nan, math.nan), math.inf, math.inf)


def fit_circle(points: np.ndarray) -> tuple[tuple[float, float], float, float]:
    """Фит окружности по точкам (N >= 3): алгебраический (Kasa) + уточнение Гаусса-Ньютона.

    Возвращает (центр, радиус, rms остатка расстояний). Для коллинеарных точек
    rms = inf, радиус = inf.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"fit_circle: ожидался массив (N, 2), получено {pts.shape}")
    if len(pts) < 3:
        return _INF_CIRCLE
    mu = pts.mean(axis=0)
    q = pts - mu
    sv = np.linalg.svd(q, compute_uv=False)
    if sv[0] < _DEGENERATE or sv[1] < 1e-9 * sv[0]:
        return _INF_CIRCLE
    scale = float(sv[0]) / math.sqrt(len(pts))
    q = q / scale
    # Kasa: x^2 + y^2 = 2 a x + 2 b y + c
    A = np.column_stack([2.0 * q[:, 0], 2.0 * q[:, 1], np.ones(len(q))])
    rhs = (q**2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    cx, cy = float(sol[0]), float(sol[1])
    r2 = float(sol[2]) + cx * cx + cy * cy
    if not np.isfinite(r2) or r2 <= 0:
        return _INF_CIRCLE
    r = math.sqrt(r2)
    for _ in range(_GN_ITERS):
        d = q - np.array([cx, cy])
        dist = np.hypot(d[:, 0], d[:, 1])
        if dist.min() < 1e-12:
            break
        res = dist - r
        J = np.column_stack([-d[:, 0] / dist, -d[:, 1] / dist, -np.ones(len(q))])
        step, *_ = np.linalg.lstsq(J, -res, rcond=None)
        cx += float(step[0])
        cy += float(step[1])
        r += float(step[2])
        if float(np.abs(step).max()) < 1e-13:
            break
    if not np.isfinite(r) or r <= 0:
        return _INF_CIRCLE
    dist = np.hypot(q[:, 0] - cx, q[:, 1] - cy)
    rms = float(np.sqrt(np.mean((dist - r) ** 2)))
    center = (float(mu[0] + cx * scale), float(mu[1] + cy * scale))
    return center, r * scale, rms * scale


# ---------------------------------------------------------------------------
# Пути -> штрихи
# ---------------------------------------------------------------------------


def _curve_ctrl(item: PathItem) -> np.ndarray | None:
    n = 3 if item.kind == "quad" else 4
    if len(item.points) != n:
        return None
    return np.asarray(item.points, dtype=float)


def _chain_to_arc(
    ctrls: list[np.ndarray], arc_fit_tol: float
) -> tuple[tuple[float, float], float, float, float] | None:
    """Цепочка кривых -> (центр, r, t0, t1), если она дуга/окружность, иначе None."""
    parts: list[np.ndarray] = []
    for k, c in enumerate(ctrls):
        s = _sample_curve(c, _CURVE_SAMPLES)
        parts.append(s if k == 0 else s[1:])
    pts = np.concatenate(parts)
    if len(pts) < 3:
        return None
    center, r, rms = fit_circle(pts)
    if not (math.isfinite(r) and math.isfinite(rms)) or rms >= arc_fit_tol:
        return None
    extent = float(np.hypot(*(pts.max(axis=0) - pts.min(axis=0))))
    if extent < _DEGENERATE or r >= _MAX_R_FACTOR * extent:
        return None
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    unwrapped = np.unwrap(ang)
    total = float(unwrapped[-1] - unwrapped[0])
    closed = float(np.hypot(*(pts[-1] - pts[0]))) <= arc_fit_tol
    abs_total = abs(total)
    if closed and (_TWO_PI - abs_total) * r <= arc_fit_tol and abs_total > math.pi:
        return center, r, 0.0, _TWO_PI
    if abs_total >= _TWO_PI or abs_total * r < _DEGENERATE:
        return None
    start = float(unwrapped[0])
    t0 = start if total > 0 else start + total
    t1 = t0 + abs_total
    shift = math.floor(t0 / _TWO_PI) * _TWO_PI
    return center, r, t0 - shift, t1 - shift


def parents_to_strokes(
    parents: dict[int, ParentPath],
    flatten_tol: float,
    arc_fit_tol: float,
    start_id: int = 0,
) -> list[Segment | Arc]:
    """Исходные пути -> штрихи (Segment/Arc), id подряд с ``start_id``, ``stroke_id`` = id.

    - ``line`` -> Segment.
    - Подряд идущие кривые (quad/cubic) пути группируются в цепочки; цепочка, сэмплы
      которой лежат на окружности с абсолютным rms < ``arc_fit_tol`` и конечным радиусом,
      -> один Arc (полный круг, если цепочка замкнута и покрывает 2π, иначе дуга с
      правильными t0 < t1 по соглашению ``core/types.py``).
      Иначе каждая кривая -> полилиния ``flatten_bezier`` -> Segment'ы.
    - Отдельная кривая, которая сама по себе дуга (фит < tol), -> Arc.
    - ``text`` items и пути с пустой геометрией пропускаются. Вырожденные (нулевой длины)
      отрезки отбрасываются. ``parent_id`` и ``style`` берутся из родителя.
    """
    out: list[Segment | Arc] = []
    next_id = start_id

    for pid in sorted(parents):
        parent = parents[pid]
        style = parent.style

        def emit_segment(
            p0: tuple[float, float], p1: tuple[float, float], pid: int = pid, style: Style = style
        ) -> None:
            nonlocal next_id
            if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < _DEGENERATE:
                return
            out.append(Segment(next_id, p0, p1, pid, style, next_id))
            next_id += 1

        def emit_arc(
            arc: tuple[tuple[float, float], float, float, float],
            pid: int = pid,
            style: Style = style,
        ) -> None:
            nonlocal next_id
            center, r, t0, t1 = arc
            out.append(Arc(next_id, center, r, t0, t1, pid, style, next_id))
            next_id += 1

        def emit_polyline(ctrl: np.ndarray) -> None:
            poly = flatten_bezier(ctrl, flatten_tol)
            for k in range(len(poly) - 1):
                emit_segment(
                    (float(poly[k, 0]), float(poly[k, 1])),
                    (float(poly[k + 1, 0]), float(poly[k + 1, 1])),
                )

        def flush(chain: list[np.ndarray]) -> None:
            if not chain:
                return
            arc = _chain_to_arc(chain, arc_fit_tol)
            if arc is not None:
                emit_arc(arc)
                return
            for ctrl in chain:
                single = _chain_to_arc([ctrl], arc_fit_tol) if len(chain) > 1 else None
                if single is not None:
                    emit_arc(single)
                else:
                    emit_polyline(ctrl)

        chain: list[np.ndarray] = []
        for item in parent.items:
            if item.kind == "line":
                flush(chain)
                chain = []
                if len(item.points) == 2:
                    p0, p1 = item.points
                    emit_segment((float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])))
            elif item.kind in ("quad", "cubic"):
                ctrl = _curve_ctrl(item)
                if ctrl is None:
                    flush(chain)
                    chain = []
                    continue
                if chain and float(np.hypot(*(ctrl[0] - chain[-1][-1]))) > arc_fit_tol:
                    flush(chain)
                    chain = []
                chain.append(ctrl)
            else:  # text и неизвестное
                flush(chain)
                chain = []
        flush(chain)
    return out
