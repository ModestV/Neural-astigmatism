"""Общая геометрия: Umeyama, трансформация и сэмплинг примитивов, рамки."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Sequence

import numpy as np

from symspot.core.types import Arc, BBox, Primitive, Segment, Similarity, TextPrim

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Углы
# ---------------------------------------------------------------------------


def wrap_angle(a: float | np.ndarray) -> float | np.ndarray:
    """Угол в [-π, π)."""
    if isinstance(a, np.ndarray):
        return (a + math.pi) % TWO_PI - math.pi
    return (a + math.pi) % TWO_PI - math.pi


def undirected_angle_diff(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    """Разница неориентированных направлений, в [0, π/2]."""
    d = np.abs((np.asarray(a) - np.asarray(b)) % math.pi)
    d = np.minimum(d, math.pi - d)
    return float(d) if np.ndim(d) == 0 else d


# ---------------------------------------------------------------------------
# Umeyama
# ---------------------------------------------------------------------------


def umeyama(
    src: np.ndarray,
    dst: np.ndarray,
    allow_mirror: bool = False,
    weights: np.ndarray | None = None,
) -> Similarity:
    """Подобие T, минимизирующее Σ w |T(src) - dst|². src, dst: (N, 2), N >= 2.

    При ``allow_mirror`` выбирается лучшее из собственного и несобственного вращения.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError(f"umeyama: ожидались массивы (N, 2) одной формы, {src.shape} {dst.shape}")
    if len(src) < 2:
        raise ValueError("umeyama: нужно минимум 2 точки")
    w = np.ones(len(src)) if weights is None else np.asarray(weights, dtype=float)
    W = w.sum()
    if W <= 0:
        raise ValueError("umeyama: сумма весов должна быть > 0")
    mu_s = (w[:, None] * src).sum(0) / W
    mu_d = (w[:, None] * dst).sum(0) / W
    xs = src - mu_s
    xd = dst - mu_d
    var_s = (w * (xs**2).sum(1)).sum() / W
    if var_s <= 1e-24:
        raise ValueError("umeyama: вырожденный src (все точки совпадают)")
    cov = (w[:, None, None] * (xd[:, :, None] * xs[:, None, :])).sum(0) / W
    U, D, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    S = np.eye(2)
    if not allow_mirror and d < 0:
        S[1, 1] = -1.0
    R = U @ S @ Vt
    scale = float((D * np.diag(S)).sum() / var_s)
    mirror = bool(np.linalg.det(R) < 0)
    R0 = R @ np.diag([1.0, -1.0]) if mirror else R
    theta = math.atan2(R0[1, 0], R0[0, 0])
    t = mu_d - scale * (R @ mu_s)
    return Similarity(scale, theta, float(t[0]), float(t[1]), mirror)


def similarity_from_two_points(
    a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray, mirror: bool = False
) -> Similarity | None:
    """Подобие, переводящее a0->b0, a1->b1 (с отражением по Y до поворота при mirror)."""
    va = np.asarray(a1, float) - np.asarray(a0, float)
    vb = np.asarray(b1, float) - np.asarray(b0, float)
    la = math.hypot(va[0], va[1])
    lb = math.hypot(vb[0], vb[1])
    if la < 1e-12 or lb < 1e-12:
        return None
    if mirror:
        va = np.array([va[0], -va[1]])
    s = lb / la
    theta = math.atan2(vb[1], vb[0]) - math.atan2(va[1], va[0])
    T0 = Similarity(s, float(wrap_angle(theta)), 0.0, 0.0, mirror)
    p = T0.apply(np.asarray(a0, float))
    return Similarity(s, T0.theta, float(b0[0] - p[0]), float(b0[1] - p[1]), mirror)


# ---------------------------------------------------------------------------
# Примитивы
# ---------------------------------------------------------------------------


def arc_point(arc: Arc, t: float | np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return np.stack([arc.center[0] + arc.r * np.cos(t), arc.center[1] + arc.r * np.sin(t)], axis=-1)


def prim_length(p: Primitive) -> float:
    if isinstance(p, Segment):
        return math.hypot(p.p1[0] - p.p0[0], p.p1[1] - p.p0[1])
    if isinstance(p, Arc):
        return p.r * (p.t1 - p.t0)
    return 0.0


def prim_bbox(p: Primitive) -> BBox:
    if isinstance(p, Segment):
        return (
            min(p.p0[0], p.p1[0]),
            min(p.p0[1], p.p1[1]),
            max(p.p0[0], p.p1[0]),
            max(p.p0[1], p.p1[1]),
        )
    if isinstance(p, Arc):
        if p.is_full:
            cx, cy = p.center
            return (cx - p.r, cy - p.r, cx + p.r, cy + p.r)
        ts = [p.t0, p.t1]
        k0 = math.ceil(p.t0 / (math.pi / 2))
        k1 = math.floor(p.t1 / (math.pi / 2))
        ts.extend(k * math.pi / 2 for k in range(k0, k1 + 1))
        pts = arc_point(p, np.array(ts))
        return (
            float(pts[:, 0].min()),
            float(pts[:, 1].min()),
            float(pts[:, 0].max()),
            float(pts[:, 1].max()),
        )
    return p.bbox


def prims_bbox(prims: Iterable[Primitive]) -> BBox | None:
    boxes = [prim_bbox(p) for p in prims]
    if not boxes:
        return None
    a = np.array(boxes)
    return (float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 2].max()), float(a[:, 3].max()))


def transform_prim(p: Primitive, T: Similarity) -> Primitive:
    """Применить подобие к примитиву (id и родители сохраняются)."""
    if isinstance(p, Segment):
        q = T.apply(np.array([p.p0, p.p1]))
        return dataclasses.replace(
            p, p0=(float(q[0, 0]), float(q[0, 1])), p1=(float(q[1, 0]), float(q[1, 1]))
        )
    if isinstance(p, Arc):
        c = T.apply(np.array(p.center))
        if p.is_full:
            t0, t1 = 0.0, TWO_PI
        elif T.mirror:
            t0, t1 = T.theta - p.t1, T.theta - p.t0
        else:
            t0, t1 = p.t0 + T.theta, p.t1 + T.theta
        if not p.is_full:
            shift = math.floor(t0 / TWO_PI) * TWO_PI
            t0, t1 = t0 - shift, t1 - shift
        return dataclasses.replace(p, center=(float(c[0]), float(c[1])), r=p.r * T.s, t0=t0, t1=t1)
    if isinstance(p, TextPrim):
        x0, y0, x1, y1 = p.bbox
        corners = T.apply(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
        mn = corners.min(0)
        mx = corners.max(0)
        return dataclasses.replace(
            p,
            bbox=(float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1])),
            angle=float(wrap_angle(T.apply_angle(p.angle))),
        )
    raise TypeError(f"неизвестный примитив {type(p)}")


def sample_primitive(p: Primitive, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Сэмплы и единичные касательные с шагом ``step`` по длине.

    Сэмплы в серединах n равных интервалов, n = max(1, ceil(L / step)), чтобы стыки
    соседних примитивов не дублировались. Текст сэмплов не дает (проверяется отдельно).
    """
    if step <= 0:
        raise ValueError("step должен быть > 0")
    if isinstance(p, Segment):
        a = np.array(p.p0, float)
        b = np.array(p.p1, float)
        L = float(np.hypot(*(b - a)))
        if L < 1e-12:
            return np.empty((0, 2)), np.empty((0, 2))
        n = max(1, math.ceil(L / step))
        t = (np.arange(n) + 0.5) / n
        pts = a[None, :] + t[:, None] * (b - a)[None, :]
        tan = np.repeat(((b - a) / L)[None, :], n, axis=0)
        return pts, tan
    if isinstance(p, Arc):
        L = p.r * (p.t1 - p.t0)
        if L < 1e-12:
            return np.empty((0, 2)), np.empty((0, 2))
        n = max(1, math.ceil(L / step))
        t = p.t0 + (np.arange(n) + 0.5) / n * (p.t1 - p.t0)
        pts = arc_point(p, t)
        tan = np.stack([-np.sin(t), np.cos(t)], axis=-1)
        return pts, tan
    return np.empty((0, 2)), np.empty((0, 2))


def sample_primitives(
    prims: Sequence[Primitive], step: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Сэмплы всех примитивов: точки (N, 2), касательные (N, 2), индекс примитива (N,)."""
    pts_l: list[np.ndarray] = []
    tan_l: list[np.ndarray] = []
    idx_l: list[np.ndarray] = []
    for i, p in enumerate(prims):
        pts, tan = sample_primitive(p, step)
        if len(pts):
            pts_l.append(pts)
            tan_l.append(tan)
            idx_l.append(np.full(len(pts), i, dtype=np.int64))
    if not pts_l:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0, dtype=np.int64)
    return np.concatenate(pts_l), np.concatenate(tan_l), np.concatenate(idx_l)


def tangent_cos(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """|cos| угла между неориентированными касательными (построчно)."""
    return np.abs((np.asarray(t1) * np.asarray(t2)).sum(-1))


# ---------------------------------------------------------------------------
# Рамки
# ---------------------------------------------------------------------------


def bbox_diag(b: BBox) -> float:
    return math.hypot(b[2] - b[0], b[3] - b[1])


def bbox_center(b: BBox) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def bbox_area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def bbox_iou(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def bbox_expand(b: BBox, pad: float) -> BBox:
    return (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)


def bbox_of_points(pts: np.ndarray) -> BBox:
    mn = pts.min(0)
    mx = pts.max(0)
    return (float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1]))


def transform_bbox(b: BBox, T: Similarity) -> BBox:
    corners = np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
    return bbox_of_points(T.apply(corners))


def normalize_text(s: str) -> str:
    """Строка текста для сравнения: верхний регистр, без пробелов."""
    return "".join(s.split()).upper()
