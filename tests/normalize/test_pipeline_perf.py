import dataclasses
import math
import time

import numpy as np
import pytest

from symspot.core.config import Config
from symspot.core.types import Arc, Drawing, ParentPath, PathItem, Segment, TextPrim
from symspot.normalize import normalize_drawing
from symspot.normalize.merge import merge_collinear
from symspot.normalize.planarize import planarize

K = 0.5522847498307936
N_SEG = 30_000
N_CIRC = 2_000
FIELD = 2000.0


def _random_prims(seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, FIELD, (N_SEG, 2))
    ang = rng.uniform(0, 2 * math.pi, N_SEG)
    ln = rng.uniform(1, 20, N_SEG)
    q = p + np.stack([ln * np.cos(ang), ln * np.sin(ang)], axis=1)
    prims: list = [
        Segment(
            i, (float(p[i, 0]), float(p[i, 1])), (float(q[i, 0]), float(q[i, 1])), i, stroke_id=i
        )
        for i in range(N_SEG)
    ]
    cen = rng.uniform(0, FIELD, (N_CIRC, 2))
    rad = rng.uniform(1, 10, N_CIRC)
    for k in range(N_CIRC):
        i = N_SEG + k
        prims.append(
            Arc(
                i,
                (float(cen[k, 0]), float(cen[k, 1])),
                float(rad[k]),
                0.0,
                2 * math.pi,
                i,
                stroke_id=i,
            )
        )
    return prims, p, q, cen, rad


@pytest.mark.slow
def test_merge_planarize_30k_under_10s():
    prims, *_ = _random_prims()
    t = time.perf_counter()
    merged = merge_collinear(prims, 1.0, 0.05, 1e-3)
    pieces = planarize(merged, 0.02, 1e-3, start_id=10**6)
    dt = time.perf_counter() - t
    assert dt < 10.0, dt
    assert len(pieces) >= len(merged) * 0.9


@pytest.mark.slow
def test_normalize_drawing_30k_under_10s():
    _, p, q, cen, rad = _random_prims(1)
    parents = {}
    for i in range(N_SEG):
        items = [
            PathItem("line", ((float(p[i, 0]), float(p[i, 1])), (float(q[i, 0]), float(q[i, 1]))))
        ]
        parents[i] = ParentPath(i, items, np.zeros((0, 2)))
    for k in range(N_CIRC):
        cx, cy, r = float(cen[k, 0]), float(cen[k, 1]), float(rad[k])
        pts = [
            [(cx + r, cy), (cx + r, cy + K * r), (cx + K * r, cy + r), (cx, cy + r)],
            [(cx, cy + r), (cx - K * r, cy + r), (cx - r, cy + K * r), (cx - r, cy)],
            [(cx - r, cy), (cx - r, cy - K * r), (cx - K * r, cy - r), (cx, cy - r)],
            [(cx, cy - r), (cx + K * r, cy - r), (cx + r, cy - K * r), (cx + r, cy)],
        ]
        pid = N_SEG + k
        parents[pid] = ParentPath(pid, [PathItem("cubic", tuple(c)) for c in pts], np.zeros((0, 2)))
    text = TextPrim(10**7, "A", (0, 0, 5, 5), 10**7)
    drawing = Drawing("vector", FIELD, FIELD, 1.0, [text], parents)
    t = time.perf_counter()
    out = normalize_drawing(drawing, 10.0, Config())
    dt = time.perf_counter() - t
    assert dt < 10.0, dt
    assert out.normalized
    assert sum(isinstance(x, TextPrim) for x in out.primitives) == 1
    arcs = [x for x in out.primitives if isinstance(x, Arc)]
    assert len(arcs) >= N_CIRC


def test_normalize_drawing_small_end_to_end():
    k = K
    r = 1.0
    circle = [
        PathItem("cubic", ((r, 0), (r, k * r), (k * r, r), (0, r))),
        PathItem("cubic", ((0, r), (-k * r, r), (-r, k * r), (-r, 0))),
        PathItem("cubic", ((-r, 0), (-r, -k * r), (-k * r, -r), (0, -r))),
        PathItem("cubic", ((0, -r), (k * r, -r), (r, -k * r), (r, 0))),
    ]
    parents = {
        0: ParentPath(0, circle, np.zeros((0, 2))),
        1: ParentPath(1, [PathItem("line", ((-5, 0), (5, 0)))], np.zeros((0, 2))),
    }
    d = Drawing("vector", 20, 20, 1.0, [TextPrim(9, "X", (1, 1, 2, 2), 9)], parents)
    out = normalize_drawing(d, 10.0, Config())
    segs = [p for p in out.primitives if isinstance(p, Segment)]
    arcs = [p for p in out.primitives if isinstance(p, Arc)]
    assert len(segs) == 3 and len(arcs) == 2
    assert dataclasses.is_dataclass(out)
    with pytest.raises(ValueError):
        normalize_drawing(Drawing("raster", 1, 1, 1.0), 10.0, Config())
