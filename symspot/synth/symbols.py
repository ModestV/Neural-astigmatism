"""Библиотека синтетических символов. Задача T1."""

from __future__ import annotations

import math
from dataclasses import dataclass

from symspot.core.geometry_utils import bbox_diag, prims_bbox, transform_prim
from symspot.core.types import Arc, Primitive, Segment, Similarity, TextPrim

TWO_PI = 2 * math.pi


@dataclass(frozen=True)
class SymbolDef:
    """Символ в локальной системе: центр (0, 0), размер порядка 1 (диагональ рамки ~1).

    ``prims`` — Segment/Arc/TextPrim (id локальные, parent_id = номер штриха символа);
    ``group`` — имя группы похожих символов или None.
    """

    legend_id: str
    label: str
    prims: tuple[Primitive, ...]
    group: str | None = None

    @property
    def has_text(self) -> bool:
        return any(isinstance(p, TextPrim) for p in self.prims)

    @property
    def diag(self) -> float:
        b = prims_bbox(self.prims)
        assert b is not None
        return bbox_diag(b)


class _Builder:
    def __init__(self) -> None:
        self.prims: list[Primitive] = []

    def _next(self) -> int:
        return len(self.prims)

    def seg(self, p0: tuple[float, float], p1: tuple[float, float]) -> None:
        i = self._next()
        self.prims.append(Segment(id=i, p0=p0, p1=p1, parent_id=i))

    def circle(self, c: tuple[float, float], r: float) -> None:
        i = self._next()
        self.prims.append(Arc(id=i, center=c, r=r, t0=0.0, t1=TWO_PI, parent_id=i))

    def arc(self, c: tuple[float, float], r: float, t0: float, t1: float) -> None:
        i = self._next()
        k = math.floor(t0 / TWO_PI)
        self.prims.append(
            Arc(id=i, center=c, r=r, t0=t0 - k * TWO_PI, t1=t1 - k * TWO_PI, parent_id=i)
        )

    def rect(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.seg((x0, y0), (x1, y0))
        self.seg((x1, y0), (x1, y1))
        self.seg((x1, y1), (x0, y1))
        self.seg((x0, y1), (x0, y0))

    def text(self, s: str, c: tuple[float, float], fs: float) -> None:
        i = self._next()
        w = 0.6 * fs * len(s)
        bbox = (c[0] - w / 2, c[1] - fs / 2, c[0] + w / 2, c[1] + fs / 2)
        self.prims.append(TextPrim(id=i, text=s, bbox=bbox, parent_id=i))

    def done(self, legend_id: str, label: str, group: str | None) -> SymbolDef:
        b = prims_bbox(self.prims)
        assert b is not None
        T = Similarity(tx=-(b[0] + b[2]) / 2, ty=-(b[1] + b[3]) / 2)
        prims = tuple(transform_prim(p, T) for p in self.prims)
        return SymbolDef(legend_id, label, prims, group)


def _duplex(b: _Builder) -> None:
    b.circle((0.0, 0.0), 0.3)
    b.seg((-0.12, -0.17), (-0.12, 0.17))
    b.seg((0.12, -0.17), (0.12, 0.17))


def _with_text_right(legend_id: str, label: str, s: str) -> SymbolDef:
    b = _Builder()
    _duplex(b)
    fs = 0.22
    w = 0.6 * fs * len(s)
    b.text(s, (0.3 + 0.07 + w / 2, 0.0), fs)
    return b.done(legend_id, label, "receptacle")


def _switch(legend_id: str, label: str, s: str) -> SymbolDef:
    b = _Builder()
    fs = 0.5
    w = 0.6 * fs * len(s)
    b.text(s, (0.0, 0.0), fs)
    b.seg((w / 2 + 0.04, -0.12), (w / 2 + 0.34, -0.32))
    return b.done(legend_id, label, "switch")


def _detector(legend_id: str, label: str, s: str) -> SymbolDef:
    b = _Builder()
    b.circle((0.0, 0.0), 0.35)
    b.text(s, (0.0, 0.0), 0.28)
    return b.done(legend_id, label, "detector")


def symbol_library() -> dict[str, SymbolDef]:
    """>= 12 символов, включая группы похожих (см. PLAN.md T1)."""
    lib: list[SymbolDef] = []

    b = _Builder()
    _duplex(b)
    lib.append(b.done("duplex", "Duplex receptacle", "receptacle"))
    lib.append(_with_text_right("gfci", "GFCI receptacle", "GFCI"))
    lib.append(_with_text_right("r220", "220V receptacle", "220"))

    lib.append(_switch("switch_s", "Switch", "S"))
    lib.append(_switch("switch_s3", "3-way switch", "S3"))
    lib.append(_switch("switch_sd", "Dimmer switch", "SD"))

    d = 0.35 / math.sqrt(2)
    b = _Builder()
    b.circle((0.0, 0.0), 0.35)
    b.seg((-d, -d), (d, d))
    lib.append(b.done("light_recessed", "Recessed light", "light"))
    b = _Builder()
    b.circle((0.0, 0.0), 0.35)
    b.seg((-d, -d), (d, d))
    b.seg((-d, d), (d, -d))
    lib.append(b.done("light_surface", "Surface light", "light"))

    lib.append(_detector("det_smoke", "Smoke detector", "SD"))
    lib.append(_detector("det_co", "CO detector", "CO"))

    b = _Builder()
    b.circle((0.0, 0.0), 0.35)
    for k in range(3):
        a = k * TWO_PI / 3
        c = (0.17 * math.cos(a), 0.17 * math.sin(a))
        b.arc(c, 0.17, a + 0.6, a + 0.6 + math.pi)
    lib.append(b.done("fan", "Ceiling fan", None))

    b = _Builder()
    b.rect(-0.3, -0.3, 0.3, 0.3)
    b.text("J", (0.0, 0.0), 0.4)
    lib.append(b.done("junction", "Junction box", None))

    b = _Builder()
    b.seg((0.0, -0.4), (0.4, 0.3))
    b.seg((0.4, 0.3), (-0.4, 0.3))
    b.seg((-0.4, 0.3), (0.0, -0.4))
    b.seg((-0.2, 0.1), (0.2, 0.1))
    lib.append(b.done("telephone", "Telephone outlet", None))

    b = _Builder()
    b.rect(-0.4, -0.25, 0.4, 0.25)
    b.seg((-0.4, 0.25), (0.4, -0.25))
    lib.append(b.done("panel", "Panel board", None))

    b = _Builder()
    b.rect(-0.5, -0.2, 0.5, 0.2)
    b.text("EX", (0.0, 0.0), 0.28)
    lib.append(b.done("exit_sign", "Exit sign", None))

    b = _Builder()
    b.arc((0.0, 0.0), 0.35, math.pi, TWO_PI)
    b.seg((-0.35, 0.0), (0.35, 0.0))
    b.seg((0.0, 0.0), (0.0, 0.2))
    lib.append(b.done("bell", "Alarm bell", None))

    return {s.legend_id: s for s in lib}


def similar_groups(lib: dict[str, SymbolDef] | None = None) -> list[list[str]]:
    """Группы похожих символов (порядок библиотеки), только группы из >= 2 классов."""
    lib = lib or symbol_library()
    groups: dict[str, list[str]] = {}
    for s in lib.values():
        if s.group:
            groups.setdefault(s.group, []).append(s.legend_id)
    return [g for g in groups.values() if len(g) >= 2]
