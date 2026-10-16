"""Генерация плана с размещенными символами. Задача T1."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from symspot.core.geometry_utils import bbox_diag, bbox_expand, prims_bbox, transform_prim
from symspot.core.types import Arc, BBox, QuerySpec, Segment, Similarity, TextPrim
from symspot.synth.elements import BoxIndex, Element, text_prim
from symspot.synth.layout import (
    NOTE_TEXTS,
    ROOM_NAMES,
    Layout,
    Rect,
    build_layout,
    hatch_segments,
)
from symspot.synth.symbols import SymbolDef, similar_groups, symbol_library

# --- параметры генератора (константы синтетики, не пороги алгоритма поиска) ---
PAGES: dict[str, tuple[float, float]] = {"small": (600.0, 400.0), "large": (1728.0, 1152.0)}
TARGET_PRIMS: dict[str, int] = {"small": 2000, "large": 30000}
N_INSTANCES: dict[str, tuple[int, int]] = {"small": (40, 80), "large": (400, 800)}
MIN_ROOM: dict[str, float] = {"small": 60.0, "large": 80.0}
STOP_P: dict[str, float] = {"small": 0.4, "large": 0.3}
HATCH_SPACING: dict[str, tuple[float, float]] = {"small": (1.3, 2.0), "large": (0.9, 1.4)}
CROSS_HATCH_P = 0.5

SYMBOL_DIAG = 15.0  # номинальная диагональ символа, pt
SCALE_JITTER = 0.15
GT_PAD = 0.08  # расширение рамки GT (доля диагонали)
HARD_FRACTION = 0.30
LEGEND_GAP_FACTOR = 0.6  # зазор до подписи, доли диагонали рамки эталона

FS_DIM = 5.0
FS_ROOM = 6.0
FS_NOTE = 5.0
FS_LABEL = 5.0
DIM_SPACING = 11.0

W_OUTER, W_INNER, W_DOOR, W_DIM, W_HATCH, W_SYMBOL, W_BORDER = 1.0, 0.7, 0.5, 0.3, 0.25, 0.6, 0.8

HardTag = Literal["wall_through", "near_hatch", "split_lines", "bezier_circle"]
ALL_TAGS: tuple[str, ...] = ("wall_through", "near_hatch", "split_lines", "bezier_circle")


@dataclass
class PlanInstance:
    """Экземпляр символа (или эталон легенды) с геометрией в координатах листа."""

    index: int
    legend_id: str
    T: Similarity
    bbox: BBox  # рамка GT (геометрия + 8 % диагонали с каждой стороны)
    elements: list[Element]
    is_query: bool = False
    hard: list[str] = field(default_factory=list)


@dataclass
class SynthPlan:
    seed: int
    size: str
    width: float
    height: float
    background: list[Element]
    instances: list[PlanInstance]  # сначала экземпляры плана, затем эталоны легенды
    queries: list[QuerySpec]
    similar_groups: list[list[str]]
    library: dict[str, SymbolDef]

    @property
    def elements(self) -> list[Element]:
        out = list(self.background)
        for inst in self.instances:
            out.extend(inst.elements)
        return out

    @property
    def n_prims(self) -> int:
        return len(self.background) + sum(len(i.elements) for i in self.instances)

    @property
    def plan_instances(self) -> list[PlanInstance]:
        return [i for i in self.instances if not i.is_query]


def _shrink(r: Rect, d: float) -> Rect:
    return (r[0] + d, r[1] + d, r[2] - d, r[3] - d)


def _inside(b: BBox, r: Rect) -> bool:
    return b[0] >= r[0] and b[1] >= r[1] and b[2] <= r[2] and b[3] <= r[3]


def gt_bbox(tight: BBox) -> BBox:
    return bbox_expand(tight, GT_PAD * bbox_diag(tight))


class _Generator:
    def __init__(self, seed: int, size: str) -> None:
        self.seed = seed
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.W, self.H = PAGES[size]
        self.lib = symbol_library()
        self.bg: list[Element] = []
        self.instances: list[PlanInstance] = []
        self.occ = BoxIndex()
        self._pid = 0
        self.nominal = {k: SYMBOL_DIAG / s.diag for k, s in self.lib.items()}
        self.layout: Layout | None = None
        self.room_inner: list[Rect] = []

    # --- примитивы фона ---
    def pid(self) -> int:
        self._pid += 1
        return self._pid

    def seg(self, p0: tuple[float, float], p1: tuple[float, float], width: float) -> None:
        i = self.pid()
        self.bg.append(Element("", Segment(i, p0, p1, i), width))

    def arc(self, c: tuple[float, float], r: float, t0: float, t1: float, width: float) -> None:
        i = self.pid()
        self.bg.append(Element("", Arc(i, c, r, t0, t1, i), width))

    def text(self, s: str, cx: float, cy: float, fs: float, angle: float = 0.0) -> BBox:
        p = text_prim(self.pid(), s, cx, cy, fs, angle)
        self.bg.append(Element("", p, 0.0, font_size=fs))
        return p.bbox

    def rect(self, r: Rect, width: float) -> None:
        x0, y0, x1, y1 = r
        self.seg((x0, y0), (x1, y0), width)
        self.seg((x1, y0), (x1, y1), width)
        self.seg((x1, y1), (x0, y1), width)
        self.seg((x0, y1), (x0, y0), width)

    # --- легенда ---
    def build_legend(self) -> tuple[float, list[PlanInstance], list[tuple[BBox, str]]]:
        """Легенда у правого края; возвращает левую границу и экземпляры-эталоны."""
        classes = list(self.lib.values())
        frames = []
        for sym in classes:
            T = Similarity(s=self.nominal[sym.legend_id])
            tp = [transform_prim(p, T) for p in sym.prims]
            frames.append(gt_bbox(prims_bbox(tp)))  # type: ignore[arg-type]
        half_w = max((f[2] - f[0]) / 2 for f in frames)
        gap = LEGEND_GAP_FACTOR * max(bbox_diag(f) for f in frames)
        label_w = max(0.6 * FS_LABEL * len(s.label) for s in classes)
        pad = 8.0
        legend_w = pad + 2 * half_w + gap + label_w + pad
        lx0 = self.W - 8.0 - legend_w
        cx = lx0 + pad + half_w
        label_x0 = cx + half_w + gap
        top = 8.0 + 18.0
        pitch = min(30.0, (self.H - 8.0 - 8.0 - 18.0 - 6.0) / len(classes))
        self.rect((lx0, 8.0, self.W - 8.0, self.H - 8.0), W_BORDER)
        self.text("LEGEND", lx0 + legend_w / 2, 8.0 + 9.0, FS_LABEL + 1.0)
        insts = []
        n_plan = 0  # индексы эталонов уточняются позже
        for j, sym in enumerate(classes):
            cy = top + pitch * (j + 0.5)
            T = Similarity(s=self.nominal[sym.legend_id], tx=cx, ty=cy)
            inst = self._make_instance(n_plan + j, sym, T, is_query=True)
            insts.append(inst)
            w = 0.6 * FS_LABEL * len(sym.label)
            self.text(sym.label, label_x0 + w / 2, cy, FS_LABEL)
        return lx0, insts, []

    # --- создание экземпляра ---
    def _make_instance(
        self,
        index: int,
        sym: SymbolDef,
        T: Similarity,
        *,
        is_query: bool = False,
        split: bool = False,
        bezier: bool = False,
        hard: list[str] | None = None,
    ) -> PlanInstance:
        rng = self.rng
        tprims = [transform_prim(p, T) for p in sym.prims]
        tight = prims_bbox(tprims)
        assert tight is not None
        elements: list[Element] = []
        k = 0

        def eid() -> str:
            nonlocal k
            s = f"i{index}_{k}"
            k += 1
            return s

        segs = [i for i, p in enumerate(tprims) if isinstance(p, Segment)]
        split_set: set[int] = set()
        if split and segs:
            for i in segs:
                if rng.random() < 0.7:
                    split_set.add(i)
            if not split_set:
                split_set.add(segs[int(rng.integers(len(segs)))])
        for i, (p, lp) in enumerate(zip(tprims, sym.prims, strict=True)):
            if isinstance(p, Segment):
                if i in split_set:
                    for q0, q1 in self._split_segment(p):
                        pid = self.pid()
                        elements.append(Element(eid(), Segment(pid, q0, q1, pid), W_SYMBOL))
                else:
                    elements.append(Element(eid(), p, W_SYMBOL))
            elif isinstance(p, Arc):
                elements.append(Element(eid(), p, W_SYMBOL, bezier=bezier and p.is_full))
            elif isinstance(p, TextPrim):
                assert isinstance(lp, TextPrim)
                fs = (lp.bbox[3] - lp.bbox[1]) * T.s
                elements.append(Element(eid(), p, 0.0, font_size=fs))
        return PlanInstance(
            index=index,
            legend_id=sym.legend_id,
            T=T,
            bbox=gt_bbox(tight),
            elements=elements,
            is_query=is_query,
            hard=list(hard or []),
        )

    def _split_segment(self, s: Segment) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        n = int(self.rng.integers(2, 4))
        cuts = sorted(float(c) for c in self.rng.uniform(0.2, 0.8, n - 1))
        edges = [0.0, *cuts, 1.0]
        if min(b - a for a, b in zip(edges, edges[1:], strict=False)) < 0.12:
            edges = [i / n for i in range(n + 1)]
        pts = [
            (s.p0[0] + (s.p1[0] - s.p0[0]) * f, s.p0[1] + (s.p1[1] - s.p0[1]) * f) for f in edges
        ]
        return [(pts[i], pts[i + 1]) for i in range(n)]

    # --- стены, двери, размеры ---
    def build_building(self, region: Rect) -> None:
        rng = self.rng
        m = 2 * DIM_SPACING + FS_DIM + 10.0
        building = _shrink(region, m)
        t_out = float(rng.uniform(2.5, 3.0))
        t_in = float(rng.uniform(2.0, 2.5))
        lay = build_layout(
            rng,
            building,
            min_room=MIN_ROOM[self.size],
            stop_p=STOP_P[self.size],
            t_out=t_out,
            t_in=t_in,
        )
        self.layout = lay
        for w in lay.walls:
            width = W_INNER if w.interior else W_OUTER
            for p0, p1 in w.lines():
                self.seg(p0, p1, width)
        for d in lay.doors:
            h, e = d.leaf()
            self.seg(h, e, W_DOOR)
            c, r, t0, t1 = d.arc()
            self.arc(c, r, t0, t1, W_DOOR)
            self.occ.add(d.bbox())
        self.room_inner = [_shrink(r, t_in / 2 + 1.0) for r in lay.rooms]
        self._dims(lay)

    def _dim_line(
        self, breaks: list[float], axis: str, line: float, edge: float, nrm: float
    ) -> None:
        """Размерная линия по цепочке координат ``breaks``; axis "h": вдоль X на y=line."""

        def pt(a: float, c: float) -> tuple[float, float]:
            return (a, c) if axis == "h" else (c, a)

        for a in breaks:
            self.seg(pt(a, edge + nrm * 1.5), pt(a, line + nrm * 2.0), W_DIM)
            self.seg(pt(a - 1.8, line + 1.8), pt(a + 1.8, line - 1.8), W_DIM)
        for a0, a1 in zip(breaks, breaks[1:], strict=False):
            self.seg(pt(a0, line), pt(a1, line), W_DIM)
            s = str(int(round((a1 - a0) * 25 / 10) * 10))
            if a1 - a0 < 0.6 * FS_DIM * len(s) + 2:
                continue
            mid = (a0 + a1) / 2
            off = line + nrm * (FS_DIM / 2 + 1.2)
            if axis == "h":
                self.text(s, mid, off, FS_DIM)
            else:
                self.text(s, off, mid, FS_DIM, -math.pi / 2)

    @staticmethod
    def _chain(vals: list[float], lo: float, hi: float, gap: float = 16.0) -> list[float]:
        out = [lo]
        for v in sorted(vals):
            if v - out[-1] >= gap and hi - v >= gap:
                out.append(v)
        out.append(hi)
        return out

    def _dims(self, lay: Layout) -> None:
        x0, y0, x1, y1 = lay.building
        d = DIM_SPACING
        vx = [w.c for w in lay.walls if w.interior and w.orient == "v"]
        hy = [w.c for w in lay.walls if w.interior and w.orient == "h"]
        self._dim_line(self._chain(vx, x0, x1), "h", y0 - d, y0, -1.0)
        self._dim_line([x0, x1], "h", y0 - 2 * d, y0, -1.0)
        self._dim_line([x0, x1], "h", y1 + d, y1, 1.0)
        self._dim_line(self._chain(hy, y0, y1), "v", x0 - d, x0, -1.0)
        self._dim_line([y0, y1], "v", x0 - 2 * d, x0, -1.0)
        self._dim_line([y0, y1], "v", x1 + d, x1, 1.0)

    # --- текст в комнатах ---
    def room_labels(self) -> None:
        rng = self.rng
        for r in self.room_inner:
            name = f"{ROOM_NAMES[int(rng.integers(len(ROOM_NAMES)))]} {int(rng.integers(100, 400))}"
            w = 0.6 * FS_ROOM * len(name)
            if r[2] - r[0] < w + 10 or r[3] - r[1] < 24:
                continue
            cx = (r[0] + r[2]) / 2 + float(rng.uniform(-0.15, 0.15)) * (r[2] - r[0] - w)
            cy = (r[1] + r[3]) / 2 + float(rng.uniform(-0.25, 0.25)) * (r[3] - r[1]) / 2
            b = self.text(name, cx, cy, FS_ROOM)
            self.occ.add(b)

    def _pick_room(self, min_dim: float = 34.0) -> Rect | None:
        areas = np.array(
            [
                (r[2] - r[0]) * (r[3] - r[1]) if min(r[2] - r[0], r[3] - r[1]) >= min_dim else 0.0
                for r in self.room_inner
            ]
        )
        if areas.sum() <= 0:
            return None
        i = int(self.rng.choice(len(areas), p=areas / areas.sum()))
        return self.room_inner[i]

    def _wall_point(self) -> tuple[float, float, Rect] | None:
        """Точка на оси внутренней стены вдали от проемов и концов (для wall_through)."""
        assert self.layout is not None
        cand = []
        for w in self.layout.walls:
            if not w.interior:
                continue
            for p0, p1 in w.line_pieces():
                if p1 - p0 > 40.0:
                    cand.append((w, p0, p1))
        if not cand:
            return None
        lens = np.array([p1 - p0 for _, p0, p1 in cand])
        w, p0, p1 = cand[int(self.rng.choice(len(cand), p=lens / lens.sum()))]
        a = float(self.rng.uniform(p0 + 14.0, p1 - 14.0))
        x, y = (a, w.c) if w.orient == "h" else (w.c, a)
        return x, y, self.layout.building

    # --- размещение экземпляров ---
    def _random_T(self, sym: SymbolDef, cx: float, cy: float) -> Similarity:
        rng = self.rng
        s = self.nominal[sym.legend_id] * float(rng.uniform(1 - SCALE_JITTER, 1 + SCALE_JITTER))
        if rng.random() < 0.4:
            theta = float(rng.integers(0, 4)) * math.pi / 2
        else:
            theta = float(rng.uniform(-math.pi, math.pi))
        mirror = (not sym.has_text) and bool(rng.random() < 0.5)
        return Similarity(s=s, theta=theta, tx=cx, ty=cy, mirror=mirror)

    def _choose_tags(self, sym: SymbolDef) -> list[str]:
        rng = self.rng
        if rng.random() >= HARD_FRACTION:
            return []
        has_circle = any(isinstance(p, Arc) and p.is_full for p in sym.prims)
        has_seg = any(isinstance(p, Segment) for p in sym.prims)
        opts = ["wall_through", "near_hatch"]
        if has_seg:
            opts.append("split_lines")
        if has_circle:
            opts.append("bezier_circle")
        tags = [opts[int(rng.integers(len(opts)))]]
        if rng.random() < 0.2:
            extra = [o for o in ("split_lines", "bezier_circle") if o in opts and o not in tags]
            if extra:
                tags.append(extra[int(rng.integers(len(extra)))])
        return tags

    def _hatch_beside(self, tight: BBox) -> Rect:
        rng = self.rng
        gap = 0.8
        side = int(rng.integers(4))
        depth = float(rng.uniform(14.0, 30.0))
        cx, cy = (tight[0] + tight[2]) / 2, (tight[1] + tight[3]) / 2
        if side in (0, 1):
            h = tight[3] - tight[1] + float(rng.uniform(6.0, 16.0))
            if side == 0:
                return (tight[2] + gap, cy - h / 2, tight[2] + gap + depth, cy + h / 2)
            return (tight[0] - gap - depth, cy - h / 2, tight[0] - gap, cy + h / 2)
        w = tight[2] - tight[0] + float(rng.uniform(6.0, 16.0))
        if side == 2:
            return (cx - w / 2, tight[3] + gap, cx + w / 2, tight[3] + gap + depth)
        return (cx - w / 2, tight[1] - gap - depth, cx + w / 2, tight[1] - gap)

    def add_hatch(self, r: Rect) -> None:
        rng = self.rng
        lo, hi = HATCH_SPACING[self.size]
        spacing = float(rng.uniform(lo, hi))
        angle = float(
            rng.choice([math.pi / 4, 3 * math.pi / 4, 0.0, math.pi / 2], p=[0.4, 0.4, 0.1, 0.1])
        )
        self.rect(r, W_HATCH)
        for p0, p1 in hatch_segments(r, spacing, angle):
            self.seg(p0, p1, W_HATCH)
        if rng.random() < CROSS_HATCH_P:
            for p0, p1 in hatch_segments(r, spacing, angle + math.pi / 2):
                self.seg(p0, p1, W_HATCH)
        self.occ.add(r)

    def place_instance(self, index: int, sym: SymbolDef) -> PlanInstance | None:
        rng = self.rng
        assert self.layout is not None
        tags = self._choose_tags(sym)
        for attempt in range(80):
            if attempt == 40 and tags:
                tags = []  # не удается разместить сложный случай: обычный экземпляр
            use_wall = "wall_through" in tags
            if use_wall:
                wp = self._wall_point()
                if wp is None:
                    tags = [t for t in tags if t != "wall_through"] or []
                    continue
                cx, cy, container = wp
            else:
                room = self._pick_room()
                if room is None:
                    return None
                cx = float(rng.uniform(room[0] + 6.0, room[2] - 6.0))
                cy = float(rng.uniform(room[1] + 6.0, room[3] - 6.0))
                container = room
            T = self._random_T(sym, cx, cy)
            tprims = [transform_prim(p, T) for p in sym.prims]
            tight = prims_bbox(tprims)
            assert tight is not None
            gt = gt_bbox(tight)
            if not _inside(gt, container):
                continue
            if self.occ.hits(gt, 1.0):
                continue
            hatch: Rect | None = None
            if "near_hatch" in tags:
                hatch = self._hatch_beside(tight)
                if not _inside(hatch, container) or self.occ.hits(hatch, 1.0):
                    continue
            inst = self._make_instance(
                index,
                sym,
                T,
                split="split_lines" in tags,
                bezier="bezier_circle" in tags,
                hard=tags,
            )
            self.occ.add(inst.bbox)
            if hatch is not None:
                self.add_hatch(hatch)
            return inst
        return None

    def place_all(self) -> None:
        rng = self.rng
        lo, hi = N_INSTANCES[self.size]
        n = int(rng.integers(lo, hi + 1))
        ids = list(self.lib)
        order = [ids[i] for i in rng.permutation(len(ids))]
        order += [ids[int(i)] for i in rng.integers(len(ids), size=max(0, n - len(ids)))]
        for lid in order[:n]:
            inst = self.place_instance(len(self.instances), self.lib[lid])
            if inst is not None:
                self.instances.append(inst)

    # --- наполнение ---
    def fill(self) -> None:
        rng = self.rng
        target = int(TARGET_PRIMS[self.size] * float(rng.uniform(0.92, 1.08)))
        # случайные заметки
        n_notes = int(rng.integers(12, 20)) if self.size == "small" else int(rng.integers(80, 140))
        for _ in range(n_notes):
            for _try in range(20):
                room = self._pick_room(40.0)
                if room is None:
                    break
                s = NOTE_TEXTS[int(rng.integers(len(NOTE_TEXTS)))]
                vertical = rng.random() < 0.15
                w = 0.6 * FS_NOTE * len(s)
                ext = (FS_NOTE, w) if vertical else (w, FS_NOTE)
                if room[2] - room[0] < ext[0] + 4 or room[3] - room[1] < ext[1] + 4:
                    continue
                cx = float(rng.uniform(room[0] + ext[0] / 2 + 1, room[2] - ext[0] / 2 - 1))
                cy = float(rng.uniform(room[1] + ext[1] / 2 + 1, room[3] - ext[1] / 2 - 1))
                box = (cx - ext[0] / 2, cy - ext[1] / 2, cx + ext[0] / 2, cy + ext[1] / 2)
                if self.occ.hits(box, 2.0):
                    continue
                self.text(s, cx, cy, FS_NOTE, -math.pi / 2 if vertical else 0.0)
                self.occ.add(box)
                break
        # штриховки
        inst_prims = sum(len(i.elements) for i in self.instances)
        legend_est = len(self.lib) * 8
        scale = 1.0
        for _ in range(60000):
            if len(self.bg) + inst_prims + legend_est >= target:
                break
            room = self._pick_room(20.0)
            if room is None:
                break
            w = float(rng.uniform(14.0, 44.0)) * scale
            h = float(rng.uniform(12.0, 36.0)) * scale
            if room[2] - room[0] < w + 4 or room[3] - room[1] < h + 4:
                continue
            x = float(rng.uniform(room[0] + 1, room[2] - w - 1))
            y = float(rng.uniform(room[1] + 1, room[3] - h - 1))
            r = (x, y, x + w, y + h)
            if self.occ.hits(r, 2.0):
                continue
            self.add_hatch(r)


def generate_plan(seed: int, size: Literal["small", "large"]) -> SynthPlan:
    """Детерминированный по seed план (стены, двери, размеры, штриховки, текст) +
    экземпляры символов + легенда (по одному эталону каждого класса)."""
    if size not in PAGES:
        raise ValueError(f"size должен быть 'small' или 'large', получено {size!r}")
    g = _Generator(seed, size)
    lib = g.lib
    # рамка листа
    g.rect((4.0, 4.0, g.W - 4.0, g.H - 4.0), W_BORDER)
    lx0, legend_insts, _ = g.build_legend()
    g.build_building((12.0, 12.0, lx0 - 8.0, g.H - 12.0))
    g.room_labels()
    g.place_all()
    g.fill()

    n_plan = len(g.instances)
    for j, inst in enumerate(legend_insts):
        old = inst.index
        inst.index = n_plan + j
        for e in inst.elements:
            e.id = e.id.replace(f"i{old}_", f"i{inst.index}_", 1)
    instances = g.instances + legend_insts
    for k, e in enumerate(g.bg):
        e.id = f"bg{k}"
    queries = [QuerySpec(i.legend_id, i.bbox, lib[i.legend_id].label) for i in legend_insts]
    return SynthPlan(
        seed=seed,
        size=size,
        width=g.W,
        height=g.H,
        background=g.bg,
        instances=instances,
        queries=queries,
        similar_groups=similar_groups(lib),
        library=lib,
    )
