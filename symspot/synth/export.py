"""Экспорт синтетики: SVG, PDF, PNG, грязный PNG, GT JSON, COCO. Задача T1."""

from __future__ import annotations

import ctypes.util
import json
import math
import os
import sys
from pathlib import Path
from types import ModuleType
from xml.sax.saxutils import escape

import cv2
import numpy as np

from symspot.core.types import Arc, BBox, Segment, TextPrim
from symspot.synth.elements import BEZIER_KAPPA, Element
from symspot.synth.plan_generator import PlanInstance, SynthPlan, generate_plan

DPI = 300
PX_PER_PT = DPI / 72.0
FONT_FAMILY = "DejaVu Sans, Arial, sans-serif"
BASELINE_SHIFT = 0.365  # смещение базовой линии от центра рамки текста, доли кегля
DIRTY_ANGLE_DEG = 1.0
DIRTY_BLUR_SIGMA = (0.6, 1.2)  # px
DIRTY_NOISE_STD = (5.0, 12.0)  # уровни серого


def _load_cairosvg() -> ModuleType:
    """Импорт cairosvg; на macOS подсказываем путь к Homebrew, если libcairo не найдена."""
    try:
        import cairosvg
    except OSError:
        if sys.platform != "darwin":
            raise
        orig = ctypes.util.find_library

        def find(name: str) -> str | None:
            r = orig(name)
            if r:
                return r
            for d in ("/opt/homebrew/lib", "/usr/local/lib"):
                for cand in (f"lib{name}.dylib", f"lib{name}.2.dylib", f"{name}.dylib"):
                    path = os.path.join(d, cand)
                    if os.path.exists(path):
                        return path
            return None

        ctypes.util.find_library = find
        for m in [m for m in sys.modules if m.startswith(("cairosvg", "cairocffi"))]:
            del sys.modules[m]
        import cairosvg
    return cairosvg


def _f(v: float) -> str:
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


# ---------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------


def _path_d(e: Element) -> str:
    p = e.prim
    if isinstance(p, Segment):
        return f"M {_f(p.p0[0])} {_f(p.p0[1])} L {_f(p.p1[0])} {_f(p.p1[1])}"
    assert isinstance(p, Arc)
    cx, cy = p.center
    r = p.r
    if p.is_full:
        if e.bezier:
            k = BEZIER_KAPPA * r
            pts = [
                (cx + r, cy),
                (cx + r, cy + k, cx + k, cy + r, cx, cy + r),
                (cx - k, cy + r, cx - r, cy + k, cx - r, cy),
                (cx - r, cy - k, cx - k, cy - r, cx, cy - r),
                (cx + k, cy - r, cx + r, cy - k, cx + r, cy),
            ]
            d = f"M {_f(pts[0][0])} {_f(pts[0][1])}"
            for seg in pts[1:]:
                d += " C " + " ".join(_f(v) for v in seg)
            return d
        return (
            f"M {_f(cx + r)} {_f(cy)} A {_f(r)} {_f(r)} 0 1 1 {_f(cx - r)} {_f(cy)} "
            f"A {_f(r)} {_f(r)} 0 1 1 {_f(cx + r)} {_f(cy)}"
        )
    sx, sy = cx + r * math.cos(p.t0), cy + r * math.sin(p.t0)
    ex, ey = cx + r * math.cos(p.t1), cy + r * math.sin(p.t1)
    large = 1 if p.t1 - p.t0 > math.pi + 1e-9 else 0
    return f"M {_f(sx)} {_f(sy)} A {_f(r)} {_f(r)} 0 {large} 1 {_f(ex)} {_f(ey)}"


def _element_svg(e: Element) -> str:
    p = e.prim
    if isinstance(p, TextPrim):
        cx = (p.bbox[0] + p.bbox[2]) / 2
        cy = (p.bbox[1] + p.bbox[3]) / 2
        d = BASELINE_SHIFT * e.font_size
        bx = cx - d * math.sin(p.angle)
        by = cy + d * math.cos(p.angle)
        tr = ""
        if abs(p.angle) > 1e-9:
            tr = f' transform="rotate({_f(math.degrees(p.angle))} {_f(bx)} {_f(by)})"'
        return (
            f'<text id="{e.id}" x="{_f(bx)}" y="{_f(by)}" font-size="{_f(e.font_size)}" '
            f'font-family="{FONT_FAMILY}" text-anchor="middle" fill="black"{tr}>'
            f"{escape(p.text)}</text>"
        )
    return (
        f'<path id="{e.id}" d="{_path_d(e)}" stroke="black" '
        f'stroke-width="{_f(e.width)}" fill="none"/>'
    )


def plan_to_svg(plan: SynthPlan) -> str:
    W, H = _f(plan.width), _f(plan.height)
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}pt" height="{H}pt" '
        f'viewBox="0 0 {W} {H}">',
    ]
    parts.extend(_element_svg(e) for e in plan.elements)
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# GT и COCO
# ---------------------------------------------------------------------------


def _round_box(b: BBox) -> list[float]:
    return [round(float(v), 4) for v in b]


def plan_to_gt(plan: SynthPlan, name: str) -> dict:
    return {
        "name": name,
        "width": plan.width,
        "height": plan.height,
        "units": "pt",
        "similar_groups": plan.similar_groups,
        "queries": [
            {"legend_id": q.legend_id, "bbox": _round_box(q.bbox), "label": q.label}
            for q in plan.queries
        ],
        "instances": [
            {
                "legend_id": i.legend_id,
                "T": i.T.to_dict(),
                "bbox": _round_box(i.bbox),
                "path_ids": [e.id for e in i.elements],
                "is_query": i.is_query,
                "hard": list(i.hard),
            }
            for i in plan.instances
        ],
    }


def _px_size(plan: SynthPlan) -> tuple[int, int]:
    return round(plan.width * PX_PER_PT), round(plan.height * PX_PER_PT)


def _coco(plan: SynthPlan, file_name: str, boxes: list[BBox], w_px: int, h_px: int) -> dict:
    """COCO (CVAT 1.0). ``boxes`` в пикселях (x0, y0, x1, y1), по одному на plan.instances."""
    annotations = []
    for n, (inst, b) in enumerate(zip(plan.instances, boxes, strict=True), start=1):
        bw, bh = b[2] - b[0], b[3] - b[1]
        annotations.append(
            {
                "id": n,
                "image_id": 1,
                "category_id": 1 if inst.is_query else 2,
                "bbox": [round(b[0], 3), round(b[1], 3), round(bw, 3), round(bh, 3)],
                "area": round(bw * bh, 3),
                "iscrowd": 0,
                "segmentation": [],
                "attributes": {
                    "legend_id": inst.legend_id,
                    "meaning": plan.library[inst.legend_id].label,
                    "occluded": False,
                    "rotation": 0.0,
                },
            }
        )
    return {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {
            "contributor": "",
            "date_created": "",
            "description": "SymSpot synth",
            "url": "",
            "version": "",
            "year": "",
        },
        "categories": [
            {"id": 1, "name": "legend_item", "supercategory": ""},
            {"id": 2, "name": "object", "supercategory": ""},
        ],
        "images": [
            {
                "id": 1,
                "width": w_px,
                "height": h_px,
                "file_name": file_name,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0,
            }
        ],
        "annotations": annotations,
    }


def _boxes_px(plan: SynthPlan) -> list[BBox]:
    k = PX_PER_PT
    return [(i.bbox[0] * k, i.bbox[1] * k, i.bbox[2] * k, i.bbox[3] * k) for i in plan.instances]


def _rotate_boxes(boxes: list[BBox], M: np.ndarray, w_px: int, h_px: int) -> list[BBox]:
    out = []
    for b in boxes:
        c = np.array([[b[0], b[1], 1], [b[2], b[1], 1], [b[2], b[3], 1], [b[0], b[3], 1]], float)
        q = c @ M.T
        x0, y0 = q.min(axis=0)
        x1, y1 = q.max(axis=0)
        out.append((max(0.0, x0), max(0.0, y0), min(float(w_px), x1), min(float(h_px), y1)))
    return out


# ---------------------------------------------------------------------------
# Растр
# ---------------------------------------------------------------------------


def _render_png(cairosvg: ModuleType, svg: bytes, w_px: int, h_px: int) -> np.ndarray:
    data = cairosvg.svg2png(
        bytestring=svg,
        background_color="white",
        output_width=w_px,
        output_height=h_px,
    )
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("не удалось декодировать PNG от cairosvg")
    return img


def make_dirty(img: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Шум, размытие, поворот ±1° вокруг центра; параметры детерминированы seed.

    Возвращает изображение и матрицу 2x3 (исходные пиксели -> грязные).
    """
    rng = np.random.default_rng(seed + 7919)
    h, w = img.shape
    angle = float(rng.uniform(-DIRTY_ANGLE_DEG, DIRTY_ANGLE_DEG))
    sigma = float(rng.uniform(*DIRTY_BLUR_SIGMA))
    noise_std = float(rng.uniform(*DIRTY_NOISE_STD))
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    out = cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255
    )
    out = cv2.GaussianBlur(out, (0, 0), sigma)
    noise = rng.standard_normal(out.shape, dtype=np.float32)
    noise *= noise_std
    noise += out
    np.clip(noise, 0, 255, out=noise)
    return noise.astype(np.uint8), M


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")


def export_plan(plan: SynthPlan, out_dir: Path, name: str, raster: bool = True) -> dict[str, Path]:
    """Записать SVG, PDF, GT, COCO (+ PNG и грязный PNG при ``raster``); вернуть пути."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    svg_text = plan_to_svg(plan)
    svg_bytes = svg_text.encode("utf-8")
    paths["svg"] = out_dir / f"{name}.svg"
    paths["svg"].write_text(svg_text, encoding="utf-8")
    cairosvg = _load_cairosvg()
    paths["pdf"] = out_dir / f"{name}.pdf"
    paths["pdf"].write_bytes(cairosvg.svg2pdf(bytestring=svg_bytes))
    paths["gt"] = out_dir / f"{name}_gt.json"
    _write_json(paths["gt"], plan_to_gt(plan, name))

    w_px, h_px = _px_size(plan)
    boxes = _boxes_px(plan)
    paths["coco"] = out_dir / f"{name}_coco.json"
    _write_json(paths["coco"], _coco(plan, f"{name}.png", boxes, w_px, h_px))
    if raster:
        img = _render_png(cairosvg, svg_bytes, w_px, h_px)
        paths["png"] = out_dir / f"{name}.png"
        cv2.imwrite(str(paths["png"]), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        dirty, M = make_dirty(img, plan.seed)
        paths["dirty_png"] = out_dir / f"{name}_dirty.png"
        cv2.imwrite(str(paths["dirty_png"]), dirty, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        dboxes = _rotate_boxes(boxes, M, w_px, h_px)
        paths["dirty_coco"] = out_dir / f"{name}_dirty_coco.json"
        _write_json(paths["dirty_coco"], _coco(plan, f"{name}_dirty.png", dboxes, w_px, h_px))
    return paths


def generate(n: int, size: str, seed: int, out_dir: Path, raster: bool = True) -> list[str]:
    """Сгенерировать n чертежей; имена ``{size}_{i:03d}``, seed чертежа i = seed * 1000 + i."""
    names = []
    for i in range(n):
        name = f"{size}_{i:03d}"
        plan = generate_plan(seed * 1000 + i, size)  # type: ignore[arg-type]
        export_plan(plan, Path(out_dir), name, raster=raster)
        names.append(name)
    return names


__all__ = ["PlanInstance", "export_plan", "generate", "make_dirty", "plan_to_gt", "plan_to_svg"]
