"""Сравнение методов на одном фрагменте листа: разметка и предсказания каждого метода рядом.

Цвета: зеленый = верно (класс и IoU>=0.5), красный = ложное, оранжевый пунктир = пропуск.
Фрагмент выбирается автоматически: окно с наибольшим числом объектов разметки.

    uv run --project ../.. python m_compare.py [--sheets 02,05,06,09]
"""

from __future__ import annotations

import argparse
import json

import common
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIN_W, WIN_H = 1500, 1000
PANEL_W = 750
METHODS = [
    ("symspot", "SymSpot MVP"),
    ("template", "Шаблоны"),
    ("yolo_synth", "YOLO синтетика"),
    ("yolo_real", "YOLO реальная разметка"),
]
FALLBACK = {"template": "final_cal2"}  # для листов 04, 06-10 шаблоны после калибровки
OUT = common.OUT / "compare"


def font(size: int) -> ImageFont.ImageFont:
    for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def load(method: str, stem: str) -> list[dict] | None:
    d = common.OUT / method
    for name in (f"{stem}.json", f"{stem}__left.json"):
        if (d / name).exists():
            preds = json.loads((d / name).read_text())
            if name.endswith("__left.json") and (d / f"{stem}__right.json").exists():
                preds += json.loads((d / f"{stem}__right.json").read_text())
            return preds
    return None


def best_window(objs: list[common.Box], w: int, h: int) -> tuple[int, int]:
    best, xy = -1, (0, 0)
    cs = np.array([[o.bbox[0] + o.bbox[2] / 2, o.bbox[1] + o.bbox[3] / 2] for o in objs])
    for x in range(0, max(1, w - WIN_W), 100):
        for y in range(0, max(1, h - WIN_H), 100):
            n = int(
                (
                    (cs[:, 0] >= x)
                    & (cs[:, 0] < x + WIN_W)
                    & (cs[:, 1] >= y)
                    & (cs[:, 1] < y + WIN_H)
                ).sum()
            )
            if n > best:
                best, xy = n, (x, y)
    return xy


def match(preds: list[dict], gts: list[common.Box]) -> tuple[list[bool], list[bool]]:
    """Жадно по score, с учетом класса. Флаги tp для предсказаний и found для разметки."""
    used = [False] * len(gts)
    tp = [False] * len(preds)
    for i in sorted(range(len(preds)), key=lambda k: -preds[k]["score"]):
        best, bj = 0.5, -1
        for j, g in enumerate(gts):
            if used[j] or g.legend_id != preds[i]["legend_id"]:
                continue
            v = common.iou_xywh(preds[i]["bbox"], g.bbox)
            if v >= best:
                best, bj = v, j
        if bj >= 0:
            used[bj] = tp[i] = True
    return tp, used


def panel(gray: np.ndarray, x0: int, y0: int, title: str, boxes: list[tuple]) -> Image.Image:
    crop = Image.fromarray(gray[y0 : y0 + WIN_H, x0 : x0 + WIN_W]).convert("RGB")
    d = ImageDraw.Draw(crop)
    f = font(16)
    for bbox, color, label, dashed in boxes:
        x, y, w, h = bbox[0] - x0, bbox[1] - y0, bbox[2], bbox[3]
        if x + w < 0 or y + h < 0 or x > WIN_W or y > WIN_H:
            continue
        if dashed:
            for k in range(0, int(2 * (w + h)), 10):
                pts = _perimeter_point(x, y, w, h, k), _perimeter_point(x, y, w, h, k + 5)
                d.line(pts, fill=color, width=4)
        else:
            d.rectangle([x, y, x + w, y + h], outline=color, width=4)
        if label:
            d.text((x, y - 18), label, fill=color, font=f)
    scale = PANEL_W / WIN_W
    crop = crop.resize((PANEL_W, int(WIN_H * scale)), Image.LANCZOS)
    out = Image.new("RGB", (PANEL_W, crop.height + 40), "white")
    out.paste(crop, (0, 40))
    ImageDraw.Draw(out).text((8, 8), title, fill="black", font=font(20))
    return out


def _perimeter_point(x: float, y: float, w: float, h: float, t: float) -> tuple[float, float]:
    t %= 2 * (w + h)
    if t < w:
        return x + t, y
    if t < w + h:
        return x + w, y + t - w
    if t < 2 * w + h:
        return x + w - (t - w - h), y + h
    return x, y + h - (t - 2 * w - h)


def render(sheet: common.Sheet) -> None:
    gray = sheet.load_gray()
    stem = sheet.file_name.rsplit(".", 1)[0]
    x0, y0 = best_window(sheet.objects, sheet.width, sheet.height)
    inwin = lambda b: common.in_region(b, (x0, y0, x0 + WIN_W, y0 + WIN_H))  # noqa: E731
    gts = [g for g in sheet.objects if inwin(g.bbox)]
    known = common.known_classes(sheet.family)
    panels = [
        panel(
            gray,
            x0,
            y0,
            f"Разметка ({len(gts)})",
            [(g.bbox, (0, 90, 255), g.legend_id, False) for g in gts],
        )
    ]
    for method, name in METHODS:
        preds = load(method, stem)
        if preds is None and method in FALLBACK:
            preds, name = load(FALLBACK[method], stem), name + " (калибр.)"
        if preds is None:
            continue
        preds = [p for p in preds if inwin(p["bbox"]) and p["legend_id"] in known]
        tp, found = match(preds, gts)
        boxes = [
            (p["bbox"], (0, 170, 0) if t else (230, 0, 0), None if t else p["legend_id"], False)
            for p, t in zip(preds, tp, strict=True)
        ]
        boxes += [
            (g.bbox, (255, 140, 0), None, True) for g, f in zip(gts, found, strict=True) if not f
        ]
        n_tp = sum(tp)
        title = f"{name}: верно {n_tp}, ложных {len(preds) - n_tp}, пропусков {len(gts) - n_tp}"
        panels.append(panel(gray, x0, y0, title, boxes))
    cols = 2
    rows = (len(panels) + cols - 1) // cols
    ph = panels[0].height
    sheet_img = Image.new(
        "RGB", (PANEL_W * cols + 10 * (cols - 1), ph * rows + 10 * (rows - 1) + 50), "white"
    )
    ImageDraw.Draw(sheet_img).text(
        (8, 10),
        f"{sheet.file_name}, фрагмент x={x0}..{x0 + WIN_W}, y={y0}..{y0 + WIN_H}. "
        "Зеленый: верно, красный: ложное (подпись = класс), оранжевый пунктир: пропуск",
        fill="black",
        font=font(18),
    )
    for k, p in enumerate(panels):
        sheet_img.paste(p, ((k % cols) * (PANEL_W + 10), 50 + (k // cols) * (ph + 10)))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{stem}_compare.png"
    sheet_img.save(path)
    print(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", default="02,05,06,09")
    a = ap.parse_args()
    want = set(a.sheets.split(","))
    for s in common.labeled_plans():
        if s.file_name[:2] in want:
            render(s)


if __name__ == "__main__":
    main()
