"""Финальные прогоны лучшего метода (шаблоны m_template) с автоматическим разбором легенды.

    uv run --project ../.. python m_final.py autoleg [--sheets 01,02,03,05]
    uv run --project ../.. python m_final.py final [--sheets 01,...,10] [--reuse]

autoleg (протокол A без ручной легенды): эталоны это вырезки по рамкам из
outputs/research/legend_auto/<family>.json; автоматические id переименовываются в ручные
по IoU рамок на листе легенды >= 0.3, остальные остаются автоматическими. Результат:
outputs/research/template_autoleg/ (оценка: evaluate.py template_autoleg [--strict]).

final: все 10 планов; эталоны из ручной легенды (worcester, usc), иначе из автоматической.
Выход outputs/research/final/: <stem>.json, <stem>_overlay.png, <stem>_by_class.png,
<stem>_meta.json (время), summary.json. Параметры detect по умолчанию, порог 0.75.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import time
from pathlib import Path

import common
import m_template
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WORKERS = 6  # на машине параллельно идет обучение, поток-лимит
MATCH_IOU = 0.3  # автоматический эталон -> ручной
OVERLAY_W = 3000
CELL = 104  # сторона ячейки контрольного листа
CROP_PAD = 15
N_TOP = 12
N_LOW = 2
LABEL_W = 170
MAX_UPSCALE = 4.0

FINAL = common.OUT / "final"
MANUAL_FAMILIES = {"worcester", "usc"}


# ------------------------------------------------------------------ эталоны


def auto_items(family: str) -> list[dict]:
    return json.loads((common.OUT / "legend_auto" / f"{family}.json").read_text())


def auto_crops(family: str, pad: int = 0) -> dict[str, np.ndarray]:
    """legend_id (авто) -> вырезка с листа легенды, как common.legend_crops."""
    sheet, _ = common.legend_for(family)
    img = sheet.load_gray()
    out = {}
    for it in auto_items(family):
        x, y, w, h = it["bbox"]
        x0, y0 = max(0, int(round(x)) - pad), max(0, int(round(y)) - pad)
        x1, y1 = int(round(x + w)) + pad, int(round(y + h)) + pad
        out[it["legend_id"]] = img[y0:y1, x0:x1].copy()
    return out


def auto_to_manual(family: str) -> dict[str, str]:
    """auto id -> ручной id (лучший IoU >= MATCH_IOU); несопоставленные отсутствуют."""
    _, manual = common.legend_for(family)
    out = {}
    for it in auto_items(family):
        best, bid = MATCH_IOU, None
        for m in manual:
            v = common.iou_xywh(it["bbox"], m.bbox)
            if v >= best:
                best, bid = v, m.legend_id
        if bid is not None:
            out[it["legend_id"]] = bid
    return out


def run_detect(gray: np.ndarray, crops: dict[str, np.ndarray]) -> list[dict]:
    return m_template.detect(gray, crops, workers=WORKERS)


def slim(preds: list[dict]) -> list[dict]:
    return [dict(bbox=d["bbox"], legend_id=d["legend_id"], score=d["score"]) for d in preds]


# ------------------------------------------------------------------ часть 1


def cmd_autoleg(sheets_arg: str) -> None:
    want = set(sheets_arg.split(","))
    for s in common.labeled_plans():
        if s.file_name[:2] not in want:
            continue
        crops = auto_crops(s.family)
        mp = auto_to_manual(s.family)
        t0 = time.time()
        preds = slim(run_detect(s.load_gray(), crops))
        dt = time.time() - t0
        for d in preds:
            d["legend_id"] = mp.get(d["legend_id"], d["legend_id"])
        path = common.save_preds("template_autoleg", s.file_name, preds)
        print(
            f"{s.file_name}: {len(crops)} эталонов, {len(mp)} сопоставлено, "
            f"{len(preds)} предсказаний, {dt:.0f} s -> {path}",
            flush=True,
        )


# ------------------------------------------------------------------ визуализация


def _font(size: int) -> ImageFont.ImageFont:
    for p in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def class_colors(ids: list[str]) -> dict[str, tuple[int, int, int]]:
    out = {}
    for i, k in enumerate(sorted(ids)):
        h = (i * 0.618033988) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.95, 0.85)
        out[k] = (int(r * 255), int(g * 255), int(b * 255))
    return out


def draw_overlay(gray: np.ndarray, preds: list[dict], path: Path) -> None:
    h, w = gray.shape
    k = OVERLAY_W / w
    im = Image.fromarray(gray).convert("RGB").resize((OVERLAY_W, round(h * k)), Image.LANCZOS)
    dr = ImageDraw.Draw(im)
    colors = class_colors(sorted({d["legend_id"] for d in preds}))
    f = _font(13)
    for d in preds:
        x, y, bw, bh = (v * k for v in d["bbox"])
        c = colors[d["legend_id"]]
        dr.rectangle([x - 2, y - 2, x + bw + 2, y + bh + 2], outline=c, width=2)
        dr.text((x, max(0, y - 16)), d["legend_id"], fill=c, font=f)
    im.save(path)


def _cell(img: np.ndarray, label: str, frame: bool = False) -> Image.Image:
    im = Image.fromarray(img).convert("RGB")
    sc = min((CELL - 4) / im.width, (CELL - 4) / im.height, MAX_UPSCALE)
    im = im.resize((max(1, round(im.width * sc)), max(1, round(im.height * sc))), Image.LANCZOS)
    cell = Image.new("RGB", (CELL, CELL + 16), "white")
    cell.paste(im, ((CELL - im.width) // 2, (CELL - im.height) // 2))
    dr = ImageDraw.Draw(cell)
    if frame:
        dr.rectangle([0, 0, CELL - 1, CELL - 1], outline=(0, 0, 200), width=3)
    else:
        dr.rectangle([0, 0, CELL - 1, CELL - 1], outline=(200, 200, 200), width=1)
    dr.text((3, CELL + 2), label, fill=(0, 0, 0), font=_font(12))
    return cell


def _inst_crop(gray: np.ndarray, bbox: list[float]) -> np.ndarray:
    x, y, w, h = bbox
    x0, y0 = max(0, int(x) - CROP_PAD), max(0, int(y) - CROP_PAD)
    x1, y1 = (
        min(gray.shape[1], int(x + w) + CROP_PAD + 1),
        min(gray.shape[0], int(y + h) + CROP_PAD + 1),
    )
    return gray[y0:y1, x0:x1]


def draw_by_class(
    gray: np.ndarray, preds: list[dict], crops: dict[str, np.ndarray], path: Path
) -> None:
    by: dict[str, list[dict]] = {}
    for d in preds:
        by.setdefault(d["legend_id"], []).append(d)
    classes = sorted(by, key=lambda c: -len(by[c]))
    ncell = 1 + N_TOP + N_LOW + 1  # эталон, топ, разделитель, самые низкие
    row_h = CELL + 24
    W = LABEL_W + ncell * (CELL + 4)
    im = Image.new("RGB", (W, max(1, row_h * len(classes))), "white")
    dr = ImageDraw.Draw(im)
    for r, c in enumerate(classes):
        y = r * row_h
        lst = sorted(by[c], key=lambda d: -d["score"])
        dr.text((6, y + 10), c, fill=(0, 0, 0), font=_font(20))
        dr.text((6, y + 40), f"найдено {len(lst)}", fill=(0, 0, 0), font=_font(15))
        x = LABEL_W
        if c in crops:
            im.paste(_cell(crops[c], "эталон", frame=True), (x, y))
        x += CELL + 4
        top = lst[:N_TOP]
        low = lst[N_TOP:][-N_LOW:] if len(lst) > N_TOP else []
        for i, d in enumerate(top):
            im.paste(
                _cell(_inst_crop(gray, d["bbox"]), f"{d['score']:.3f}"), (x + i * (CELL + 4), y)
            )
        x += (N_TOP + 1) * (CELL + 4)
        for i, d in enumerate(low):
            im.paste(
                _cell(_inst_crop(gray, d["bbox"]), f"низ {d['score']:.3f}"), (x + i * (CELL + 4), y)
            )
        dr.line([(0, y + row_h - 3), (W, y + row_h - 3)], fill=(150, 150, 150), width=1)
    im.save(path)


# ------------------------------------------------------------------ часть 2


def cmd_final(sheets_arg: str, reuse: bool) -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    want = set(sheets_arg.split(","))
    allsheets = common.load_sheets()
    summary_path = FINAL / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for s in allsheets:
        if s.is_legend or s.file_name[:2] not in want:
            continue
        stem = Path(s.file_name).stem
        manual = s.family in MANUAL_FAMILIES
        crops = common.legend_crops(s.family, 0) if manual else auto_crops(s.family)
        gray = s.load_gray()
        jp, mp = FINAL / f"{stem}.json", FINAL / f"{stem}_meta.json"
        if reuse and jp.exists() and mp.exists():
            preds = json.loads(jp.read_text())
            dt = json.loads(mp.read_text())["time_s"]
        else:
            t0 = time.time()
            preds = slim(run_detect(gray, crops))
            dt = time.time() - t0
            jp.write_text(json.dumps(preds))
            mp.write_text(
                json.dumps(
                    dict(time_s=dt, n_templates=len(crops), legend="manual" if manual else "auto")
                )
            )
        draw_overlay(gray, preds, FINAL / f"{stem}_overlay.png")
        draw_by_class(gray, preds, crops, FINAL / f"{stem}_by_class.png")
        counts: dict[str, int] = {}
        for d in preds:
            counts[d["legend_id"]] = counts.get(d["legend_id"], 0) + 1
        summary[stem] = dict(
            legend="manual" if manual else "auto",
            n_templates=len(crops),
            n_found=len(preds),
            n_classes_found=len(counts),
            time_s=round(dt, 1),
            by_class=dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        )
        summary_path.write_text(
            json.dumps(dict(sorted(summary.items())), indent=1, ensure_ascii=False)
        )
        print(
            f"{stem}: {len(crops)} эт., {len(preds)} найдено, {len(counts)} кл., {dt:.0f} s",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["autoleg", "final"])
    ap.add_argument("--sheets", default=None)
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    if a.cmd == "autoleg":
        cmd_autoleg(a.sheets or "01,02,03,05")
    else:
        cmd_final(a.sheets or ",".join(f"{i:02d}" for i in range(1, 11)), a.reuse)


if __name__ == "__main__":
    main()
