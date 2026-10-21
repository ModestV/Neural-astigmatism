"""Базовая линия H0: растровая ветка SymSpot на составных листах (план + полоса эталонов).

SymSpot ищет символы по эталонам с того же чертежа, а легенда здесь на отдельном листе.
Поэтому к плану справа приклеивается белая полоса с вырезками эталонов семейства, а эталоны
задаются в COCO рамками вклеек. Экземпляры с центром в полосе отбрасываются.

    python m_symspot.py [--scale 1.0] [--only SUBSTR] [--keep-existing]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import common
import numpy as np
from PIL import Image

WORK = common.OUT / "symspot_work"
GAP = 40  # промежуток между вклейками, px
MARGIN = 40  # отступ полосы от плана и от краев, px
PAD = 4  # поле вокруг вырезки эталона


def build_composite(sheet: common.Sheet) -> tuple[np.ndarray, list[dict], int]:
    """Составной лист (серый uint8) и список вклеек [{legend_id, bbox=[x,y,w,h]}], x0 полосы."""
    plan = sheet.load_gray()
    h, w = plan.shape
    crops = common.legend_crops(sheet.family, pad=PAD)
    items = sorted(crops.items(), key=lambda kv: (len(kv[0]), kv[0]))
    # колонки заполняются сверху вниз; при выходе за высоту плана начинается новая
    cols: list[list[tuple[str, np.ndarray]]] = [[]]
    used = MARGIN
    for lid, c in items:
        if used + c.shape[0] > h - MARGIN and cols[-1]:
            cols.append([])
            used = MARGIN
        cols[-1].append((lid, c))
        used += c.shape[0] + GAP
    col_w = [max(c.shape[1] for _, c in col) for col in cols]
    strip_w = MARGIN + sum(col_w) + GAP * (len(cols) - 1) + MARGIN
    canvas = np.full((h, w + strip_w), 255, np.uint8)
    canvas[:, :w] = plan
    boxes = []
    x = w + MARGIN
    for col, cw in zip(cols, col_w, strict=True):
        y = MARGIN
        for lid, c in col:
            ch, cw_ = c.shape
            canvas[y : y + ch, x : x + cw_] = c
            boxes.append(dict(legend_id=lid, bbox=[float(x), float(y), float(cw_), float(ch)]))
            y += ch + GAP
        x += cw + GAP
    return canvas, boxes, w


def write_inputs(sheet: common.Sheet, scale: float, d: Path) -> tuple[Path, Path, int]:
    canvas, boxes, strip_x = build_composite(sheet)
    img = Image.fromarray(canvas)
    if scale != 1.0:
        img = Image.fromarray(canvas).resize(
            (round(canvas.shape[1] * scale), round(canvas.shape[0] * scale)), Image.LANCZOS
        )
    stem = Path(sheet.file_name).stem
    png = d / f"{stem}_composite.png"
    img.save(png)
    anns = []
    for i, b in enumerate(boxes, 1):
        anns.append(
            dict(
                id=i,
                image_id=1,
                category_id=1,
                bbox=[v * scale for v in b["bbox"]],
                area=0,
                iscrowd=0,
                attributes=dict(legend_id=b["legend_id"]),
            )
        )
    coco = dict(
        categories=[dict(id=1, name="legend_item")],
        images=[dict(id=1, width=img.width, height=img.height, file_name=png.name)],
        annotations=anns,
    )
    q = d / f"{stem}_queries.json"
    q.write_text(json.dumps(coco))
    return png, q, strip_x


def run_symspot(png: Path, q: Path, out: Path) -> float:
    t0 = time.time()
    subprocess.run(
        ["uv", "run", "symspot", "run", str(png), "--queries", str(q)]
        + ["--out", str(out), "--formats", ""],
        cwd=common.ROOT,
        check=True,
    )
    return time.time() - t0


def convert(result: Path, scale: float, strip_x: int) -> tuple[list[dict], dict]:
    """result.json -> предсказания [x,y,w,h] в координатах плана.

    Возвращает (preds, статистика): сколько экземпляров в полосе (из них is_query, то есть сами
    эталоны, и найденные поиском копии) и распределение статусов экземпляров на плане.
    """
    res = json.loads(result.read_text())
    preds: list[dict] = []
    stat = dict(strip_query=0, strip_other=0, status={})
    for inst in res["instances"]:
        x0, y0, x1, y1 = (v / scale for v in inst["bbox"])
        if (x0 + x1) / 2 >= strip_x:
            stat["strip_query" if inst.get("is_query") else "strip_other"] += 1
            continue
        stat["status"][inst["status"]] = stat["status"].get(inst["status"], 0) + 1
        preds.append(
            dict(
                bbox=[x0, y0, x1 - x0, y1 - y0], legend_id=str(inst["legend_id"]),
                score=float(inst.get("score", 1.0)),
            )
        )
    return preds, stat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--reuse", action="store_true", help="не запускать, если result.json есть")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    for s in common.labeled_plans():
        if a.only not in s.file_name:
            continue
        stem = Path(s.file_name).stem
        out = WORK / f"{stem}_out"
        png, q, strip_x = write_inputs(s, a.scale, WORK)
        result = out / "result.json"
        dt = 0.0
        if not (a.reuse and result.exists()):
            dt = run_symspot(png, q, out)
        preds, stat = convert(result, a.scale, strip_x)
        common.save_preds("symspot", s.file_name, preds)
        print(
            f"{s.file_name}: composite {Image.open(png).size}, scale {a.scale}, {dt:.0f} s, "
            f"pred {len(preds)}, strip {stat}, legend {len(common.legend_crops(s.family))}",
            flush=True,
        )


if __name__ == "__main__":
    main()
