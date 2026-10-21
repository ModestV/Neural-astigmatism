"""Проверка ручной доразметки на пропуски.

Для каждого класса листа до 2 принятых рамок (с самого плана) служат шаблонами; находки
с score >= thr, не совпавшие с принятыми рамками, выводятся сеткой для осмотра.

    uv run --project ../.. python relabel_check.py <stem> [--thr 0.7]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from PIL import Image, ImageDraw

import common
import label_tool
import m_template


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--thr", type=float, default=0.7)
    a = ap.parse_args()
    s = label_tool.sheet_by_stem(a.stem)
    ok = [b for b in label_tool.load(s) if b["status"] == "ok"]
    gray = s.load_gray()
    by = defaultdict(list)
    for b in ok:
        by[b["legend_id"]].append(b)
    tpl = {}
    for cls, bs in by.items():
        for i, b in enumerate(sorted(bs, key=lambda d: -d["score"])[:2]):
            x, y, w, h = (int(round(v)) for v in b["bbox"])
            tpl[f"{cls}#{i}"] = gray[y : y + h, x : x + w].copy()
    preds = m_template.detect(gray, tpl, score_thr=a.thr)
    miss = [p for p in preds if not any(common.iou_xywh(p["bbox"], b["bbox"]) > 0.3 for b in ok)]
    miss.sort(key=lambda p: -p["score"])
    print(f"{s.file_name}: шаблонов {len(tpl)}, находок {len(preds)}, без рамки {len(miss)}")
    img = Image.open(s.path).convert("RGB")
    cell = 150
    cv = Image.new("RGB", (cell * 8, cell * max(1, (len(miss) + 7) // 8)), "white")
    for k, p in enumerate(miss):
        x, y, w, h = p["bbox"]
        cx, cy = x + w / 2, y + h / 2
        c = img.crop((int(cx - 75), int(cy - 75), int(cx + 75), int(cy + 75)))
        d = ImageDraw.Draw(c)
        d.rectangle([x - cx + 75, y - cy + 75, x - cx + 75 + w, y - cy + 75 + h], outline="red")
        d.text((2, 2), f"{k} {p['legend_id']} {p['score']:.2f}", fill="red")
        cv.paste(c, ((k % 8) * cell, (k // 8) * cell))
        print(k, p["legend_id"], [round(v) for v in p["bbox"]], round(p["score"], 3))
    out = label_tool.VIEW / f"check_{a.stem}.png"
    cv.save(out)
    print(out)
    (label_tool.VIEW / f"check_{a.stem}.json").write_text(json.dumps(miss))


if __name__ == "__main__":
    main()
