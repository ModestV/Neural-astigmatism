"""Предсказания шаблонного метода для неразмеченных планов (для дозаполнения разметки в CVAT).

Эталоны: ручная легенда, если размечена, иначе автоматический разбор (outputs/research/legend_auto).
Выход: outputs/research/fill/<stem>.json
"""

from __future__ import annotations

import json
import sys

import common
import m_template


def auto_legend(family: str) -> dict:
    sheet, _ = common.legend_for(family)
    img = sheet.load_gray()
    items = json.loads((common.OUT / "legend_auto" / f"{family}.json").read_text())
    out = {}
    for it in items:
        x, y, w, h = (int(round(v)) for v in it["bbox"])
        out[it["legend_id"]] = img[y : y + h, x : x + w].copy()
    return out


def main() -> None:
    sheets = common.load_sheets()
    want = sys.argv[1:]
    for s in sheets:
        if s.is_legend or s.labeled or (want and not any(s.file_name.startswith(w) for w in want)):
            continue
        _, items = common.legend_for(s.family, sheets)
        tpl = common.legend_crops(s.family) if items else auto_legend(s.family)
        preds = m_template.detect(s.load_gray(), tpl)
        out = [dict(bbox=p["bbox"], legend_id=p["legend_id"], score=p["score"]) for p in preds]
        common.save_preds("fill", s.file_name, out)
        print(s.file_name, len(out), flush=True)


if __name__ == "__main__":
    main()
