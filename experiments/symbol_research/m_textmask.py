"""Фильтр текстовой маски: отбросить находки, лежащие внутри строк текста плана.

Строки текста находятся той же геометрией, что в разборе легенды (legend_parse.build_blobs:
выровненные мелкие компоненты >= 3 в ряд). Находка отбрасывается, если доля ее рамки,
покрытая рамками текстовых блоков, не меньше COVER. Разметка не используется.

    uv run --project ../.. python m_textmask.py SRC_METHOD DST_METHOD [--all]
(--all: все планы, иначе только размеченные)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import common
import legend_parse
import numpy as np

COVER = 0.5


def text_boxes(gray: np.ndarray) -> list[tuple[float, float, float, float]]:
    blobs, _ = legend_parse.build_blobs(gray, legend_parse.Params())
    return [(b.x0, b.y0, b.x1, b.y1) for b in blobs if b.kind == "text"]


def covered(box: list[float], tb: np.ndarray) -> float:
    x, y, w, h = box
    if len(tb) == 0:
        return 0.0
    iw = np.clip(np.minimum(x + w, tb[:, 2]) - np.maximum(x, tb[:, 0]), 0, None)
    ih = np.clip(np.minimum(y + h, tb[:, 3]) - np.maximum(y, tb[:, 1]), 0, None)
    # блоки текста почти не перекрываются, сумма пересечений близка к площади объединения
    return float(min(1.0, (iw * ih).sum() / max(w * h, 1e-6)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    for s in common.load_sheets():
        if s.is_legend or not (a.all or s.labeled):
            continue
        stem = Path(s.file_name).stem
        src = common.OUT / a.src / f"{stem}.json"
        if not src.exists():
            continue
        preds = json.loads(src.read_text())
        t0 = time.time()
        tb = np.array(text_boxes(s.load_gray()), dtype=np.float64).reshape(-1, 4)
        kept = [p for p in preds if covered(p["bbox"], tb) < COVER]
        common.save_preds(a.dst, s.file_name, kept)
        dt = time.time() - t0
        print(f"{stem}: текстовых блоков {len(tb)}, {len(preds)} -> {len(kept)}, {dt:.0f} s")


if __name__ == "__main__":
    main()
