"""template_fs: H1 с несколькими примерами с плана (протокол B).

    uv run --project ../.. python m_template_fs.py [--sheets 01,02,03,05] [--k 3] [--seed 0]

К эталону легенды каждого класса добавляются до k вырезок реальных экземпляров этого класса из
ОБУЧАЮЩЕЙ области (случайный выбор, seed=0). Обучающая область:
  - 01, 02, 03 (Worcester): экземпляры из двух остальных листов семейства;
  - 05 (USC): левая половина (x < USC_SPLIT_X) берет экземпляры из правой половины, правая из левой;
    лист 05 обрабатывается двумя прогонами по половинам (с полем 150 px), результат пишется в
    <stem>__left.json и <stem>__right.json.
Дополнительные шаблоны имеют ключ "<класс>#fsN" и в предсказаниях неотличимы от класса.
Поиск, постобработка и порог те же, что в m_template.detect.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import common
import m_template
import numpy as np

METHOD = "template_fs"
MARGIN = 150  # поле вокруг половины листа, px


def pick_examples(
    pools: list[tuple[common.Sheet, list[common.Box]]], k: int, seed: int
) -> dict[str, list[tuple[common.Sheet, common.Box]]]:
    """До k случайных (seed) экземпляров на класс из обучающих пулов."""
    by_class: dict[str, list[tuple[common.Sheet, common.Box]]] = {}
    for sh, objs in pools:
        for o in objs:
            by_class.setdefault(o.legend_id, []).append((sh, o))
    rng = random.Random(seed)
    out = {}
    for cls in sorted(by_class, key=lambda c: (len(c), c)):
        items = sorted(by_class[cls], key=lambda t: (t[0].file_name, t[1].bbox[1], t[1].bbox[0]))
        out[cls] = rng.sample(items, min(k, len(items)))
    return out


def crop_templates(
    examples: dict[str, list[tuple[common.Sheet, common.Box]]], grays: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    out = {}
    for cls, items in examples.items():
        for i, (sh, o) in enumerate(items):
            g = grays[sh.file_name]
            x, y, w, h = o.bbox
            x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
            x1, y1 = min(g.shape[1], int(round(x + w))), min(g.shape[0], int(round(y + h)))
            if x1 - x0 >= 6 and y1 - y0 >= 6:
                out[f"{cls}#fs{i}"] = g[y0:y1, x0:x1].copy()
    return out


def run_sheet(s: common.Sheet, plans: list[common.Sheet], k: int, seed: int, **params) -> None:
    legend = common.legend_crops(s.family)
    grays: dict[str, np.ndarray] = {}

    def gray(sh: common.Sheet) -> np.ndarray:
        if sh.file_name not in grays:
            grays[sh.file_name] = sh.load_gray()
        return grays[sh.file_name]

    others = [p for p in plans if p.family == s.family and p.file_name != s.file_name]
    img = gray(s)
    if others:  # Worcester: обучение на остальных листах семейства
        pools = [(p, p.objects) for p in others]
        for p in others:
            gray(p)
        ex = pick_examples(pools, k, seed)
        tpl = {**legend, **crop_templates(ex, grays)}
        t0 = time.time()
        preds = m_template.detect(img, tpl, **params)
        save(s.file_name, preds)
        print(f"{s.file_name}: {len(tpl) - len(legend)} примеров, {len(preds)} предсказаний, "
              f"{time.time() - t0:.0f} s")
        return
    # единственный размеченный лист семейства: половины обучают друг друга
    for part, region, train in (
        ("left", common.HALF_LEFT, common.HALF_RIGHT),
        ("right", common.HALF_RIGHT, common.HALF_LEFT),
    ):
        pool = [o for o in s.objects if common.in_region(o.bbox, train)]
        ex = pick_examples([(s, pool)], k, seed)
        tpl = {**legend, **crop_templates(ex, grays | {s.file_name: img})}
        x0 = 0 if part == "left" else int(common.USC_SPLIT_X) - MARGIN
        x1 = int(common.USC_SPLIT_X) + MARGIN if part == "left" else img.shape[1]
        t0 = time.time()
        raw = m_template.detect(np.ascontiguousarray(img[:, x0:x1]), tpl, **params)
        preds = []
        for d in raw:
            b = [d["bbox"][0] + x0, *d["bbox"][1:]]
            if common.in_region(b, region):
                preds.append({**d, "bbox": b})
        save(Path(s.file_name).stem + f"__{part}.png", preds)
        print(f"{s.file_name}[{part}]: {len(tpl) - len(legend)} примеров, {len(preds)} "
              f"предсказаний, {time.time() - t0:.0f} s")


def save(name: str, preds: list[dict]) -> None:
    out = [dict(bbox=d["bbox"], legend_id=d["legend_id"], score=d["score"]) for d in preds]
    common.save_preds(METHOD, name, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", default="01,02,03,05")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    want = set(a.sheets.split(","))
    plans = common.labeled_plans()
    for s in plans:
        if s.file_name[:2] in want:
            run_sheet(s, plans, a.k, a.seed)


if __name__ == "__main__":
    main()
