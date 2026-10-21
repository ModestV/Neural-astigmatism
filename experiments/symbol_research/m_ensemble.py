"""Ансамбль двух методов по предсказаниям (без переобучения).

    python m_ensemble.py A B OUT --mode union|inter [--iou 0.5]

union: объединение, совпадающие рамки одного класса (IoU >= iou) сливаются, score = максимум;
inter: остаются только рамки, подтвержденные вторым методом с тем же классом.
Работает с файлами <stem>.json и <stem>__left/right.json (протокол B).
"""

from __future__ import annotations

import argparse
import json

import common


def merge(a: list[dict], b: list[dict], mode: str, iou: float) -> list[dict]:
    def match(p: dict, pool: list[dict]) -> bool:
        return any(
            q["legend_id"] == p["legend_id"] and common.iou_xywh(p["bbox"], q["bbox"]) >= iou
            for q in pool
        )

    if mode == "inter":
        return [p for p in a if match(p, b)]
    out = list(a) + [q for q in b if not match(q, a)]
    return common.nms(out, 0.3, class_agnostic=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("out")
    ap.add_argument("--mode", default="union", choices=["union", "inter"])
    ap.add_argument("--iou", type=float, default=0.5)
    x = ap.parse_args()
    for f in sorted((common.OUT / x.a).glob("*.json")):
        if f.name.startswith("metrics"):
            continue
        g = common.OUT / x.b / f.name
        if not g.exists():
            continue
        res = merge(json.loads(f.read_text()), json.loads(g.read_text()), x.mode, x.iou)
        (common.OUT / x.out).mkdir(parents=True, exist_ok=True)
        (common.OUT / x.out / f.name).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
