"""Оценка метода по размеченным планам.

    python evaluate.py METHOD [--protocol A|B] [--strict] [--iou 0.5]

Читает outputs/research/METHOD/<stem>.json (для протокола B части листа 05 берутся из
<stem>__left.json / <stem>__right.json, если они есть, иначе из общего файла), печатает
таблицу, пишет metrics_<protocol>[_strict].json.

Протокол A («новый комплект»): метод видит только легенду; тест на всех 4 размеченных планах.
Протокол B («часть комплекта размечена»): тестовые области 01, 02, 03 целиком и обе
половины 05. Методы с обучением обязаны держать тестовую область вне обучения
(для 01-03 обучение на остальных двух листах Worcester и любых листах USC, для половин 05
обучение на другой половине и листах Worcester). Методы без обучения дают одни и те же
предсказания в обоих протоколах.

--set: human (по умолчанию, листы 01, 02, 03, 05 с ручной разметкой), agent (04, 06-10,
разметка агента с предразметкой шаблонным методом, смещена в его пользу), all.

По умолчанию считается метрика по известным классам (см. common.known_classes);
--strict учитывает все предсказания.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (
    HALF_LEFT,
    HALF_RIGHT,
    OUT,
    evaluate,
    known_classes,
    labeled_plans,
    load_sheets,
    summarize,
)


def _load(method: str, stem: str, part: str | None) -> list[dict] | None:
    d = OUT / method
    if part and (d / f"{stem}__{part}.json").exists():
        return json.loads((d / f"{stem}__{part}.json").read_text())
    if (d / f"{stem}.json").exists():
        return json.loads((d / f"{stem}.json").read_text())
    return None


def run(
    method: str,
    protocol: str = "A",
    strict: bool = False,
    iou: float = 0.5,
    quiet: bool = False,
    subset: str = "human",
) -> dict:
    sheets = load_sheets()
    res = {}
    for s in labeled_plans(sheets, subset):
        stem = Path(s.file_name).stem
        known = None if strict else known_classes(s.family, sheets, subset)
        parts = [(None, None)]
        if protocol == "B" and s.family == "usc":
            parts = [("left", HALF_LEFT), ("right", HALF_RIGHT)]
        for part, region in parts:
            preds = _load(method, stem, part)
            if preds is None:
                continue
            key = s.file_name if part is None else f"{s.file_name}[{part}]"
            res[key] = evaluate(preds, s.objects, iou, known=known, region=region)
    if not res:
        raise SystemExit(f"нет предсказаний для {method}")
    summ = summarize(res)
    summ["per_sheet_full"] = res
    sub = "" if subset == "human" else f"_{subset}"
    tag = f"metrics_{protocol}{'_strict' if strict else ''}{sub}.json"
    (OUT / method / tag).write_text(json.dumps(summ, indent=1, default=float))
    if not quiet:
        print(
            f"{method} [{protocol}{' strict' if strict else ''} {subset}]: "
            f"P={summ['P']:.3f} R={summ['R']:.3f} "
            f"F1={summ['F1']:.3f} F1_agn={summ['F1_agnostic']:.3f} mAP50={summ['mAP50']:.3f} "
            f"(tp {summ['tp']}/{summ['n_gt']}, pred {summ['n_pred']})"
        )
        for k, v in summ["sheets"].items():
            print(
                f"  {k[:46]:46s} P={v['P']:.3f} R={v['R']:.3f} F1={v['F1']:.3f} "
                f"F1agn={v['F1_agnostic']:.3f} mAP={v['mAP50']:.3f}"
            )
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("method")
    ap.add_argument("--protocol", default="A", choices=["A", "B"])
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--set", default="human", choices=["human", "agent", "all"])
    a = ap.parse_args()
    run(a.method, a.protocol, a.strict, a.iou, subset=a.set)
