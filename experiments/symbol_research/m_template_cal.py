"""Калибровка порогов эталонов по чужим чертежам (без разметки).

Идея: эталон прогоняется по планам ДРУГИХ комплектов, где символов этого комплекта, как правило,
нет. Если эталон массово срабатывает на чужом листе (линия, пустой прямоугольник, буква «C»),
он неспецифичен: его порог поднимается, пока среднее число чужих срабатываний на лист не станет
не больше MAX_FOREIGN; если это недостижимо, класс отключается. Настоящие общие символы
(например, «F» в квадрате есть и в worcester, и в wisd) дают единицы-десятки срабатываний и
порог не меняют. Разметка планов не используется; MAX_FOREIGN задан заранее.

Порог класса применяется к уже сохраненным предсказаниям шаблонного метода (score >= 0.75),
поэтому межклассовый NMS выполнен до калибровки (упрощение, отмечено в отчете).

    uv run --project ../.. python m_template_cal.py [--families worcester,usc,canada,wisd,va]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import common
import m_final
import m_template
import numpy as np

MAX_FOREIGN = 20.0  # допустимое среднее число срабатываний эталона на чужой лист
THR_GRID = np.round(np.arange(0.75, 1.0001, 0.01), 2)
N_FOREIGN = 3  # число чужих листов (по одному плану из разных других комплектов)
MAX_LEGEND = 3.0  # допустимое число срабатываний на своей легенде с забеленными значками
MASK_PAD = 12  # поле забеливания вокруг значков легенды, px
CAL = common.OUT / "template_cal"


def foreign_plans(family: str, sheets: list[common.Sheet]) -> list[common.Sheet]:
    """Первый по имени план каждого другого комплекта, не больше N_FOREIGN."""
    out, seen = [], set()
    for s in sorted(sheets, key=lambda s: s.file_name):
        if s.is_legend or s.family == family or s.family in seen:
            continue
        out.append(s)
        seen.add(s.family)
    return out[:N_FOREIGN]


def templates_for(family: str) -> dict[str, np.ndarray]:
    if family in m_final.MANUAL_FAMILIES:
        return common.legend_crops(family)
    return m_final.auto_crops(family)


def masked_legend(family: str, sheets: list[common.Sheet]) -> np.ndarray:
    """Лист легенды комплекта, где забелены все значки (ручные и найденные автоматически).

    Остаются описания, заголовки, таблицы, штамп: тот же шрифт и графика, что на планах
    комплекта, но без экземпляров символов.
    """
    sheet, items = common.legend_for(family, sheets)
    g = sheet.load_gray().copy()
    boxes = [it.bbox for it in items] + [d["bbox"] for d in m_final.auto_items(family)]
    for x, y, w, h in boxes:
        x0, y0 = max(0, int(x) - MASK_PAD), max(0, int(y) - MASK_PAD)
        g[y0 : int(y + h) + MASK_PAD, x0 : int(x + w) + MASK_PAD] = 255
    return g


def first_ok(mean_hits: np.ndarray, limit: float) -> float | None:
    ok = np.nonzero(mean_hits <= limit)[0]
    return float(THR_GRID[ok[0]]) if len(ok) else None


def calibrate_legend(family: str, sheets: list[common.Sheet]) -> dict:
    tpl = templates_for(family)
    counts = {k: np.zeros(len(THR_GRID)) for k in tpl}
    t0 = time.time()
    for c in m_template.detect_candidates(masked_legend(family, sheets), tpl, workers=6):
        counts[c["legend_id"]] += c["score"] >= THR_GRID
    res = dict(
        family=family,
        seconds=time.time() - t0,
        thr={k: first_ok(v, MAX_LEGEND) for k, v in counts.items()},
        legend_hits_at_075={k: float(v[0]) for k, v in counts.items()},
    )
    (CAL / f"cal_legend_{family}.json").write_text(json.dumps(res, indent=1))
    return res


def combine(a: dict[str, float | None], b: dict[str, float | None]) -> dict[str, float | None]:
    """Итоговый порог: максимум двух; класс отключен, если отключен хотя бы одним правилом."""
    out = {}
    for k in a:
        out[k] = None if a[k] is None or b.get(k, 0.75) is None else max(a[k], b.get(k, 0.75))
    return out


def calibrate(family: str, sheets: list[common.Sheet]) -> dict:
    tpl = templates_for(family)
    counts = {k: np.zeros(len(THR_GRID)) for k in tpl}
    used = foreign_plans(family, sheets)
    t0 = time.time()
    for s in used:
        cands = m_template.detect_candidates(s.load_gray(), tpl, workers=6)
        for c in cands:
            counts[c["legend_id"]] += c["score"] >= THR_GRID
    thr = {k: first_ok(cnt / max(len(used), 1), MAX_FOREIGN) for k, cnt in counts.items()}
    res = dict(
        family=family,
        foreign=[s.file_name for s in used],
        seconds=time.time() - t0,
        thr=thr,
        foreign_hits_at_075={k: float(v[0] / max(len(used), 1)) for k, v in counts.items()},
    )
    CAL.mkdir(parents=True, exist_ok=True)
    (CAL / f"cal_{family}.json").write_text(json.dumps(res, indent=1))
    return res


def apply(preds: list[dict], thr: dict[str, float | None]) -> list[dict]:
    out = []
    for p in preds:
        t = thr.get(p["legend_id"])
        if t is not None and p["score"] >= t:
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="worcester,usc,canada,wisd,va")
    ap.add_argument("--legend", action="store_true", help="добавить калибровку по своей легенде")
    a = ap.parse_args()
    suffix = "2" if a.legend else ""
    sheets = common.load_sheets()
    for fam in a.families.split(","):
        path = CAL / f"cal_{fam}.json"
        cal = json.loads(path.read_text()) if path.exists() else calibrate(fam, sheets)
        thr = cal["thr"]
        if a.legend:
            lpath = CAL / f"cal_legend_{fam}.json"
            if lpath.exists():
                lcal = json.loads(lpath.read_text())
            else:
                lcal = calibrate_legend(fam, sheets)
            thr = combine(thr, lcal["thr"])
        off = sorted(k for k, v in thr.items() if v is None)
        raised = {k: v for k, v in thr.items() if v is not None and v > 0.75}
        print(f"{fam}: чужие листы {cal['foreign']}, {cal['seconds']:.0f} s; "
              f"отключено {len(off)}: {off}; порог поднят {len(raised)}: {raised}")
        # Протокол A на размеченных планах (метод template_cal) и финальные листы (final_cal).
        for s in sheets:
            if s.is_legend or s.family != fam:
                continue
            stem = Path(s.file_name).stem
            final = json.loads((common.OUT / "final" / f"{stem}.json").read_text())
            kept = apply(final, thr)
            common.save_preds("final_cal" + suffix, s.file_name, kept)
            if s.labeled:
                base = common.load_preds("template", s.file_name)
                common.save_preds("template_cal" + suffix, s.file_name, apply(base, thr))
            print(f"  {stem}: {len(final)} -> {len(kept)}")


if __name__ == "__main__":
    main()
