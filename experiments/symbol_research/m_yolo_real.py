"""H2: YOLO на реальной разметке (протокол B).

Фолды: worcester (тест 01 / 02 / 03, обучение на двух других планах), usc (тест левая / правая
половина листа 05 по x=3500, обучение на другой половине). В обучение каждого фолда добавлен лист
легенды семейства с легендными эталонами как размеченными экземплярами.

Запуск:
    .venv/bin/python m_yolo_real.py --fold wor01 [--epochs 10]   # обучение, сырые предсказания
    .venv/bin/python m_yolo_real.py --finalize                   # выбор conf, запись yolo_real*
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import common
import yolo_common as yc

# fold -> (семейство, тестовый лист (префикс файла), часть листа или None)
FOLDS = {
    "wor01": ("worcester", "01_", None),
    "wor02": ("worcester", "02_", None),
    "wor03": ("worcester", "03_", None),
    "uscL": ("usc", "05_", "left"),
    "uscR": ("usc", "05_", "right"),
}
CONF_DEFAULT = (0.25, 0.5)
RAW_CONF = 0.05
RAW_DIR = yc.YOLO_DATA / "preds_raw" / "yolo_real"
CURVE_DIR = yc.YOLO_DATA / "curves" / "yolo_real"
SPLIT_X = int(common.USC_SPLIT_X)


def _boxes(sheet: common.Sheet, names: list[str], items: list[common.Box]) -> list[yc.Box5]:
    idx = {n: i for i, n in enumerate(names)}
    return [(idx[b.legend_id], *b.bbox) for b in items if b.legend_id in idx]


def build_fold(
    fold: str, seed: int = 0, jitter: int = 2, legend_jitter: int = 3
) -> tuple[Path, dict]:
    family, test_prefix, part = FOLDS[fold]
    sheets = common.load_sheets()
    names = yc.class_names(family)
    rng = random.Random(seed)
    root = yc.YOLO_DATA / f"real_{fold}"
    if root.exists():
        shutil.rmtree(root)

    leg_sheet, leg_items = common.legend_for(family, sheets)
    plans = [s for s in common.labeled_plans(sheets) if s.family == family]
    test_sheet = next(s for s in plans if s.file_name.startswith(test_prefix))

    sources: list[yc.Source] = []
    for s in plans:
        gray = s.load_gray()
        h, w = gray.shape
        if s.file_name == test_sheet.file_name:
            if part is None:
                continue  # тестовый лист целиком вне обучения
            # половина листа: обучение только на тайлах, целиком лежащих в другой половине
            region = (SPLIT_X, 0, w, h) if part == "left" else (0, 0, SPLIT_X, h)
        else:
            region = (0, 0, w, h)
        sources.append(
            yc.Source(
                Path(s.file_name).stem[:2],
                gray,
                _boxes(s, names, s.objects),
                region,
                jitter_per_obj=jitter,
            )
        )
    lg = leg_sheet.load_gray()
    lsrc = yc.Source(
        "leg",
        lg,
        _boxes(leg_sheet, names, leg_items),
        (0, 0, lg.shape[1], lg.shape[0]),
        jitter_per_obj=legend_jitter,
        val_frac=0.0,
    )

    stats = {
        "fold": fold,
        "family": family,
        "test": test_sheet.file_name,
        "part": part,
        "train_tiles": 0,
        "val_tiles": 0,
        "train_images": 0,
        "val_images": 0,
        "train_boxes": 0,
        "cut_boxes": 0,
        "sources": {},
    }
    leak_rects = []
    if part == "left":
        leak_rects = [(0, 0, SPLIT_X, 10**9)]
    elif part == "right":
        leak_rects = [(SPLIT_X, 0, 10**9, 10**9)]
    for src in sources + [lsrc]:
        vw = yc.choose_val_windows(src, rng) if src is not lsrc else []
        train_tiles, val_tiles = yc.build_source_tiles(src, rng, vw)
        # проверка утечки: тайлы обучения не пересекают тестовую область тестового листа
        for x, y, _k in train_tiles:
            for lr in leak_rects:
                if src.name == test_prefix[:2]:
                    assert not yc._intersects((x, y, x + yc.TILE, y + yc.TILE), lr), (
                        fold,
                        src.name,
                        x,
                        y,
                    )
        assert not (src.name == test_prefix[:2] and part is None)
        ni, nb, nc = yc.write_tiles(root, "train", src, train_tiles)
        vi, _vb, _vc = yc.write_tiles(root, "val", src, val_tiles, rotations=(0,))
        stats["train_tiles"] += len(train_tiles)
        stats["val_tiles"] += len(val_tiles)
        stats["train_images"] += ni
        stats["val_images"] += vi
        stats["train_boxes"] += nb
        stats["cut_boxes"] += nc
        stats["sources"][src.name] = dict(
            train_tiles=len(train_tiles), val_tiles=len(val_tiles), images=ni, boxes=nb
        )
    yc.write_yaml(root, names)
    (root / "stats.json").write_text(json.dumps(stats, indent=1))
    return root, stats


def run_fold(
    fold: str, epochs: int, patience: int, weights: str, time_hours: float | None, batch: int
) -> None:
    family, test_prefix, part = FOLDS[fold]
    names = yc.class_names(family)
    root, stats = build_fold(fold)
    print(json.dumps(stats), flush=True)
    t0 = time.time()
    best = yc.train_yolo(
        root / "data.yaml", f"real_{fold}", epochs, patience, weights, batch, time_hours
    )
    train_min = (time.time() - t0) / 60
    from ultralytics import YOLO

    model = YOLO(str(best))
    # кривая F1(conf) на валидации фолда (тест не используется)
    vi, vb = yc.load_val(root)
    curve = yc.val_f1_curve(model, vi, vb)
    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    (CURVE_DIR / f"{fold}.json").write_text(json.dumps({str(k): v for k, v in curve.items()}))
    # сырые предсказания теста при низком conf
    sheet = next(s for s in common.labeled_plans() if s.file_name.startswith(test_prefix))
    gray = sheet.load_gray()
    h, w = gray.shape
    region = None
    if part == "left":
        region = (0, 0, min(w, SPLIT_X + yc.OVERLAP), h)
    elif part == "right":
        region = (SPLIT_X - yc.OVERLAP, 0, w, h)
    t1 = time.time()
    preds = yc.predict_sheet(model, gray, conf=RAW_CONF, names=names, region=region)
    stem = Path(sheet.file_name).stem + (f"__{part}" if part else "")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / f"{stem}.json").write_text(json.dumps(preds))
    info = dict(
        fold=fold,
        train_min=train_min,
        predict_s=time.time() - t1,
        n_raw=len(preds),
        weights=str(best),
    )
    (CURVE_DIR / f"{fold}_info.json").write_text(json.dumps(info))
    print(json.dumps(info), flush=True)


def finalize() -> None:
    """Pooled conf по валидации фолдов семейства (F1-максимум), запись yolo_real (выбранный conf),
    yolo_real_c25 и yolo_real_c50."""
    chosen: dict[str, float] = {}
    for fam in ("worcester", "usc"):
        folds = [f for f, v in FOLDS.items() if v[0] == fam and (CURVE_DIR / f"{f}.json").exists()]
        pooled: dict[float, list[int]] = {}
        for f in folds:
            for k, v in json.loads((CURVE_DIR / f"{f}.json").read_text()).items():
                acc = pooled.setdefault(float(k), [0, 0, 0])
                acc[0] += v[3]
                acc[1] += v[4]
                acc[2] += v[5]
        curve = {}
        for k, (tp, npd, ng) in pooled.items():
            p, r, f1 = common.prf(tp, npd, ng)
            curve[k] = (p, r, f1, tp, npd, ng)
        if curve:
            chosen[fam] = yc.best_conf(curve)
            print(
                fam,
                "pooled val F1 curve:",
                {k: round(v[2], 3) for k, v in sorted(curve.items())},
                "-> conf",
                chosen[fam],
            )
    (CURVE_DIR / "chosen_conf.json").write_text(json.dumps(chosen))
    for tag, fixed in (("yolo_real", None), ("yolo_real_c25", 0.25), ("yolo_real_c50", 0.5)):
        for f in RAW_DIR.glob("*.json"):
            fam = common.family_of(f.name)
            conf = fixed if fixed is not None else chosen[fam]
            preds = [p for p in json.loads(f.read_text()) if p["score"] >= conf]
            out = common.OUT / tag
            out.mkdir(parents=True, exist_ok=True)
            (out / f.name).write_text(json.dumps(preds))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", choices=list(FOLDS))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--weights", default="yolo11s.pt")
    ap.add_argument(
        "--time", type=float, default=None, help="бюджет обучения в часах (переопределяет epochs)"
    )
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    a = ap.parse_args()
    if a.finalize:
        finalize()
    elif a.build_only:
        print(json.dumps(build_fold(a.fold)[1], indent=1))
    else:
        run_fold(a.fold, a.epochs, a.patience, a.weights, a.time, a.batch)
