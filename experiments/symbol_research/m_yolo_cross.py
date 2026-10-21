"""Протокол C: синтетика легенды целевого комплекта + реальная разметка ДРУГИХ комплектов.

Целевое семейство T. Обучающая выборка модели:
  - синтетика из легенды T (тот же генератор и объем, что в H3, m_yolo_synth.build_synth);
  - реальные тайлы 640 всех размеченных планов и листов легенд (legend_item как экземпляры)
    семейств, отличных от T, со своими классами; случайная выборка не более N_REAL тайлов
    (seed=0), каждый тайл в одном случайном повороте (k*90).
Пространство классов: объединение id пяти легенд. На планах T оставляются только классы T
(фильтр внутри NMS модели, `classes=`), на синтетической валидации T выбирается conf.

Отсутствие утечки: ни лист, ни легенда семейства T не попадают в реальную часть (assert по
семейству и по хешу пикселей); фоны синтетики берутся только с листов других семейств
(assert в m_yolo_synth.Pool); валидация только синтетическая; команда --check-leak ищет
тайлы обучения на планах T шаблонным совпадением.

Запуск:
    .venv/bin/python m_yolo_cross.py --target worcester [--epochs 14]
    .venv/bin/python m_yolo_cross.py --check-leak --target worcester
    .venv/bin/python m_yolo_cross.py --finalize --families worcester usc
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path

import common
import cv2
import m_yolo_synth as ms
import numpy as np
import yolo_common as yc

N_REAL = 2000
RAW_CONF = 0.05
RAW_DIR = yc.YOLO_DATA / "preds_raw" / "yolo_cross"
CURVE_DIR = yc.YOLO_DATA / "curves" / "yolo_cross"


def _digest(gray: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(gray[::7, ::7]).tobytes()).hexdigest()


def real_sources(
    target: str, sheets: list[common.Sheet], names: list[str]
) -> tuple[list[yc.Source], dict[str, str]]:
    """Источники реальных тайлов: планы и легенды всех семейств кроме target."""
    idx = {n: i for i, n in enumerate(names)}
    srcs: list[yc.Source] = []
    digests: dict[str, str] = {}
    for s in sheets:
        if s.family == target:
            continue
        items = s.legend_items if s.is_legend else s.objects
        boxes = [(idx[b.legend_id], *b.bbox) for b in items if b.legend_id in idx]
        if not boxes:
            continue
        gray = s.load_gray()
        h, w = gray.shape
        tag = f"leg{s.family}" if s.is_legend else Path(s.file_name).stem[:2]
        srcs.append(
            yc.Source(
                tag,
                gray,
                boxes,
                (0, 0, w, h),
                jitter_per_obj=3 if s.is_legend else 2,
                val_frac=0.0,
            )
        )
        digests[tag] = _digest(gray)
    return srcs, digests


def build_cross(
    target: str, n_synth: int = 2000, n_real: int = N_REAL, seed: int = 0
) -> tuple[Path, dict]:
    names = yc.all_class_names()
    root = yc.YOLO_DATA / f"cross_{target}"
    _, st = ms.build_synth(target, n_synth, seed=seed, names=names, root=root)
    sheets = common.load_sheets()

    # хеши листов целевого семейства: ни один источник реальных тайлов не должен совпасть
    tgt_hash = {_digest(s.load_gray()) for s in sheets if s.family == target}
    srcs, digests = real_sources(target, sheets, names)
    assert not (set(digests.values()) & tgt_hash), "лист целевого семейства в обучении"
    assert all(not t.startswith(f"leg{target}") for t in digests)

    rng = random.Random(seed)
    pool: list[tuple[int, int, int, str]] = []
    for si, src in enumerate(srcs):
        tiles, _ = yc.build_source_tiles(src, rng, [])
        pool += [(si, x, y, kind) for x, y, kind in tiles]
    n_avail = len(pool)
    pick = rng.sample(pool, min(n_real, n_avail))
    per_src: dict[str, int] = {}
    n_box = n_cut = 0
    for si, x, y, kind in pick:
        src = srcs[si]
        img = yc.crop_tile(src.gray, x, y)
        ins, cut = yc.boxes_in_tile(src.boxes, x, y)
        k = rng.randrange(4)
        ri, rb = yc.rotate_tile(img, ins, k)
        yc.write_sample(root, "train", f"real_{src.name}_{x}_{y}_{kind}_r{k * 90}", ri, rb)
        per_src[src.name] = per_src.get(src.name, 0) + 1
        n_box += len(rb)
        n_cut += cut
    st.update(
        target=target,
        real_available=n_avail,
        real_tiles=len(pick),
        real_boxes=n_box,
        real_cut_boxes=n_cut,
        real_per_source=per_src,
        real_sources=sorted(digests),
        n_classes=len(names),
    )
    (root / "stats.json").write_text(json.dumps(st, indent=1))
    return root, st


def check_leak(target: str, n_check: int = 25, seed: int = 1) -> None:
    """Ищет случайные тайлы обучения (с чернилами, реальные) на планах target (шаблонное
    совпадение после обратного поворота). Максимум нормированной корреляции должен быть далек
    от 1 (дубликат дал бы ~1.0)."""
    root = yc.YOLO_DATA / f"cross_{target}"
    files = sorted((root / "images" / "train").glob("real_*.png"))
    rng = random.Random(seed)
    rng.shuffle(files)
    plans = [s.load_gray() for s in common.labeled_plans(subset="all") if s.family == target]
    worst = 0.0
    done = 0
    for f in files:
        im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if (im < 128).mean() < 0.01:
            continue
        k = int(re.search(r"_r(\d+)\.png$", f.name).group(1)) // 90
        tile = np.rot90(im, -k).copy()
        for g in plans:
            r = cv2.matchTemplate(g, tile, cv2.TM_CCOEFF_NORMED)
            worst = max(worst, float(r.max()))
        done += 1
        if done >= n_check:
            break
    print(f"{target}: проверено {done} тайлов на {len(plans)} планах, max corr = {worst:.3f}")
    assert worst < 0.9, "тайл обучения найден на плане целевого семейства"


def run_target(
    target: str, epochs: int, patience: int, weights: str, batch: int, n_synth: int, n_real: int
) -> None:
    names = yc.all_class_names()
    own = [i for i, n in enumerate(names) if n in set(yc.class_names(target))]
    root, stats = build_cross(target, n_synth, n_real)
    print(json.dumps(stats), flush=True)
    t0 = time.time()
    best = yc.train_yolo(root / "data.yaml", f"cross_{target}", epochs, patience, weights, batch)
    train_min = (time.time() - t0) / 60
    from ultralytics import YOLO

    model = YOLO(str(best))
    vi, vb = yc.load_val(root)  # только синтетическая валидация целевого семейства
    curve = yc.val_f1_curve(model, vi, vb, classes=own)
    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    (CURVE_DIR / f"{target}.json").write_text(json.dumps({str(k): v for k, v in curve.items()}))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    t1 = time.time()
    for s in common.labeled_plans(subset="all"):
        if s.family != target:
            continue
        preds = yc.predict_sheet(model, s.load_gray(), conf=RAW_CONF, names=names, classes=own)
        (RAW_DIR / (Path(s.file_name).stem + ".json")).write_text(json.dumps(preds))
        print(s.file_name, len(preds), flush=True)
    info = dict(family=target, train_min=train_min, predict_s=time.time() - t1, weights=str(best))
    (CURVE_DIR / f"{target}_info.json").write_text(json.dumps(info))
    print(json.dumps(info), flush=True)


def finalize(families: tuple[str, ...] = yc.ALL_FAMILIES) -> None:
    """conf по синтетической валидации целевого семейства (максимум F1); yolo_cross (+ c25, c50)."""
    cp = CURVE_DIR / "chosen_conf.json"
    chosen: dict[str, float] = json.loads(cp.read_text()) if cp.exists() else {}
    for fam in families:
        p = CURVE_DIR / f"{fam}.json"
        if not p.exists():
            continue
        curve = {float(k): v for k, v in json.loads(p.read_text()).items()}
        chosen[fam] = yc.best_conf(curve)
        print(
            fam,
            "val F1:",
            {k: round(v[2], 3) for k, v in sorted(curve.items())},
            "-> conf",
            chosen[fam],
        )
    cp.write_text(json.dumps(chosen))
    for tag, fixed in (("yolo_cross", None), ("yolo_cross_c25", 0.25), ("yolo_cross_c50", 0.5)):
        out = common.OUT / tag
        out.mkdir(parents=True, exist_ok=True)
        for f in RAW_DIR.glob("*.json"):
            fam = common.family_of(f.name)
            if fam not in families:
                continue
            conf = fixed if fixed is not None else chosen[fam]
            preds = [p for p in json.loads(f.read_text()) if p["score"] >= conf]
            (out / f.name).write_text(json.dumps(preds))


# ------------------------------------------------------------ сравнение методов


def _match_idx(preds: list[dict], gts: list[common.Box], thr: float = 0.5):
    """Жадное сопоставление с классом: (индексы gt, найденные предсказаниями; пары (pred, gt))."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i]["score"])
    used: dict[int, int] = {}
    for i in order:
        best, bj = thr, -1
        for j, g in enumerate(gts):
            if j in used or g.legend_id != preds[i]["legend_id"]:
                continue
            v = common.iou_xywh(preds[i]["bbox"], g.bbox)
            if v >= best:
                best, bj = v, j
        if bj >= 0:
            used[bj] = i
    return used


def _group(n: int) -> str:
    return "freq>=20" if n >= 20 else "mid 5-19" if n >= 5 else "rare 1-4"


def compare(methods: list[str], subset: str, families: list[str], strict: bool = False) -> None:
    """Полнота по частотным группам классов и типичные ошибки методов на одних листах."""
    from collections import Counter

    sheets = common.load_sheets()
    plans = [s for s in common.labeled_plans(sheets, subset) if s.family in families]
    cnt: Counter = Counter()  # (семейство, класс) -> число экземпляров на тестовых листах
    for s in plans:
        for o in s.objects:
            cnt[(s.family, o.legend_id)] += 1
    for m in methods:
        grp = {g: [0, 0] for g in ("freq>=20", "mid 5-19", "rare 1-4")}
        fam_tot: dict[str, list[int]] = {}
        err: dict[str, Counter] = {k: Counter() for k in ("wrong_cls", "missed", "fp_bg", "fp_loc")}
        confus: Counter = Counter()
        fp_cls: Counter = Counter()
        miss_cls: Counter = Counter()
        for s in plans:
            known = None if strict else common.known_classes(s.family, sheets, subset)
            try:
                preds = common.load_preds(m, s.file_name)
            except FileNotFoundError:
                continue
            if known is not None:
                preds = [p for p in preds if p["legend_id"] in known]
            used = _match_idx(preds, s.objects)
            ft = fam_tot.setdefault(s.family, [0, 0, 0])
            ft[0] += len(used)
            ft[1] += len(preds)
            ft[2] += len(s.objects)
            for j, g in enumerate(s.objects):
                gname = _group(cnt[(s.family, g.legend_id)])
                grp[gname][1] += 1
                if j in used:
                    grp[gname][0] += 1
                    continue
                ov = [
                    (common.iou_xywh(p["bbox"], g.bbox), p["legend_id"])
                    for p in preds
                    if p["legend_id"] != g.legend_id
                ]
                ov = [o for o in ov if o[0] >= 0.5]
                if ov:
                    err["wrong_cls"][s.family] += 1
                    confus[(g.legend_id, max(ov)[1])] += 1
                else:
                    err["missed"][s.family] += 1
                    miss_cls[(s.family, g.legend_id)] += 1
            matched_p = set(used.values())
            for i, p in enumerate(preds):
                if i in matched_p:
                    continue
                v = max((common.iou_xywh(p["bbox"], g.bbox) for g in s.objects), default=0.0)
                if v >= 0.5:
                    continue  # ошибка класса, уже учтена со стороны эталона (или дубль)
                if v >= 0.1:
                    err["fp_loc"][s.family] += 1
                else:
                    err["fp_bg"][s.family] += 1
                    fp_cls[p["legend_id"]] += 1
        print(f"== {m} [{subset}{' strict' if strict else ''}] ==")
        for g, (tp, n) in grp.items():
            nc = sum(1 for k, v in cnt.items() if _group(v) == g)
            rec = tp / n if n else float("nan")
            print(f"  {g:9s}: классов {nc:3d}, экз. {n:4d}, полнота {rec:.3f}")
        for fam, (tp, npd, ng) in fam_tot.items():
            p_, r_, f_ = common.prf(tp, npd, ng)
            print(
                f"  {fam:10s}: P={p_:.3f} R={r_:.3f} F1={f_:.3f} (tp {tp}, pred {npd}, gt {ng}); "
                f"wrong_cls={err['wrong_cls'][fam]} missed={err['missed'][fam]} "
                f"fp_bg={err['fp_bg'][fam]} fp_loc={err['fp_loc'][fam]}"
            )
        print("  частые путаницы gt->pred:", confus.most_common(6))
        print("  частые пропуски (семейство, класс):", miss_cls.most_common(6))
        print("  классы фоновых ложных:", fp_cls.most_common(6))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(yc.ALL_FAMILIES))
    ap.add_argument("--n-synth", type=int, default=2000)
    ap.add_argument("--n-real", type=int, default=N_REAL)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--check-leak", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--compare", nargs="+", metavar="METHOD")
    ap.add_argument("--set", default="human", choices=["human", "agent", "all"])
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--families", nargs="+", default=list(yc.ALL_FAMILIES))
    a = ap.parse_args()
    if a.compare:
        compare(a.compare, a.set, a.families, a.strict)
    elif a.finalize:
        finalize(tuple(a.families))
    elif a.check_leak:
        check_leak(a.target)
    elif a.build_only:
        print(json.dumps(build_cross(a.target, a.n_synth, a.n_real)[1], indent=1))
    else:
        run_target(a.target, a.epochs, a.patience, a.weights, a.batch, a.n_synth, a.n_real)
