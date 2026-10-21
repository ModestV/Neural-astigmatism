"""H3: YOLO только на синтетике из легенды (протокол A: реальной разметки планов нет).

Для семейства с размеченной легендой (worcester, usc, canada, wisd, va) строятся
синтетические тайлы 640x640:
фон берется с листов ДРУГИХ семейств, эталоны целевой легенды вклеиваются через np.minimum
(чернила поверх фона), рамка = рамка чернил вставленного символа. Дистракторами служат эталоны
чужих легенд (без рамки) и фрагменты текста.

Запуск:
    .venv/bin/python m_yolo_synth.py --family worcester [--n 2000 --epochs 14]
    .venv/bin/python m_yolo_synth.py --family usc --predict-only   # новые планы существующей модели
    .venv/bin/python m_yolo_synth.py --finalize --families canada wisd va usc

finalize пишет только листы указанных семейств, которых еще нет в yolo_synth (уже
оцененные 01, 02, 03, 05 не перезаписываются).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import common
import cv2
import numpy as np
import yolo_common as yc

RAW_CONF = 0.05
RAW_DIR = yc.YOLO_DATA / "preds_raw" / "yolo_synth"
CURVE_DIR = yc.YOLO_DATA / "curves" / "yolo_synth"
EXAMPLES = yc.YOLO_DATA / "synth_examples"

# параметры генератора
SCALE_RANGE = (0.8, 1.15)
N_SYM_RANGE = (4, 20)
P_EMPTY_BG = 0.2  # доля почти пустых фонов (белое с линиями)
P_DISTRACTOR = 0.5  # вероятность вклейки чужих эталонов
N_DISTRACTOR = (1, 4)
P_TEXT = 0.4  # вероятность вклейки фрагментов текста
P_THICK = 0.12
P_THIN = 0.12
P_BLUR = 0.3
P_ADJACENT = 0.2  # доля символов, ставящихся вплотную к уже вставленному
P_NO_SYMBOLS = 0.04  # тайлы без символов (только дистракторы)
MAX_BG_INK = 0.2  # фоны с большей долей чернил (сплошные заливки) отбрасываются


class Pool:
    """Листы других семейств как источник фонов, фрагментов текста и дистракторов."""

    def __init__(self, family: str, rng: random.Random) -> None:
        self.rng = rng
        sheets = common.load_sheets()
        # фоны и тексты только с листов других семейств: листы целевого семейства не используются
        self.grays = [s.load_gray() for s in sheets if s.family != family]
        assert all(s.family != family for s in sheets if s.family != family)
        self.distractors: list[np.ndarray] = []
        for fam in yc.ALL_FAMILIES:
            if fam != family:
                self.distractors += [trim(c) for c in common.legend_crops(fam).values()]
        self.distractors = [d for d in self.distractors if d is not None]

    def background(self, empty: bool) -> np.ndarray:
        rng = self.rng
        if empty:
            return random_lines_tile(rng)
        for _ in range(60):
            g = rng.choice(self.grays)
            h, w = g.shape
            x, y = rng.randrange(0, w - yc.TILE), rng.randrange(0, h - yc.TILE)
            t = g[y : y + yc.TILE, x : x + yc.TILE]
            ink = float((t < 128).mean())
            if 0.001 < ink < MAX_BG_INK:
                return t.copy()
        return random_lines_tile(rng)

    def text_fragment(self) -> np.ndarray | None:
        rng = self.rng
        for _ in range(40):
            g = rng.choice(self.grays)
            h, w = g.shape
            fw, fh = rng.randrange(100, 260), rng.randrange(24, 60)
            x, y = rng.randrange(0, w - fw), rng.randrange(0, h - fh)
            t = g[y : y + fh, x : x + fw]
            ink = float((t < 128).mean())
            if 0.04 < ink < 0.25:
                return t.copy()
        return None


def random_lines_tile(rng: random.Random) -> np.ndarray:
    """Почти пустой белый тайл с несколькими линиями и прямоугольниками (толщина 1-2 px)."""
    t = np.full((yc.TILE, yc.TILE), 255, np.uint8)
    for _ in range(rng.randrange(0, 7)):
        th = rng.choice([1, 2, 2, 3])
        kind = rng.random()
        if kind < 0.5:
            if rng.random() < 0.5:
                y = rng.randrange(0, yc.TILE)
                cv2.line(t, (rng.randrange(0, 200), y), (rng.randrange(300, yc.TILE), y), 0, th)
            else:
                x = rng.randrange(0, yc.TILE)
                cv2.line(t, (x, rng.randrange(0, 200)), (x, rng.randrange(300, yc.TILE)), 0, th)
        elif kind < 0.8:
            x, y = rng.randrange(0, 500), rng.randrange(0, 500)
            cv2.rectangle(
                t, (x, y), (x + rng.randrange(40, 300), y + rng.randrange(30, 250)), 0, th
            )
        else:
            cv2.line(
                t,
                (rng.randrange(0, 640), rng.randrange(0, 640)),
                (rng.randrange(0, 640), rng.randrange(0, 640)),
                0,
                th,
            )
    return t


def trim(a: np.ndarray, thr: int = 180) -> np.ndarray | None:
    """Обрезка до чернил (серый uint8, белый фон)."""
    ys, xs = np.where(a < thr)
    if len(ys) == 0:
        return None
    return a[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].copy()


def augment_symbol(crop: np.ndarray, rng: random.Random) -> np.ndarray | None:
    """Поворот на k*90, масштаб, утолщение/утоньшение на ~1 px, размытие, обрезка до чернил."""
    a = np.rot90(crop, rng.randrange(4)).copy()
    s = rng.uniform(*SCALE_RANGE)
    a = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    a = cv2.copyMakeBorder(a, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=255)
    r = rng.random()
    k2 = np.ones((2, 2), np.uint8)
    if r < P_THICK:
        a = cv2.erode(a, k2)  # темное расширяется
    elif r < P_THICK + P_THIN:
        a = cv2.dilate(a, k2)
    if rng.random() < P_BLUR:
        a = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.3, 0.8))
    return trim(a)


def _overlap(
    r: tuple[int, int, int, int], rects: list[tuple[int, int, int, int]], gap: int
) -> bool:
    return any(
        r[0] < q[2] + gap and r[2] + gap > q[0] and r[1] < q[3] + gap and r[3] + gap > q[1]
        for q in rects
    )


def make_tile(
    pool: Pool,
    crops: dict[int, np.ndarray],
    queue: list[int],
    rng: random.Random,
) -> tuple[np.ndarray, list[yc.Box5]]:
    """Синтетический тайл и его рамки. queue: общая очередь классов (баланс), пополняется здесь."""
    bg = pool.background(rng.random() < P_EMPTY_BG)
    rects: list[tuple[int, int, int, int]] = []
    boxes: list[yc.Box5] = []
    T = yc.TILE

    def place(sym: np.ndarray, near: bool) -> tuple[int, int, int, int] | None:
        h, w = sym.shape
        for _ in range(25):
            if near and rects:
                q = rng.choice(rects)
                gap = rng.randrange(2, 12)
                side = rng.randrange(4)
                if side == 0:
                    x, y = q[2] + gap, q[1] + rng.randrange(-h // 2, h // 2)
                elif side == 1:
                    x, y = q[0] - gap - w, q[1] + rng.randrange(-h // 2, h // 2)
                elif side == 2:
                    x, y = q[0] + rng.randrange(-w // 2, w // 2), q[3] + gap
                else:
                    x, y = q[0] + rng.randrange(-w // 2, w // 2), q[1] - gap - h
            else:
                x, y = rng.randrange(2, T - w - 2), rng.randrange(2, T - h - 2)
            if x < 1 or y < 1 or x + w > T - 1 or y + h > T - 1:
                continue
            r = (x, y, x + w, y + h)
            if _overlap(r, rects, 2):
                continue
            return r
        return None

    def paste(sym: np.ndarray, r: tuple[int, int, int, int]) -> None:
        reg = bg[r[1] : r[3], r[0] : r[2]]
        np.minimum(reg, sym, out=reg)

    # дистракторы и текст ставятся первыми, чтобы целевые символы с ними не пересекались
    if pool.distractors and rng.random() < P_DISTRACTOR:
        for _ in range(rng.randint(*N_DISTRACTOR)):
            sym = augment_symbol(rng.choice(pool.distractors), rng)
            if sym is None:
                continue
            r = place(sym, False)
            if r:
                paste(sym, r)
                rects.append(r)
    if rng.random() < P_TEXT:
        for _ in range(rng.randint(1, 3)):
            frag = pool.text_fragment()
            if frag is None:
                continue
            fh, fw = frag.shape
            x, y = rng.randrange(0, T - fw), rng.randrange(0, T - fh)
            reg = bg[y : y + fh, x : x + fw]
            np.minimum(reg, frag, out=reg)
            # фрагмент текста не резервируется: символы могут оказаться рядом с ним или поверх

    n = 0 if rng.random() < P_NO_SYMBOLS else rng.randint(*N_SYM_RANGE)
    for _ in range(n):
        if not queue:
            queue.extend(rng.sample(sorted(crops), len(crops)))
        c = queue.pop()
        sym = augment_symbol(crops[c], rng)
        if sym is None:
            continue
        r = place(sym, rng.random() < P_ADJACENT)
        if r is None:
            queue.append(c)
            continue
        paste(sym, r)
        rects.append(r)
        boxes.append((c, float(r[0]), float(r[1]), float(r[2] - r[0]), float(r[3] - r[1])))
    return bg, boxes


def build_synth(
    family: str,
    n_total: int,
    seed: int = 0,
    val_frac: float = 0.1,
    names: list[str] | None = None,
    root: Path | None = None,
) -> tuple[Path, dict]:
    """Синтетический датасет из легенды семейства. names: пространство классов (по умолчанию
    классы легенды семейства; для протокола C объединение пяти легенд), root: каталог датасета."""
    rng = random.Random(seed)
    names = names or yc.class_names(family)
    idx = {n: i for i, n in enumerate(names)}
    crops_raw = common.legend_crops(family)
    crops = {idx[k]: t for k, v in crops_raw.items() if (t := trim(v)) is not None}
    pool = Pool(family, rng)
    root = root or yc.YOLO_DATA / f"synth_{family}"
    if root.exists():
        shutil.rmtree(root)
    n_val = int(round(n_total * val_frac))
    queue: list[int] = []
    counts = np.zeros(len(names), int)
    own = sorted(idx[k] for k in crops_raw)
    examples = []
    for i in range(n_total):
        split = "val" if i < n_val else "train"
        img, boxes = make_tile(pool, crops, queue, rng)
        yc.write_sample(root, split, f"s{i:05d}", img, boxes)
        if split == "train":
            for b in boxes:
                counts[b[0]] += 1
        if i % (n_total // 8) == 0 and boxes:
            examples.append((i, img, boxes))
    yc.write_yaml(root, names)
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    for i, img, boxes in examples[:8]:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for c, x, y, w, h in boxes:
            cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)
            cv2.putText(
                vis,
                names[c],
                (int(x), max(10, int(y) - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 0, 0),
                1,
            )
        cv2.imwrite(str(EXAMPLES / f"{root.name}_{i:05d}.png"), vis)
    stats = dict(
        family=family,
        n_train=n_total - n_val,
        n_val=n_val,
        boxes_train=int(counts.sum()),
        per_class_min=int(counts[own].min()),
        per_class_max=int(counts[own].max()),
    )
    (root / "stats.json").write_text(json.dumps(stats, indent=1))
    return root, stats


def run_family(family: str, n: int, epochs: int, patience: int, weights: str, batch: int) -> None:
    root, stats = build_synth(family, n)
    print(json.dumps(stats), flush=True)
    t0 = time.time()
    best = yc.train_yolo(root / "data.yaml", f"synth_{family}", epochs, patience, weights, batch)
    train_min = (time.time() - t0) / 60
    from ultralytics import YOLO

    model = YOLO(str(best))
    vi, vb = yc.load_val(root)
    curve = yc.val_f1_curve(model, vi, vb)
    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    (CURVE_DIR / f"{family}.json").write_text(json.dumps({str(k): v for k, v in curve.items()}))
    t1 = time.time()
    predict_family(family, model)
    info = dict(family=family, train_min=train_min, predict_s=time.time() - t1, weights=str(best))
    (CURVE_DIR / f"{family}_info.json").write_text(json.dumps(info))
    print(json.dumps(info), flush=True)


def predict_family(family: str, model, overwrite: bool = False) -> None:
    """Сырые предсказания (conf 0.05) на всех размеченных планах семейства; существующие файлы
    не перезаписываются, если не задан overwrite."""
    names = yc.class_names(family)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for s in common.labeled_plans(subset="all"):
        if s.family != family:
            continue
        out = RAW_DIR / (Path(s.file_name).stem + ".json")
        if out.exists() and not overwrite:
            print(s.file_name, "уже есть, пропуск", flush=True)
            continue
        preds = yc.predict_sheet(model, s.load_gray(), conf=RAW_CONF, names=names)
        out.write_text(json.dumps(preds))
        print(s.file_name, len(preds), flush=True)


def predict_only(family: str) -> None:
    from ultralytics import YOLO

    model = YOLO(str(yc.YOLO_RUNS / f"synth_{family}" / "weights" / "best.pt"))
    predict_family(family, model)


def finalize(families: tuple[str, ...] = yc.ALL_FAMILIES) -> None:
    """conf по синтетической валидации (максимум F1 на семейство); yolo_synth + c25 + c50.

    Пишутся только листы указанных семейств и только отсутствующие в выходной папке."""
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
    for tag, fixed in (("yolo_synth", None), ("yolo_synth_c25", 0.25), ("yolo_synth_c50", 0.5)):
        for f in RAW_DIR.glob("*.json"):
            fam = common.family_of(f.name)
            out = common.OUT / tag
            out.mkdir(parents=True, exist_ok=True)
            if fam not in families or (out / f.name).exists():
                continue
            conf = fixed if fixed is not None else chosen[fam]
            preds = [p for p in json.loads(f.read_text()) if p["score"] >= conf]
            (out / f.name).write_text(json.dumps(preds))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=list(yc.ALL_FAMILIES))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--predict-only", action="store_true")
    ap.add_argument("--families", nargs="+", default=list(yc.ALL_FAMILIES))
    a = ap.parse_args()
    if a.finalize:
        finalize(tuple(a.families))
    elif a.predict_only:
        predict_only(a.family)
    elif a.build_only:
        print(build_synth(a.family, a.n)[1])
    else:
        run_family(a.family, a.n, a.epochs, a.patience, a.weights, a.batch)
