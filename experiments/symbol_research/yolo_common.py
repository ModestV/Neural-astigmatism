"""Общие функции для YOLO-постановок: нарезка тайлов, поворот, датасет, предсказание на листе.

Формат рамок внутри модуля: (cls, x, y, w, h) в пикселях (тайла или листа), ось Y вниз.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import common
import cv2
import numpy as np

TILE = 640
OVERLAP = 160
EDGE_MARGIN = 2  # рамка ближе этого числа пикселей к внутреннему краю тайла считается обрезанной
MPS = "mps"
ALL_FAMILIES = ("worcester", "usc", "canada", "wisd", "va")

YOLO_DATA = common.OUT / "yolo_data"
YOLO_RUNS = common.OUT / "yolo_runs"
WEIGHTS = common.OUT / "weights"

Box5 = tuple[int, float, float, float, float]  # cls, x, y, w, h


def _natural_key(s: str) -> tuple[str, int]:
    """'12' -> ('', 12); 'canada_07' -> ('canada_', 7): числовой порядок внутри префикса."""
    m = re.fullmatch(r"(.*?)(\d+)", s)
    return (m.group(1), int(m.group(2))) if m else (s, -1)


def class_names(family: str) -> list[str]:
    """Все legend_id легенды семейства, в числовом порядке (индекс = номер класса YOLO)."""
    _, items = common.legend_for(family)
    return sorted({it.legend_id for it in items}, key=_natural_key)


def all_class_names(families: tuple[str, ...] = ALL_FAMILIES) -> list[str]:
    """Объединение id легенд нескольких семейств (id уникальны между семействами)."""
    out: list[str] = []
    for fam in families:
        out += class_names(fam)
    assert len(out) == len(set(out)), "legend_id не уникальны между семействами"
    return out


# ------------------------------------------------------------------ тайлы


def tile_origins(
    size: int, tile: int = TILE, overlap: int = OVERLAP, start: int = 0, end: int | None = None
) -> list[int]:
    """Левые границы тайлов на отрезке [start, end), шаг tile-overlap; последний прижат к краю."""
    end = size if end is None else end
    if end - start <= tile:
        return [max(0, end - tile)] if end - tile >= 0 else [0]
    step = tile - overlap
    xs = list(range(start, end - tile, step))
    xs.append(end - tile)
    return xs


def crop_tile(gray: np.ndarray, x0: int, y0: int, tile: int = TILE) -> np.ndarray:
    """Вырезка тайла с белым добиванием за краем листа."""
    h, w = gray.shape
    out = np.full((tile, tile), 255, np.uint8)
    xe, ye = min(w, x0 + tile), min(h, y0 + tile)
    out[: ye - y0, : xe - x0] = gray[y0:ye, x0:xe]
    return out


def boxes_in_tile(
    boxes: list[Box5], x0: int, y0: int, tile: int = TILE, eps: float = 1.0
) -> tuple[list[Box5], int]:
    """Рамки, целиком лежащие в тайле (в координатах тайла), и число частично попавших (обрезанных).

    Обрезанные рамки не добавляются в разметку (символ разрезан границей тайла).
    """
    inside: list[Box5] = []
    cut = 0
    for c, x, y, w, h in boxes:
        if (
            x >= x0 - eps
            and y >= y0 - eps
            and x + w <= x0 + tile + eps
            and y + h <= y0 + tile + eps
        ):
            inside.append((c, x - x0, y - y0, w, h))
        elif x < x0 + tile and x + w > x0 and y < y0 + tile and y + h > y0:
            cut += 1
    return inside, cut


def rotate_tile(img: np.ndarray, boxes: list[Box5], k: int) -> tuple[np.ndarray, list[Box5]]:
    """Поворот квадратного тайла на k*90 градусов против часовой стрелки вместе с рамками."""
    t = img.shape[0]
    k %= 4
    out = np.rot90(img, k).copy()
    cur = list(boxes)
    for _ in range(k):  # один поворот ccw: (x, y) -> (y, T - x)
        cur = [(c, y, t - x - w, h, w) for c, x, y, w, h in cur]
    return out, cur


def write_sample(root: Path, split: str, name: str, img: np.ndarray, boxes: list[Box5]) -> None:
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    t = img.shape[0]
    cv2.imwrite(str(root / "images" / split / f"{name}.png"), img)
    lines = []
    for c, x, y, w, h in boxes:
        x1, y1, x2, y2 = max(0.0, x), max(0.0, y), min(float(t), x + w), min(float(t), y + h)
        lines.append(
            f"{c} {(x1 + x2) / 2 / t:.6f} {(y1 + y2) / 2 / t:.6f} "
            f"{(x2 - x1) / t:.6f} {(y2 - y1) / t:.6f}"
        )
    (root / "labels" / split / f"{name}.txt").write_text("\n".join(lines))


def write_yaml(root: Path, names: list[str]) -> Path:
    lines = [f"path: {root}", "train: images/train", "val: images/val", "names:"]
    lines += [f"  {i}: '{n}'" for i, n in enumerate(names)]
    p = root / "data.yaml"
    p.write_text("\n".join(lines) + "\n")
    return p


@dataclass
class Source:
    """Лист с разметкой, из которого режутся тайлы обучения."""

    name: str
    gray: np.ndarray
    boxes: list[Box5]
    region: tuple[int, int, int, int]  # (x0, y0, x1, y1): тайл обязан лежать целиком внутри
    jitter_per_obj: int = 2  # дополнительные тайлы со случайным положением вокруг каждого объекта
    neg_frac: float = 0.15  # доля пустых тайлов (от числа положительных)
    val_frac: float = 0.12  # доля объектов в валидационных ячейках
    exclude: list[tuple[int, int, int, int]] = field(default_factory=list)


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _ink_frac(img: np.ndarray) -> float:
    return float((img < 128).mean())


def choose_val_windows(
    src: Source, rng: random.Random, cell: int = 480, neg_ratio: float = 1.0
) -> list[tuple[int, int]]:
    """Окна валидации (левые верхние углы тайлов) вокруг случайных ячеек с объектами в region."""
    x0, y0, x1, y1 = src.region
    cells: dict[tuple[int, int], int] = {}
    inb = [
        b
        for b in src.boxes
        if b[1] >= x0 and b[2] >= y0 and b[1] + b[3] <= x1 and b[2] + b[4] <= y1
    ]
    for _c, x, y, w, h in inb:
        cells[(int((x + w / 2 - x0) // cell), int((y + h / 2 - y0) // cell))] = 1
    keys = sorted(cells)
    rng.shuffle(keys)
    target = max(1, round(src.val_frac * len(inb)))
    wins: list[tuple[int, int]] = []
    used: list[tuple[int, int, int, int]] = []
    got = 0
    for cx, cy in keys:
        if got >= target:
            break
        wx = int(x0 + cx * cell + cell / 2 - TILE / 2)
        wy = int(y0 + cy * cell + cell / 2 - TILE / 2)
        wx = min(max(wx, x0), x1 - TILE)
        wy = min(max(wy, y0), y1 - TILE)
        r = (wx, wy, wx + TILE, wy + TILE)
        if any(_intersects(r, u) for u in used):
            continue
        used.append(r)
        wins.append((wx, wy))
        got += sum(
            1
            for b in inb
            if r[0] <= b[1] and r[2] >= b[1] + b[3] and r[1] <= b[2] and r[3] >= b[2] + b[4]
        )
    # негативные окна валидации (без объектов, но с чернилами) для оценки precision
    n_neg = int(round(neg_ratio * len(wins)))
    for _try in range(400):
        if n_neg <= 0:
            break
        wx = rng.randrange(x0, max(x0 + 1, x1 - TILE + 1))
        wy = rng.randrange(y0, max(y0 + 1, y1 - TILE + 1))
        r = (wx, wy, wx + TILE, wy + TILE)
        if any(_intersects(r, u) for u in used):
            continue
        if boxes_in_tile(src.boxes, wx, wy)[0] or _ink_frac(crop_tile(src.gray, wx, wy)) < 0.005:
            continue
        used.append(r)
        wins.append((wx, wy))
        n_neg -= 1
    return wins


def build_source_tiles(
    src: Source, rng: random.Random, val_windows: list[tuple[int, int]] | None = None
) -> tuple[list[tuple[int, int, str]], list[tuple[int, int, str]]]:
    """Список тайлов (x0, y0, kind) для обучения и валидации.

    Тайлы обучения не пересекают окна валидации и src.exclude и лежат в region
    (это и есть гарантия отсутствия утечки)."""
    x0r, y0r, x1r, y1r = src.region
    val_windows = val_windows or []
    vrects = [(x, y, x + TILE, y + TILE) for x, y in val_windows]
    banned = vrects + list(src.exclude)

    def ok(x: int, y: int) -> bool:
        r = (x, y, x + TILE, y + TILE)
        return (
            x >= x0r
            and y >= y0r
            and x + TILE <= x1r
            and y + TILE <= y1r
            and not any(_intersects(r, b) for b in banned)
        )

    xs = tile_origins(x1r, start=x0r, end=x1r)
    ys = tile_origins(y1r, start=y0r, end=y1r)
    pos: list[tuple[int, int, str]] = []
    neg_cands: list[tuple[int, int, str]] = []
    for y in ys:
        for x in xs:
            if not ok(x, y):
                continue
            ins, cut = boxes_in_tile(src.boxes, x, y)
            if ins:
                pos.append((x, y, "grid"))
            elif cut == 0:
                neg_cands.append((x, y, "neg"))
    for c, bx, by, bw, bh in src.boxes:
        if not (bx >= x0r and by >= y0r and bx + bw <= x1r and by + bh <= y1r):
            continue
        for _ in range(src.jitter_per_obj):
            for _try in range(20):
                x = int(bx + bw / 2 - rng.uniform(40, TILE - 40))
                y = int(by + bh / 2 - rng.uniform(40, TILE - 40))
                x = min(max(x, x0r), x1r - TILE)
                y = min(max(y, y0r), y1r - TILE)
                if ok(x, y) and boxes_in_tile([(c, bx, by, bw, bh)], x, y)[0]:
                    pos.append((x, y, "jit"))
                    break
    # негативы: только тайлы с чернилами
    rng.shuffle(neg_cands)
    n_neg = int(round(src.neg_frac * len(pos)))
    neg: list[tuple[int, int, str]] = []
    for x, y, k in neg_cands:
        if len(neg) >= n_neg:
            break
        if _ink_frac(crop_tile(src.gray, x, y)) > 0.002:
            neg.append((x, y, k))
    return pos + neg, [(x, y, "val") for x, y in val_windows]


def write_tiles(
    root: Path,
    split: str,
    src: Source,
    tiles: list[tuple[int, int, str]],
    rotations: tuple[int, ...] = (0, 1, 2, 3),
) -> tuple[int, int, int]:
    """Пишет тайлы (во всех поворотах). Возвращает (число изображений, рамок, обрезанных рамок)."""
    n_img = n_box = n_cut = 0
    for x, y, kind in tiles:
        img = crop_tile(src.gray, x, y)
        ins, cut = boxes_in_tile(src.boxes, x, y)
        n_cut += cut
        for k in rotations:
            ri, rb = rotate_tile(img, ins, k)
            write_sample(root, split, f"{src.name}_{x}_{y}_{kind}_r{k * 90}", ri, rb)
            n_img += 1
            n_box += len(rb)
    return n_img, n_box, n_cut


# --------------------------------------------------------- предсказание


def predict_tiles(
    model,
    imgs: list[np.ndarray],
    conf: float,
    tile: int = TILE,
    batch: int = 16,
    device: str = MPS,
    classes: list[int] | None = None,
):
    """Предсказания по тайлам (серые uint8): список массивов (n, 6) [x1, y1, x2, y2, conf, cls].

    classes: номера классов, остающихся в выдаче (фильтр внутри NMS модели)."""
    out = []
    for i in range(0, len(imgs), batch):
        chunk = [cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) for im in imgs[i : i + batch]]
        res = model.predict(
            chunk,
            imgsz=tile,
            conf=conf,
            iou=0.5,
            device=device,
            verbose=False,
            max_det=300,
            classes=classes,
        )
        for r in res:
            b = r.boxes
            out.append(
                np.concatenate(
                    [
                        b.xyxy.cpu().numpy(),
                        b.conf.cpu().numpy()[:, None],
                        b.cls.cpu().numpy()[:, None],
                    ],
                    1,
                )
            )
    return out


def predict_sheet(
    model,
    gray: np.ndarray,
    conf: float = 0.25,
    tile: int = TILE,
    overlap: int = OVERLAP,
    names: list[str] | None = None,
    region: tuple[int, int, int, int] | None = None,
    margin: int = EDGE_MARGIN,
    nms_iou: float = 0.5,
    device: str = MPS,
    classes: list[int] | None = None,
) -> list[dict]:
    """Предсказание по листу: тайлы, выброс рамок на внутренних краях тайла, class-agnostic NMS.

    region: (x0, y0, x1, y1), если нужна только часть листа; ее границы считаются краями листа.
    """
    h, w = gray.shape
    rx0, ry0, rx1, ry1 = region if region else (0, 0, w, h)
    names = names or (
        list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    )
    xs = tile_origins(rx1, tile, overlap, start=rx0, end=rx1)
    ys = tile_origins(ry1, tile, overlap, start=ry0, end=ry1)
    origins = [(x, y) for y in ys for x in xs]
    preds: list[dict] = []
    for i in range(0, len(origins), 32):
        part = origins[i : i + 32]
        imgs = [crop_tile(gray, x, y, tile) for x, y in part]
        arrs = predict_tiles(model, imgs, conf, tile, device=device, classes=classes)
        for (x0, y0), arr in zip(part, arrs, strict=True):
            for x1, y1, x2, y2, sc, c in arr:
                # внутренний край: край тайла, не совпадающий с краем листа (области)
                if (
                    (x0 > rx0 and x1 < margin)
                    or (y0 > ry0 and y1 < margin)
                    or (x0 + tile < rx1 and x2 > tile - margin)
                    or (y0 + tile < ry1 and y2 > tile - margin)
                ):
                    continue
                preds.append(
                    {
                        "bbox": [float(x0 + x1), float(y0 + y1), float(x2 - x1), float(y2 - y1)],
                        "legend_id": names[int(c)],
                        "score": float(sc),
                    }
                )
    return common.nms(preds, iou_thr=nms_iou, class_agnostic=True)


# ------------------------------------------------- выбор порога по валидации


def val_f1_curve(
    model,
    val_imgs: list[np.ndarray],
    val_boxes: list[list[Box5]],
    confs: list[float] | None = None,
    margin: int = EDGE_MARGIN,
    classes: list[int] | None = None,
) -> dict[float, tuple[float, float, float, int, int, int]]:
    """P/R/F1 (с классом, IoU 0.5) по валидационным тайлам: conf -> (P, R, F1, tp, npred, ngt).

    Рамки на краях тайла отбрасываются, как при склейке на листе.
    """
    confs = confs or [round(c, 2) for c in np.arange(0.05, 0.96, 0.05)]
    arrs = predict_tiles(model, val_imgs, conf=min(confs), classes=classes)
    names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    tp_tot = dict.fromkeys(confs, 0)
    np_tot = dict.fromkeys(confs, 0)
    ngt = sum(len(b) for b in val_boxes)
    t = val_imgs[0].shape[0]
    for arr, gts in zip(arrs, val_boxes, strict=True):
        keep = [
            r
            for r in arr
            if not (r[0] < margin or r[1] < margin or r[2] > t - margin or r[3] > t - margin)
        ]
        gt_boxes = [common.Box(bbox=[x, y, w, h], legend_id=names[c]) for c, x, y, w, h in gts]
        for cf in confs:
            pr = [
                {
                    "bbox": [r[0], r[1], r[2] - r[0], r[3] - r[1]],
                    "legend_id": names[int(r[5])],
                    "score": float(r[4]),
                }
                for r in keep
                if r[4] >= cf
            ]
            pr = common.nms(pr, iou_thr=0.5, class_agnostic=True)
            _, tp = common._match(pr, gt_boxes, 0.5, use_class=True)
            tp_tot[cf] += tp
            np_tot[cf] += len(pr)
    out = {}
    for cf in confs:
        p, r, f = common.prf(tp_tot[cf], np_tot[cf], ngt)
        out[cf] = (p, r, f, tp_tot[cf], np_tot[cf], ngt)
    return out


def best_conf(curve: dict[float, tuple], lo: float = 0.1, hi: float = 0.9) -> float:
    """conf с максимальным F1 (при равенстве средний из плато), в диапазоне [lo, hi]."""
    items = [(c, v[2]) for c, v in curve.items() if lo <= c <= hi]
    best = max(f for _, f in items)
    plateau = [c for c, f in items if f >= best - 1e-9]
    return float(np.median(plateau))


def load_val(root: Path) -> tuple[list[np.ndarray], list[list[Box5]]]:
    """Читает валидационные тайлы YOLO-датасета: серые изображения и рамки в пикселях."""
    imgs, boxes = [], []
    for p in sorted((root / "images" / "val").glob("*.png")):
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        t = im.shape[0]
        lab = root / "labels" / "val" / (p.stem + ".txt")
        bb: list[Box5] = []
        if lab.exists():
            for ln in lab.read_text().split("\n"):
                if ln.strip():
                    c, cx, cy, w, h = ln.split()
                    cx, cy, w, h = (float(v) * t for v in (cx, cy, w, h))
                    bb.append((int(c), cx - w / 2, cy - h / 2, w, h))
        imgs.append(im)
        boxes.append(bb)
    return imgs, boxes


def train_yolo(
    data_yaml: Path,
    run_name: str,
    epochs: int,
    patience: int,
    weights: str = "yolo11s.pt",
    batch: int = 16,
    time_hours: float | None = None,
    seed: int = 0,
):
    """Обучение YOLO с запретом отражений. Возвращает путь к best.pt."""
    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS / weights))
    kw = dict(
        data=str(data_yaml),
        imgsz=TILE,
        epochs=epochs,
        patience=patience,
        batch=batch,
        device=MPS,
        workers=4,
        project=str(YOLO_RUNS),
        name=run_name,
        exist_ok=True,
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        scale=0.3,
        mosaic=1.0,
        plots=False,
        seed=seed,
        verbose=False,
        amp=False,
        close_mosaic=max(1, min(10, epochs // 5)),
    )
    if time_hours:
        kw["time"] = time_hours
    model.train(**kw)
    return YOLO_RUNS / run_name / "weights" / "best.pt"
