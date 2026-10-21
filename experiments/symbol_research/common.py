"""Общий контракт исследования: данные, формат предсказаний, метрики.

Все методы (шаблоны, SymSpot, YOLO, гибриды) отдают предсказания в одном формате и
оцениваются одной функцией `evaluate`, чтобы сравнение было честным.

Предсказание: dict(bbox=[x, y, w, h] в пикселях листа, legend_id=str, score=float).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "real" / "supertest"
IMAGES = DATA / "images" / "default"
ANN = DATA / "annotations" / "instances_default.json"
OUT = ROOT / "outputs" / "research"

# Семейство листа (комплект одного проектировщика) по имени файла.
FAMILY_PREFIX = {
    "worcester": "worcester",
    "usc": "usc",
    "canada": "canada",
    "wisd": "wisd",
    "va": "va",
}


def family_of(file_name: str) -> str:
    stem = Path(file_name).stem
    parts = stem.split("_")
    # "01_worcester_FA-101..." или "worcester_FAN-001_legend"
    for p in parts[:2]:
        if p in FAMILY_PREFIX:
            return FAMILY_PREFIX[p]
    raise ValueError(f"неизвестное семейство: {file_name}")


@dataclass
class Box:
    bbox: list[float]  # x, y, w, h
    legend_id: str
    rotation: float = 0.0
    score: float = 1.0


@dataclass
class Sheet:
    image_id: int
    file_name: str
    family: str
    width: int
    height: int
    is_legend: bool
    objects: list[Box] = field(default_factory=list)  # разметка экземпляров (object)
    legend_items: list[Box] = field(default_factory=list)  # эталоны (legend_item)

    @property
    def path(self) -> Path:
        return IMAGES / self.file_name

    @property
    def labeled(self) -> bool:
        return bool(self.objects)

    def load_gray(self) -> np.ndarray:
        """Лист в оттенках серого, uint8, черные линии на белом."""
        return np.asarray(Image.open(self.path).convert("L"))


def _rotated_aabb(bbox: list[float], rotation: float) -> list[float]:
    """CVAT пишет в COCO неповернутый прямоугольник и угол поворота вокруг его центра.

    Возвращает описанную осевую рамку повернутого прямоугольника (для 90/270 это обмен w и h).
    """
    x, y, w, h = bbox
    if rotation % 360 == 0:
        return bbox
    t = np.deg2rad(rotation)
    c, s = abs(np.cos(t)), abs(np.sin(t))
    nw, nh = w * c + h * s, w * s + h * c
    cx, cy = x + w / 2, y + h / 2
    return [float(cx - nw / 2), float(cy - nh / 2), float(nw), float(nh)]


def load_sheets() -> list[Sheet]:
    d = json.loads(ANN.read_text())
    cats = {c["id"]: c["name"] for c in d["categories"]}
    sheets: dict[int, Sheet] = {}
    for im in d["images"]:
        fn = im["file_name"]
        sheets[im["id"]] = Sheet(
            image_id=im["id"],
            file_name=fn,
            family=family_of(fn),
            width=im["width"],
            height=im["height"],
            is_legend="legend" in fn,
        )
    for a in d["annotations"]:
        attrs = a.get("attributes", {})
        b = Box(
            bbox=_rotated_aabb([float(v) for v in a["bbox"]], float(attrs.get("rotation", 0.0))),
            legend_id=str(attrs.get("legend_id", "")),
            rotation=float(attrs.get("rotation", 0.0)),
        )
        s = sheets[a["image_id"]]
        if cats[a["category_id"]] == "object":
            s.objects.append(b)
        elif cats[a["category_id"]] == "legend_item":
            s.legend_items.append(b)
    return [sheets[k] for k in sorted(sheets)]


def legend_for(family: str, sheets: list[Sheet] | None = None) -> tuple[Sheet, list[Box]]:
    """Лист легенды семейства и его эталоны (может быть пустым, если легенда не размечена)."""
    sheets = sheets or load_sheets()
    for s in sheets:
        if s.is_legend and s.family == family:
            return s, s.legend_items
    raise KeyError(family)


def legend_crops(family: str, pad: int = 0) -> dict[str, np.ndarray]:
    """legend_id -> вырезка эталона (серый uint8) с листа легенды."""
    sheet, items = legend_for(family)
    img = sheet.load_gray()
    out = {}
    for it in items:
        x, y, w, h = it.bbox
        x0, y0 = max(0, int(round(x)) - pad), max(0, int(round(y)) - pad)
        x1, y1 = int(round(x + w)) + pad, int(round(y + h)) + pad
        out[it.legend_id] = img[y0:y1, x0:x1].copy()
    return out


# Листы, размеченные человеком (версия разметки 1). Остальные планы (04, 06-10) и легенды
# canada/wisd/va размечены позже агентом: предразметка шаблонным методом, визуальный
# отбор и добавление пропущенных; рамки прилипали к совпадениям эталона. Для оценки
# эти листы смещены в пользу шаблонного метода, поэтому основной тест: HUMAN_SHEETS.
HUMAN_SHEETS = ("01", "02", "03", "05")


def labeled_plans(sheets: list[Sheet] | None = None, subset: str = "all") -> list[Sheet]:
    """Размеченные планы. subset: all | human (01, 02, 03, 05) | agent (остальные)."""
    out = [s for s in (sheets or load_sheets()) if s.labeled and not s.is_legend]
    if subset == "human":
        out = [s for s in out if s.file_name[:2] in HUMAN_SHEETS]
    elif subset == "agent":
        out = [s for s in out if s.file_name[:2] not in HUMAN_SHEETS]
    return out


# ---------------------------------------------------------------- метрики


def iou_xywh(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _match(preds: list[dict], gts: list[Box], iou_thr: float, use_class: bool):
    """Жадное сопоставление по убыванию score. Возвращает список (score, is_tp) и число tp."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i]["score"])
    used = [False] * len(gts)
    flags = []
    for i in order:
        p = preds[i]
        best, bj = iou_thr, -1
        for j, g in enumerate(gts):
            if used[j] or (use_class and g.legend_id != p["legend_id"]):
                continue
            v = iou_xywh(p["bbox"], g.bbox)
            if v >= best:
                best, bj = v, j
        if bj >= 0:
            used[bj] = True
        flags.append((p["score"], bj >= 0))
    return flags, sum(used)


def _ap(flags: list[tuple[float, bool]], n_gt: int) -> float:
    """AP (VOC, все точки) по списку (score, is_tp)."""
    if n_gt == 0:
        return float("nan")
    flags = sorted(flags, key=lambda t: -t[0])
    tp = np.cumsum([f[1] for f in flags]) if flags else np.zeros(0)
    fp = np.cumsum([not f[1] for f in flags]) if flags else np.zeros(0)
    rec = tp / n_gt if len(tp) else np.zeros(0)
    prec = tp / np.maximum(tp + fp, 1e-9) if len(tp) else np.zeros(0)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def prf(tp: int, n_pred: int, n_gt: int) -> tuple[float, float, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt if n_gt else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def in_region(b: list[float], region: tuple[float, float, float, float] | None) -> bool:
    if region is None:
        return True
    cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
    return region[0] <= cx < region[2] and region[1] <= cy < region[3]


# Протокол B (часть листов комплекта размечена): лист 05 единственный размеченный в USC,
# поэтому он делится по x=3500 на две части (100 и 59 объектов), обучение на одной, тест на другой.
USC_SPLIT_X = 3500.0
HALF_LEFT = (0.0, 0.0, USC_SPLIT_X, 1e9)
HALF_RIGHT = (USC_SPLIT_X, 0.0, 1e9, 1e9)


def known_classes(family: str, sheets: list[Sheet] | None = None, subset: str = "all") -> set[str]:
    """Классы, размеченные хотя бы на одном плане семейства.

    Разметка неполная: часть классов легенды на планах есть, но не размечена (например,
    «R» и «M» в квадрате на листе 05). Предсказания таких классов нельзя считать ни верными,
    ни ложными, поэтому основная метрика считается только по известным классам.
    """
    plans = labeled_plans(sheets, subset)
    return {o.legend_id for s in plans if s.family == family for o in s.objects}


def evaluate(
    preds: list[dict],
    gts: list[Box],
    iou_thr: float = 0.5,
    known: set[str] | None = None,
    region: tuple[float, float, float, float] | None = None,
) -> dict:
    """Метрики одного листа (или его части `region`, отбор по центру рамки).

    known: если задано, предсказания классов вне этого множества отбрасываются до подсчета.
    """
    preds = [
        p
        for p in preds
        if in_region(p["bbox"], region) and (known is None or p["legend_id"] in known)
    ]
    gts = [g for g in gts if in_region(g.bbox, region)]
    return _evaluate(preds, gts, iou_thr)


def _evaluate(preds: list[dict], gts: list[Box], iou_thr: float) -> dict:
    """Метрики одного листа.

    - P/R/F1 с учетом класса: предсказание верно, если IoU>=iou_thr с объектом того же legend_id.
    - P/R/F1 без учета класса (локализация символов вообще).
    - AP50 по каждому классу из разметки и mAP50 (среднее по классам разметки).
    Предсказания класса, которого нет в разметке листа, считаются ложными в P,
    но не входят в mAP.
    """
    flags, tp = _match(preds, gts, iou_thr, use_class=True)
    _, tp_ag = _match(preds, gts, iou_thr, use_class=False)
    classes = sorted({g.legend_id for g in gts}, key=lambda s: (len(s), s))
    per_class = {}
    for c in classes:
        pc = [p for p in preds if p["legend_id"] == c]
        gc = [g for g in gts if g.legend_id == c]
        fl, t = _match(pc, gc, iou_thr, use_class=True)
        p, r, f = prf(t, len(pc), len(gc))
        per_class[c] = dict(
            n_gt=len(gc), n_pred=len(pc), tp=t, P=p, R=r, F1=f, AP50=_ap(fl, len(gc))
        )
    P, R, F = prf(tp, len(preds), len(gts))
    Pa, Ra, Fa = prf(tp_ag, len(preds), len(gts))
    aps = [v["AP50"] for v in per_class.values()]
    return dict(
        n_gt=len(gts),
        n_pred=len(preds),
        tp=tp,
        P=P,
        R=R,
        F1=F,
        tp_agnostic=tp_ag,
        P_agnostic=Pa,
        R_agnostic=Ra,
        F1_agnostic=Fa,
        mAP50=float(np.nanmean(aps)) if aps else float("nan"),
        AP50_all=_ap(flags, len(gts)),
        per_class=per_class,
    )


def summarize(results: dict[str, dict]) -> dict:
    """Сводка по нескольким листам: микро-P/R/F1 (сумма tp) и среднее mAP50."""
    tp = sum(r["tp"] for r in results.values())
    tpa = sum(r["tp_agnostic"] for r in results.values())
    npred = sum(r["n_pred"] for r in results.values())
    ngt = sum(r["n_gt"] for r in results.values())
    P, R, F = prf(tp, npred, ngt)
    Pa, Ra, Fa = prf(tpa, npred, ngt)
    return dict(
        n_gt=ngt,
        n_pred=npred,
        tp=tp,
        P=P,
        R=R,
        F1=F,
        P_agnostic=Pa,
        R_agnostic=Ra,
        F1_agnostic=Fa,
        mAP50=float(np.nanmean([r["mAP50"] for r in results.values()])),
        sheets={
            k: {m: r[m] for m in ("n_gt", "n_pred", "tp", "P", "R", "F1", "F1_agnostic", "mAP50")}
            for k, r in results.items()
        },
    )


def nms(preds: list[dict], iou_thr: float = 0.3, class_agnostic: bool = True) -> list[dict]:
    """Простой NMS по score; по умолчанию между классами тоже (один символ = одна метка)."""
    keep: list[dict] = []
    for p in sorted(preds, key=lambda d: -d["score"]):
        if all(
            (not class_agnostic and q["legend_id"] != p["legend_id"])
            or iou_xywh(p["bbox"], q["bbox"]) < iou_thr
            for q in keep
        ):
            keep.append(p)
    return keep


def save_preds(method: str, file_name: str, preds: list[dict]) -> Path:
    path = OUT / method / (Path(file_name).stem + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(preds))
    return path


def load_preds(method: str, file_name: str) -> list[dict]:
    return json.loads((OUT / method / (Path(file_name).stem + ".json")).read_text())
