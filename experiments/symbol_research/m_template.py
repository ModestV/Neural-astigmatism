"""H1: сопоставление с шаблонами из легенды (multi-scale, 4 поворота, NCC).

    uv run --project ../.. python m_template.py [--sheets 01,02,03,05] [--out-method template]

Схема:
  1. Эталон из легенды: серые чернила (255-gray)/255, обрезка до рамки чернил (без бинаризации,
     чтобы не терять тонкие диагонали), повороты 0/90/180/270 (симметричные дубли отбрасываются).
     Сплошные плашки (доля чернил > solid_ink_frac) получают широкую белую кайму.
  2. Грубый проход: изображение в 2 раза меньше, cv2.matchTemplate (TM_CCOEFF_NORMED) по слегка
     размытым чернилам для каждого (поворот, масштаб); локальные максимумы выше coarse_thr.
  3. Уточнение в полном разрешении в окне вокруг каждого кандидата: масштабы +-scale_refine_step,
     сдвиг +-refine_margin px. Балл = NCC по размытым чернилам и/или по размытым скелетам
     (скелет делает сравнение нечувствительным к толщине линий) умножается на покрытие скелета
     эталона чернилами плана (отсекает шаблон, часть которого на плане отсутствует).
  4. NMS внутри класса; подавление символа, вложенного в более крупный символ с почти тем же баллом
     (например, круг с двумя линиями внутри такого же круга с выводами); NMS между классами; порог.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import common
import cv2
import numpy as np
from skimage.morphology import skeletonize

# ------------------------------------------------------------------ параметры


@dataclass
class Params:
    scales: tuple[float, ...] = (0.8, 0.95, 1.1)  # грубый проход; уточнение +-scale_refine_step
    scale_refine_step: float = 0.075  # покрывает 0.725..1.175 с шагом 0.075
    blur_sigma: float = 1.0  # размытие в полном разрешении (px)
    coarse_sigma: float = 0.7  # размытие на уменьшенном в 2 раза изображении
    coarse_thr: float = 0.45  # порог отклика в грубом проходе
    coarse_max_peaks: int = 300  # максимум пиков на (класс, поворот, масштаб)
    peak_radius_frac: float = 0.5  # радиус подавления пиков, доля меньшей стороны шаблона
    refine_margin: int = 4  # окно поиска вокруг кандидата, px полного разрешения
    min_ink_frac: float = 0.35  # доля чернил шаблона, которая должна быть в окне
    ink_thr: float = 0.25  # порог по (255-gray)/255 при обрезке эталона до рамки чернил
    pad: int = 1  # белая кайма вокруг обрезанного эталона
    sym_dup_thr: float = 0.97  # корреляция повернутого эталона с предыдущим, выше которой дубль
    score_thr: float = 0.75  # итоговый порог score
    class_nms_iou: float = 0.3
    cross_nms_iou: float = 0.3
    mode: str = "mean"  # балл уточнения: plain (по размытым чернилам), skel (по скелетам), mean
    skel_sigma: float = 1.2  # размытие скелета (после утолщения до 3 px)
    bin_thr: float = 0.25  # порог бинаризации для скелета
    solid_ink_frac: float = 0.5  # шаблон с долей чернил выше этой считается сплошным (плашка)
    solid_pad: int = 6  # белая кайма вокруг плашки (иначе она совпадает с участком любой стены)
    cov_dilate: int = 2  # радиус (px) утолщения чернил плана при расчете покрытия скелета эталона
    use_coverage: bool = True  # множитель: доля скелета эталона, покрытая чернилами плана
    contain_delta: float = 0.03  # вложенный символ подавляется, если score ниже не более чем на это
    contain_frac: float = 0.8  # доля площади вложенного бокса внутри большего
    isolate_solid: bool = True  # плашка принимается, только если ее компонента чернил изолирована
    iso_ext: float = 0.25  # допустимый выход компоненты за рамку, доля большей стороны символа
    iso_thr: float = 0.5  # порог чернил для компонент (серые стены не склеивают плашку с соседями)
    iso_ring: float = 1.0  # радиус поиска соседних крупных пятен (доля большей стороны)
    iso_neighbor_area: float = 0.5  # доля площади рамки: пятно не меньше считается соседом
    iso_fill: float = 0.75  # заполнение рамки компонентой: штрих буквы тоньше плашки
    iso_open_frac: float = 0.45  # ядро размыкания, доля меньшей стороны (срезает выноски, стрелки)
    workers: int = 10


# ------------------------------------------------------------------ эталоны


def to_ink(gray: np.ndarray) -> np.ndarray:
    return (255.0 - gray.astype(np.float32)) / 255.0


def prep_template(crop: np.ndarray, p: Params) -> tuple[np.ndarray, int, bool]:
    """Серый эталон -> (чернила float32 с каймой, ширина каймы, признак сплошного эталона)."""
    ink = to_ink(crop)
    ys, xs = np.where(ink > p.ink_thr)
    if len(xs) == 0:
        return np.pad(ink, p.pad), p.pad, False
    ink = ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    solid = float((ink > p.bin_thr).mean()) > p.solid_ink_frac
    pad = p.solid_pad if solid else p.pad
    return np.pad(ink, pad), pad, solid


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def rotations(t: np.ndarray, p: Params) -> list[tuple[int, np.ndarray]]:
    """Повороты 0/90/180/270; повороты, неотличимые от уже взятых (симметрия), опускаются."""
    out: list[tuple[int, np.ndarray]] = []
    for k in range(4):
        r = np.ascontiguousarray(np.rot90(t, k))
        dup = any(q.shape == r.shape and _ncc(q, r) > p.sym_dup_thr for _, q in out)
        if not dup:
            out.append((k * 90, r))
    return out


def rescale(t: np.ndarray, s: float) -> np.ndarray:
    h, w = t.shape
    nw, nh = max(3, round(w * s)), max(3, round(h * s))
    return cv2.resize(t, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)


def blurred(t: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(t, (0, 0), sigma) if sigma > 0 else t


_K3 = np.ones((3, 3), np.uint8)


def skeleton_bin(ink: np.ndarray, p: Params) -> np.ndarray:
    return skeletonize(ink > p.bin_thr).astype(np.uint8)


def skeleton_map(ink: np.ndarray, p: Params) -> np.ndarray:
    """Чернила -> скелет, утолщенный до ~3 px и размытый: сравнение не зависит от толщины линий."""
    sk = cv2.dilate(skeleton_bin(ink, p), _K3).astype(np.float32)
    return cv2.GaussianBlur(sk, (0, 0), p.skel_sigma)


class Bank:
    """Кэш шаблонов (класс, поворот, масштаб) -> (размытые чернила, размытый скелет, скелет 0/1)."""

    def __init__(self, p: Params):
        self.p = p
        self.cache: dict[tuple, tuple[np.ndarray, np.ndarray | None, np.ndarray | None]] = {}

    def get(self, key: tuple, t: np.ndarray, s: float, need_skel: bool):
        k = (key, round(s, 3), need_skel)
        v = self.cache.get(k)
        if v is None:
            r = rescale(t, s)
            plain = blurred(r, self.p.blur_sigma)
            sb = skeleton_bin(r, self.p) if need_skel else None
            sk = None
            if sb is not None:
                sk = cv2.GaussianBlur(
                    cv2.dilate(sb, _K3).astype(np.float32), (0, 0), self.p.skel_sigma
                )
            v = (plain, sk, sb)
            self.cache[k] = v
        return v


def _peaks(
    res: np.ndarray, thr: float, radius: int, max_peaks: int
) -> list[tuple[int, int, float]]:
    """Локальные максимумы карты отклика выше thr (окно (2r+1)^2), не более max_peaks лучших."""
    k = 2 * radius + 1
    mx = cv2.dilate(res, np.ones((k, k), np.uint8))
    ys, xs = np.where((res >= mx) & (res > thr))
    if len(xs) == 0:
        return []
    v = res[ys, xs]
    order = np.argsort(-v)[:max_peaks]
    return [(int(xs[i]), int(ys[i]), float(v[i])) for i in order]


# ------------------------------------------------------------------ поиск


class Plan:
    """Подготовленный лист: чернила в полном разрешении и в 1/2, размытые; скелет (лениво)."""

    def __init__(self, gray: np.ndarray, p: Params):
        ink = to_ink(gray)
        self.full = cv2.GaussianBlur(ink, (0, 0), p.blur_sigma)
        h, w = ink.shape
        half = cv2.resize(ink, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        self.half = cv2.GaussianBlur(half, (0, 0), p.coarse_sigma)
        self.bin = (ink > p.iso_thr).astype(np.uint8)
        self.skel = skeleton_map(ink, p) if p.mode in ("skel", "mean") else None
        k = 2 * p.cov_dilate + 1
        self.dil = cv2.dilate((ink > p.bin_thr).astype(np.uint8), np.ones((k, k), np.uint8))
        self.h, self.w = h, w


def _coarse(pl: Plan, bank: Bank, key: tuple, t: np.ndarray, p: Params):
    """Один эталон (после поворота) на всех масштабах: кандидаты (scale, cx, cy, score)."""
    out = []
    for s in p.scales:
        tc = blurred(rescale(t, s * 0.5), p.coarse_sigma)
        th, tw = tc.shape
        if th >= pl.half.shape[0] or tw >= pl.half.shape[1]:
            continue
        res = cv2.matchTemplate(pl.half, tc, cv2.TM_CCOEFF_NORMED)
        # защита от пустых окон: в окне должна быть доля чернил шаблона
        dens = cv2.boxFilter(
            pl.half, -1, (tw, th), normalize=True, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT
        )
        dens = dens[: res.shape[0], : res.shape[1]]
        res = np.where(dens > p.min_ink_frac * tc.mean(), res, 0).astype(np.float32)
        np.nan_to_num(res, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        radius = max(2, int(p.peak_radius_frac * min(th, tw)))
        for x, y, v in _peaks(res, p.coarse_thr, radius, p.coarse_max_peaks):
            out.append((s, 2 * (x + tw / 2), 2 * (y + th / 2), v))
    return out


def _refine(pl: Plan, bank: Bank, key: tuple, t: np.ndarray, pad: int, cand: tuple, p: Params):
    """Уточнение кандидата в полном разрешении: (score, bbox xywh без каймы, scale) или None."""
    s0, cx, cy, _ = cand
    best = None
    m = p.refine_margin
    use_plain = p.mode in ("plain", "mean")
    use_skel = p.mode in ("skel", "mean")
    for s in (s0 - p.scale_refine_step, s0, s0 + p.scale_refine_step):
        tp, tk, tsb = bank.get(key, t, s, use_skel)
        th, tw = tp.shape
        x0 = int(round(cx - tw / 2)) - m
        y0 = int(round(cy - th / 2)) - m
        x1, y1 = x0 + tw + 2 * m, y0 + th + 2 * m
        if x0 < 0 or y0 < 0 or x1 > pl.w or y1 > pl.h:
            continue
        roi = pl.full[y0:y1, x0:x1]
        if roi.std() < 1e-4:
            continue
        maps = []
        if use_plain:
            maps.append(cv2.matchTemplate(roi, tp, cv2.TM_CCOEFF_NORMED))
        if use_skel:
            rk = pl.skel[y0:y1, x0:x1]
            if rk.std() < 1e-4:
                continue
            maps.append(cv2.matchTemplate(rk, tk, cv2.TM_CCOEFF_NORMED))
        res = np.mean(maps, axis=0)
        np.nan_to_num(res, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        _, mv, _, ml = cv2.minMaxLoc(res)
        bx, by = x0 + ml[0], y0 + ml[1]
        score = float(mv)
        if p.use_coverage and tsb is not None and tsb.sum() > 0:
            cov = float((pl.dil[by : by + th, bx : bx + tw] & tsb).sum()) / float(tsb.sum())
            score *= cov
        if best is None or score > best[0]:
            ps = pad * s
            box = [float(bx + ps), float(by + ps), float(tw - 2 * ps), float(th - 2 * ps)]
            best = (score, box, s)
    return best


def is_isolated(pl: Plan, box: list[float], p: Params) -> bool:
    """Компонента чернил плана (после размыкания), на которую легла рамка, почти не выходит за нее.

    Для залитых эталонов: отсекает штрихи крупных букв и толстые стены, у которых компонента
    намного больше символа. Размыкание ядром ~ толщины символа срезает тонкие выноски.
    """
    x, y, w, h = box
    m = int(p.iso_ring * max(w, h)) + 3
    rx0, ry0 = max(0, int(x) - m), max(0, int(y) - m)
    rx1, ry1 = min(pl.w, int(x + w) + m + 1), min(pl.h, int(y + h) + m + 1)
    roi = pl.bin[ry0:ry1, rx0:rx1]
    k = max(3, round(p.iso_open_frac * min(w, h)))
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    if n <= 1:
        return False
    bx0, by0 = int(round(x)) - rx0, int(round(y)) - ry0
    inside = lab[max(0, by0) : by0 + int(round(h)), max(0, bx0) : bx0 + int(round(w))]
    cnt = np.bincount(inside.ravel(), minlength=n)
    cnt[0] = 0
    lbl = int(cnt.argmax())
    if cnt[lbl] == 0:
        return False
    # соседние крупные пятна (буквы заголовка, ряд плашек) означают текст, а не символ
    others = st[1:, cv2.CC_STAT_AREA].copy()
    others[lbl - 1] = 0
    if others.size and others.max() >= p.iso_neighbor_area * w * h:
        return False
    if st[lbl][cv2.CC_STAT_AREA] < p.iso_fill * w * h:
        return False
    cx, cy, cw, ch = st[lbl][:4]
    ext = p.iso_ext * max(w, h)
    return (
        bx0 - cx <= ext
        and by0 - cy <= ext
        and (cx + cw) - (bx0 + w) <= ext
        and (cy + ch) - (by0 + h) <= ext
    )


def suppress_contained(preds: list[dict], delta: float, frac: float) -> list[dict]:
    """Убирает предсказание A, если его бокс (на frac площади) лежит внутри бокса большего B
    другого класса и score(B) >= score(A) - delta (меньший символ - часть большего)."""
    if not preds:
        return preds
    b = np.array([d["bbox"] for d in preds], dtype=np.float64)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    area = b[:, 2] * b[:, 3]
    sc = np.array([d["score"] for d in preds])
    cls = np.array([d["legend_id"] for d in preds])
    keep = []
    for i in range(len(preds)):
        iw = np.clip(np.minimum(x2[i], x2) - np.maximum(x1[i], x1), 0, None)
        ih = np.clip(np.minimum(y2[i], y2) - np.maximum(y1[i], y1), 0, None)
        inside = iw * ih >= frac * area[i]
        bigger = area > area[i] * 1.15
        rival = inside & bigger & (cls != cls[i]) & (sc >= sc[i] - delta)
        if not rival.any():
            keep.append(preds[i])
    return keep


def detect_candidates(gray: np.ndarray, templates: dict[str, np.ndarray], **params) -> list[dict]:
    """Все кандидаты после NMS внутри класса, без порога и без межклассовой обработки.

    Каждый элемент: dict(bbox, legend_id, score, rot, scale).
    """
    p = Params(**params)
    pl = Plan(gray, p)
    bank = Bank(p)
    base = {k: prep_template(v, p) for k, v in templates.items()}

    def work(item):
        key0, (t0, pad, solid) = item
        lid = key0.split("#")[0]  # ключ "24#1": дополнительный шаблон класса 24
        found = []
        for rot, t in rotations(t0, p):
            key = (key0, rot)
            for cand in _coarse(pl, bank, key, t, p):
                r = _refine(pl, bank, key, t, pad, cand, p)
                if r is None:
                    continue
                score, bbox, s = r
                if solid and p.isolate_solid and not is_isolated(pl, bbox, p):
                    continue
                found.append(dict(bbox=bbox, legend_id=lid, score=score, rot=rot, scale=s))
        return common.nms(found, p.class_nms_iou, class_agnostic=False)

    with ThreadPoolExecutor(p.workers) as ex:
        out = list(ex.map(work, base.items()))
    return [d for lst in out for d in lst]


def postprocess(cands: list[dict], **params) -> list[dict]:
    """Порог, подавление вложенных символов, NMS между классами."""
    p = Params(**params)
    c = [d for d in cands if d["score"] >= p.score_thr]
    c = suppress_contained(c, p.contain_delta, p.contain_frac)
    return common.nms(c, p.cross_nms_iou, class_agnostic=True)


def detect(gray: np.ndarray, templates: dict[str, np.ndarray], **params) -> list[dict]:
    """Поиск эталонов на листе. gray: uint8, чернила темные. templates: legend_id -> серая вырезка.

    Возвращает [dict(bbox=[x,y,w,h], legend_id=str, score=float, rot=int, scale=float)] в px листа.
    """
    post_keys = {"score_thr", "cross_nms_iou", "contain_delta", "contain_frac"}
    post = {k: v for k, v in params.items() if k in post_keys}
    search = {k: v for k, v in params.items() if k not in post_keys}
    cands = detect_candidates(gray, templates, **search)
    return postprocess(cands, **post)


# ------------------------------------------------------------------ CLI


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", default="01,02,03,05")
    ap.add_argument("--out-method", default="template")
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--mode", default=None, choices=["plain", "skel", "mean"])
    a = ap.parse_args()
    want = set(a.sheets.split(","))
    params: dict = {}
    if a.thr is not None:
        params["score_thr"] = a.thr
    if a.mode is not None:
        params["mode"] = a.mode
    for s in common.labeled_plans():
        if s.file_name[:2] not in want:
            continue
        t0 = time.time()
        preds = detect(s.load_gray(), common.legend_crops(s.family), **params)
        out = [dict(bbox=d["bbox"], legend_id=d["legend_id"], score=d["score"]) for d in preds]
        path = common.save_preds(a.out_method, s.file_name, out)
        print(f"{s.file_name}: {len(out)} предсказаний, {time.time() - t0:.0f} s -> {path}")


if __name__ == "__main__":
    main()
