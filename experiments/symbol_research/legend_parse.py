"""Автоматический разбор листа легенды: рамки значков-образцов без ручной разметки.

Геометрический метод без OCR:
1. Чернила (gray < порога) разбиваются на компоненты связности; слишком крупные
   (рамки таблиц, линейки, печати, логотипы) отбрасываются.
2. Оценивается высота строчной буквы H (мода высот компонентов). Все размеры дальше
   выражаются в долях H, поэтому правила не привязаны к разрешению листа.
3. Компоненты склеиваются анизотропным расширением (по горизонтали шире, чем по
   вертикали) в блоки. Блок из >=3 мелких компонентов-«букв» это текст; остальные блоки
   кандидаты в значки.
4. Для каждого кандидата ищется ближайший справа текстовый блок в той же строке
   (описание). Кандидаты группируются в колонки по левому краю описания (колонка таблицы
   легенды всегда выровнена по левому краю). Колонка принимается, если в ней >=3 строки и
   заметная доля кандидатов «сложные» (рамки, круги, фигуры). Так отбрасываются нумерация
   примечаний и колонки сокращений (одни буквы).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, iou_xywh, legend_for, load_sheets  # noqa: E402

# ------------------------------------------------------------------ параметры (в долях H)


@dataclass
class Params:
    ink_thr: int = 128  # порог бинаризации (gray < thr это чернила)
    max_comp: float = 11.0  # компоненты крупнее (по любой стороне) отбрасываются
    glyph_h: float = 1.3  # компонент не выше этого (любой ширины до max_comp) считается «буквой»
    glyph_w: float = 4.0  # ... но не шире этого (длиннее это уже линия/шина, а не буква)
    glyph_max: float = 1.7  # компонент не крупнее этого считается «буквой»
    text_min_comps: int = 3  # блок из >= стольких букв это текст
    merge_x: float = 1.2  # горизонтальный зазор склейки в блок
    merge_y: float = 0.4  # вертикальный зазор склейки в блок
    icon_max_w: float = 15.0  # максимальные размеры значка
    icon_max_h: float = 9.0
    text_line_h: float = 1.8  # однострочный текст не выше этого (многострочный: много букв)
    text_multi_comps: int = 8
    text_multi_w: float = 3.0
    nest_min: float = 1.3  # рамка, внутри которой лежит другой компонент, не меньше этого (в H)
    rule_max_h: float = 0.35  # блок тоньше этого и длиннее 4H это линия (пунктир, подчеркивание)
    row_overlap: float = 0.3  # минимальное вертикальное перекрытие значка и описания
    col_tol: float = 0.8  # допуск совпадения левого края описаний в колонке
    col_min_rows: int = 3  # минимум строк в колонке
    col_min_complex: float = 0.25  # минимум доли «сложных» значков в колонке
    max_gap: float = 60.0  # максимальный зазор значок-описание
    min_icon: float = 0.7  # значок не мельче этого по обеим сторонам (кроме тонких линий)


@dataclass
class Blob:
    x0: int
    y0: int
    x1: int
    y1: int
    comps: list[int] = field(default_factory=list)
    max_side: float = 0.0
    kind: str = "icon"  # text | icon | rule
    complex: bool = False

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    def bbox(self) -> list[float]:
        return [float(self.x0), float(self.y0), float(self.w), float(self.h)]


def estimate_h(heights: np.ndarray) -> float:
    """Высота «буквы»: мода гистограммы высот компонентов (с небольшим сглаживанием)."""
    m = (heights >= 8) & (heights <= 100)
    hist = np.bincount(heights[m], minlength=101).astype(float)
    hist = np.convolve(hist, np.ones(3) / 3, mode="same")
    return float(hist.argmax())


def _has_nested(bl: Blob, st: np.ndarray, H: float, nest_min: float) -> bool:
    """Есть ли компонент, целиком лежащий внутри рамки другого (буквы в коробке, точка в круге)."""
    if len(bl.comps) < 2:
        return False
    b = st[bl.comps]
    big = (b[:, 2] >= nest_min * H) & (b[:, 3] >= nest_min * H)
    if not big.any():
        return False
    x0, y0, x1, y1 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    inside = (
        (x0[:, None] > x0[None, :]) & (y0[:, None] > y0[None, :])
        & (x1[:, None] < x1[None, :]) & (y1[:, None] < y1[None, :])
    )
    return bool(inside[:, big].any())


def build_blobs(gray: np.ndarray, p: Params) -> tuple[list[Blob], float]:
    ink = (gray < p.ink_thr).astype(np.uint8)
    n, labels, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    w, h = st[:, 2], st[:, 3]
    H = estimate_h(h[1:])
    keep = np.ones(n, bool)
    keep[0] = False
    keep &= (np.maximum(w, h) <= p.max_comp * H)
    lut = keep.astype(np.uint8)
    mask = lut[labels]
    kx = max(1, int(round(p.merge_x * H)))
    ky = max(1, int(round(p.merge_y * H)))
    dil = cv2.dilate(mask, np.ones((ky | 1, kx | 1), np.uint8))
    nb, dl = cv2.connectedComponents(dil, connectivity=8)
    ys, xs = np.nonzero(mask)
    comp_of = labels[ys, xs]
    blob_of = dl[ys, xs]
    uc, first = np.unique(comp_of, return_index=True)
    comp_blob = dict(zip(uc.tolist(), blob_of[first].tolist(), strict=True))
    blobs: dict[int, Blob] = {}
    for c, b in comp_blob.items():
        x, y, cw, ch = st[c, 0], st[c, 1], st[c, 2], st[c, 3]
        bl = blobs.get(b)
        if bl is None:
            bl = blobs[b] = Blob(int(x), int(y), int(x + cw), int(y + ch))
        else:
            bl.x0, bl.y0 = min(bl.x0, int(x)), min(bl.y0, int(y))
            bl.x1, bl.y1 = max(bl.x1, int(x + cw)), max(bl.y1, int(y + ch))
        bl.comps.append(c)
        # «буква»: невысокий компонент (склеенные соседние буквы могут быть широкими) или компактный
        is_glyph = (ch <= p.glyph_h * H and cw <= p.glyph_w * H) or (
            ch <= p.glyph_max * H and cw <= p.glyph_max * H
        )
        bl.max_side = max(bl.max_side, 0.0 if is_glyph else 1.0)
    out = []
    for bl in blobs.values():
        glyph_only = bl.max_side == 0.0
        multi = len(bl.comps) >= p.text_multi_comps and bl.w >= p.text_multi_w * H
        bl.complex = (not glyph_only) or _has_nested(bl, st, H, p.nest_min)
        if bl.h < p.rule_max_h * H and bl.w > 4 * H:
            bl.kind = "rule"
            bl.complex = False
        elif (
            len(bl.comps) >= p.text_min_comps
            and glyph_only
            and not bl.complex
            and (bl.h <= p.text_line_h * H or multi)
        ):
            bl.kind = "text"
        else:
            bl.kind = "icon"
        out.append(bl)
    return out, H


def parse_legend(gray: np.ndarray, params: Params | None = None) -> list[dict]:
    p = params or Params()
    blobs, H = build_blobs(gray, p)
    texts = [b for b in blobs if b.kind == "text"]
    icons = [
        b
        for b in blobs
        if b.kind in ("icon", "rule")
        and b.w <= p.icon_max_w * H
        and b.h <= p.icon_max_h * H
        and (b.h >= p.min_icon * H or b.w >= 1.5 * H)
        and b.w >= 0.3 * H
    ]
    if not texts or not icons:
        return []
    # быстрый поиск описаний справа: сортировка текстов по левому краю
    tx0 = np.array([t.x0 for t in texts])
    order = np.argsort(tx0)
    tx0s = tx0[order]
    cand: list[tuple[Blob, Blob]] = []
    allx0 = np.array([b.x0 for b in blobs])
    ao = np.argsort(allx0)
    allx0s = allx0[ao]
    for ic in icons:
        lo = np.searchsorted(tx0s, ic.x1)
        hi = np.searchsorted(tx0s, ic.x1 + p.max_gap * H)
        best, best_gap = None, 1e9
        for k in order[lo:hi]:
            t = texts[k]
            ov = min(ic.y1, t.y1) - max(ic.y0, t.y0)
            if ov < p.row_overlap * min(ic.h, t.h) or ov <= 0:
                continue
            gap = t.x0 - ic.x1
            if gap < best_gap:
                best, best_gap = t, gap
        if best is None:
            continue
        # между значком и описанием не должно быть другого блока (чужого значка/линии)
        l2 = np.searchsorted(allx0s, ic.x1)
        h2 = np.searchsorted(allx0s, best.x0)
        blocked = False
        for k in ao[l2:h2]:
            o = blobs[k]
            if o is ic or o is best:
                continue
            ov = min(ic.y1, o.y1) - max(ic.y0, o.y0)
            if ov > 0.5 * ic.h and o.x1 <= best.x0:
                blocked = True
                break
        if blocked:
            continue
        cand.append((ic, best))
    if not cand:
        return []
    # колонки по левому краю описания
    cand.sort(key=lambda c: c[1].x0)
    cols: list[list[tuple[Blob, Blob]]] = []
    for c in cand:
        if cols and c[1].x0 - cols[-1][-1][1].x0 <= p.col_tol * H:
            cols[-1].append(c)
        else:
            cols.append([c])
    res: list[dict] = []
    for col in cols:
        if len(col) < p.col_min_rows:
            continue
        share = sum(ic.complex for ic, _ in col) / len(col)
        if share < p.col_min_complex:
            continue
        for ic, t in col:
            score = 0.5 * float(ic.complex) + 0.5 * share
            res.append(dict(bbox=ic.bbox(), row_bbox=t.bbox(), score=round(score, 3),
                            _col=float(np.median([c[1].x0 for c in col]))))
    # порядок: колонки слева направо, внутри сверху вниз
    res.sort(key=lambda d: (d["_col"], d["bbox"][1]))
    for d in res:
        d.pop("_col")
    return res


# ------------------------------------------------------------------ проверка и визуализация


def match_gt(found: list[dict], gt: list, thr: float) -> tuple[int, int, list, list]:
    """Возвращает (число найденных GT, число найденных, пропущенные GT, лишние найденные)."""
    best_for_gt = []
    for g in gt:
        v = max((iou_xywh(f["bbox"], g.bbox) for f in found), default=0.0)
        best_for_gt.append(v)
    hit = sum(v >= thr for v in best_for_gt)
    good_found = 0
    extra = []
    for f in found:
        v = max((iou_xywh(f["bbox"], g.bbox) for g in gt), default=0.0)
        if v >= thr:
            good_found += 1
        else:
            extra.append(f)
    missed = [g for g, v in zip(gt, best_for_gt, strict=True) if v < thr]
    return hit, good_found, missed, extra


def contact_sheet(
    gray: np.ndarray, items: list[dict], cell: int = 120, ncol: int = 10
) -> np.ndarray:
    n = len(items)
    nrow = max(1, (n + ncol - 1) // ncol)
    sheet = np.full((nrow * (cell + 20), ncol * cell), 255, np.uint8)
    for k, it in enumerate(items):
        x, y, w, h = (int(round(v)) for v in it["bbox"])
        crop = gray[max(0, y - 4): y + h + 4, max(0, x - 4): x + w + 4]
        s = min((cell - 8) / crop.shape[1], (cell - 8) / crop.shape[0])
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        r, c = divmod(k, ncol)
        oy, ox = r * (cell + 20) + 18, c * cell + 4
        sheet[oy: oy + crop.shape[0], ox: ox + crop.shape[1]] = crop
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(sheet, str(k), (c * cell + 4, r * (cell + 20) + 14), font, 0.5, 0, 1)
        p0 = (c * cell, r * (cell + 20))
        p1 = ((c + 1) * cell - 1, (r + 1) * (cell + 20) - 1)
        cv2.rectangle(sheet, p0, p1, 180, 1)
    return sheet


def overlay(gray: np.ndarray, items: list[dict], width: int = 2400) -> np.ndarray:
    k = width / gray.shape[1]
    small = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    im = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    for i, it in enumerate(items):
        x, y, w, h = (v * k for v in it["bbox"])
        pa, pb = (int(x) - 2, int(y) - 2), (int(x + w) + 2, int(y + h) + 2)
        cv2.rectangle(im, pa, pb, (0, 0, 255), 2)
        org = (int(x + w) + 4, int(y + h))
        cv2.putText(im, str(i), org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
    return im


def run_family(family: str, out_dir: Path, sheets=None) -> dict:
    sheet, gt = legend_for(family, sheets)
    gray = sheet.load_gray()
    found = parse_legend(gray)
    items = [
        dict(
            legend_id=f"{family}_{k:02d}",
            bbox=f["bbox"],
            desc_bbox=f["row_bbox"],
            score=f["score"],
        )
        for k, f in enumerate(found)
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{family}.json").write_text(json.dumps(items, indent=1))
    cv2.imwrite(str(out_dir / f"{family}_contact.png"), contact_sheet(gray, items))
    cv2.imwrite(str(out_dir / f"{family}_overlay.png"), overlay(gray, items))
    info = dict(family=family, n_found=len(items), n_gt=len(gt))
    if gt:
        for thr in (0.5, 0.3):
            hit, good, missed, extra = match_gt(found, gt, thr)
            info[f"recall@{thr}"] = hit / len(gt)
            info[f"precision@{thr}"] = good / len(found) if found else 0.0
            if thr == 0.3:
                info["missed"] = [(g.legend_id, [round(v) for v in g.bbox]) for g in missed]
                info["extra"] = [[round(v) for v in f["bbox"]] for f in extra]
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="worcester,usc,canada,wisd,va")
    ap.add_argument("--out", default=str(OUT / "legend_auto"))
    a = ap.parse_args()
    sheets = load_sheets()
    for fam in a.families.split(","):
        info = run_family(fam, Path(a.out), sheets)
        print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
