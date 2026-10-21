"""Инструмент ручной (визуальной) доразметки неразмеченных планов.

Автоматика только предлагает кандидатов; каждое решение принимает разметчик, глядя на
картинки. Состояние листа: outputs/research/relabel/<stem>.json, список рамок
{id, bbox [x,y,w,h], legend_id, status: cand|ok|rej, source: auto|manual, score}.
Картинки для просмотра пишутся в outputs/research/relabel/view/.

Команды (запуск: uv run --project ../.. python label_tool.py <cmd> ...):
  legend <family>                     лист эталонов: значок + вырезка описания из легенды
  init <stem> [--thr 0.6] [--skip ID,ID]  кандидаты шаблонным методом (порог ниже обычного)
  grid <stem> [--cls ID] [--status cand] [--page N]  сетка вырезок кандидатов с номерами
  set <stem> ok|rej|cand ID [ID ...]  статус рамок; 'all-cand-ok CLS' нет: только явные id
  relabel <stem> CLS ID [ID ...]      сменить класс рамок (статус -> ok)
  tiles <stem>                        сколько тайлов и их координаты
  tile <stem> R C                     тайл листа 1400x1400 в полном разрешении, рамки ok/cand,
                                      сетка с подписями координат каждые 100 px
  add <stem> CLS X Y                  добавить символ класса CLS рядом с точкой (X,Y) листа:
                                      рамка прилипает к лучшему совпадению эталона в окне
  addbox <stem> CLS X Y W H           добавить рамку как есть (если add не справился)
  delete <stem> ID [ID ...]           удалить рамку
  stats <stem>                        сводка по классам и статусам
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import common
import m_template
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RELABEL = common.OUT / "relabel"
VIEW = RELABEL / "view"
TILE = 1400
TILE_STEP = 1200


def _font(size: int) -> ImageFont.ImageFont:
    for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"):
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def sheet_by_stem(stem: str) -> common.Sheet:
    for s in common.load_sheets():
        if Path(s.file_name).stem == stem or s.file_name.startswith(stem):
            return s
    raise SystemExit(f"нет листа {stem}")


def legend_items(family: str) -> list[dict]:
    """Эталоны семейства: ручная легенда, если есть, иначе автоматический разбор."""
    sheet, items = common.legend_for(family)
    if items:
        return [dict(legend_id=b.legend_id, bbox=b.bbox, desc_bbox=None) for b in items]
    return json.loads((common.OUT / "legend_auto" / f"{family}.json").read_text())


def legend_templates(family: str) -> dict[str, np.ndarray]:
    sheet, _ = common.legend_for(family)
    img = sheet.load_gray()
    out = {}
    for it in legend_items(family):
        x, y, w, h = (int(round(v)) for v in it["bbox"])
        out[it["legend_id"]] = img[y : y + h, x : x + w].copy()
    return out


def path_of(s: common.Sheet) -> Path:
    return RELABEL / f"{Path(s.file_name).stem}.json"


def load(s: common.Sheet) -> list[dict]:
    p = path_of(s)
    return json.loads(p.read_text()) if p.exists() else []


def save(s: common.Sheet, boxes: list[dict]) -> None:
    RELABEL.mkdir(parents=True, exist_ok=True)
    path_of(s).write_text(json.dumps(boxes, indent=0))


def _next_id(boxes: list[dict]) -> int:
    return max((b["id"] for b in boxes), default=-1) + 1


# ------------------------------------------------------------------ команды


def cmd_legend(family: str) -> None:
    sheet, _ = common.legend_for(family)
    img = Image.open(sheet.path).convert("L")
    items = legend_items(family)
    font = _font(18)
    rows = []
    for it in items:
        x, y, w, h = it["bbox"]
        sym = img.crop((int(x) - 6, int(y) - 6, int(x + w) + 6, int(y + h) + 6))
        sym.thumbnail((160, 110))
        if it.get("desc_bbox"):
            dx, dy, dw, dh = it["desc_bbox"]
            desc = img.crop((int(dx) - 4, int(dy) - 4, int(dx + dw) + 4, int(dy + dh) + 4))
        else:  # ручная легенда: описание правее значка
            desc = img.crop((int(x + w) + 10, int(y) - 10, int(x + w) + 1100, int(y + h) + 10))
        desc.thumbnail((900, 110))
        rows.append((it["legend_id"], sym, desc))
    per_page = 24
    VIEW.mkdir(parents=True, exist_ok=True)
    for page in range(0, len(rows), per_page):
        chunk = rows[page : page + per_page]
        canvas = Image.new("RGB", (1300, 120 * len(chunk)), "white")
        d = ImageDraw.Draw(canvas)
        for i, (lid, sym, desc) in enumerate(chunk):
            yy = i * 120
            d.text((5, yy + 45), lid, fill="red", font=font)
            canvas.paste(sym, (170, yy + 5))
            canvas.paste(desc, (370, yy + 5))
            d.line([(0, yy + 118), (1300, yy + 118)], fill=(200, 200, 200))
        out = VIEW / f"legend_{family}_{page // per_page}.png"
        canvas.save(out)
        print(out)


def cmd_init(stem: str, thr: float, skip: list[str]) -> None:
    s = sheet_by_stem(stem)
    tpl = {k: v for k, v in legend_templates(s.family).items() if k not in skip}
    preds = m_template.detect(s.load_gray(), tpl, score_thr=thr)
    boxes = [
        dict(id=i, bbox=[round(v, 1) for v in p["bbox"]], legend_id=p["legend_id"],
             status="cand", source="auto", score=round(p["score"], 3))
        for i, p in enumerate(sorted(preds, key=lambda d: (d["legend_id"], -d["score"])))
    ]
    save(s, boxes)
    print(f"{s.file_name}: {len(boxes)} кандидатов")
    cmd_stats(stem)


def cmd_grid(stem: str, cls: str | None, status: str | None, page: int) -> None:
    s = sheet_by_stem(stem)
    boxes = [b for b in load(s) if (cls is None or b["legend_id"] == cls)
             and (status is None or b["status"] == status)]
    img = Image.open(s.path).convert("RGB")
    cell, per = 260, 30
    chunk = boxes[page * per : (page + 1) * per]
    if not chunk:
        print("пусто")
        return
    font = _font(16)
    head = 150
    canvas = Image.new("RGB", (cell * 6, head + cell * ((len(chunk) + 5) // 6)), "white")
    d = ImageDraw.Draw(canvas)
    if cls is not None:  # эталон класса сверху
        tpl = legend_templates(s.family)[cls]
        t = Image.fromarray(tpl).convert("RGB")
        t.thumbnail((130, 130))
        canvas.paste(t, (10, 10))
        d.text((160, 60), f"эталон {cls}", fill="blue", font=font)
    d.text((400, 60), f"{s.file_name} стр. {page} из {(len(boxes) - 1) // per}", fill="black", font=font)
    for i, b in enumerate(chunk):
        x, y, w, h = b["bbox"]
        side = max(w, h) * 2.4
        cx, cy = x + w / 2, y + h / 2
        crop = img.crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)))
        k = (cell - 30) / side
        crop = crop.resize((cell - 30, cell - 30))
        cd = ImageDraw.Draw(crop)
        bx0, by0 = (x - (cx - side / 2)) * k, (y - (cy - side / 2)) * k
        color = {"ok": "green", "rej": "gray", "cand": "red"}[b["status"]]
        cd.rectangle([bx0, by0, bx0 + w * k, by0 + h * k], outline=color, width=2)
        X, Y = (i % 6) * cell, head + (i // 6) * cell
        canvas.paste(crop, (X + 5, Y + 25))
        d.text((X + 5, Y + 3), f"#{b['id']} {b['legend_id']} {b['score']}", fill=color, font=font)
    VIEW.mkdir(parents=True, exist_ok=True)
    out = VIEW / f"grid_{Path(s.file_name).stem}_{cls or 'all'}_{status or 'any'}_{page}.png"
    canvas.save(out)
    print(out, f"(рамок в выборке {len(boxes)})")


def cmd_set(stem: str, status: str, ids: list[int]) -> None:
    s = sheet_by_stem(stem)
    boxes = load(s)
    want = set(ids)
    n = 0
    for b in boxes:
        if b["id"] in want:
            b["status"] = status
            n += 1
    save(s, boxes)
    print(f"изменено {n}")


def cmd_relabel(stem: str, cls: str, ids: list[int]) -> None:
    s = sheet_by_stem(stem)
    boxes = load(s)
    for b in boxes:
        if b["id"] in ids:
            b["legend_id"], b["status"] = cls, "ok"
    save(s, boxes)
    print("готово")


def _tiles(s: common.Sheet) -> list[tuple[int, int, int, int]]:
    out = []
    for r, y in enumerate(range(0, s.height, TILE_STEP)):
        for c, x in enumerate(range(0, s.width, TILE_STEP)):
            out.append((r, c, x, y))
    return out


def cmd_tiles(stem: str) -> None:
    s = sheet_by_stem(stem)
    t = _tiles(s)
    print(f"{s.file_name}: {s.width}x{s.height}, тайлов {len(t)}, строк {t[-1][0] + 1}, столбцов {t[-1][1] + 1}")


def cmd_tile(stem: str, r: int, c: int) -> None:
    s = sheet_by_stem(stem)
    x0, y0 = c * TILE_STEP, r * TILE_STEP
    img = Image.open(s.path).convert("RGB").crop((x0, y0, x0 + TILE, y0 + TILE))
    d = ImageDraw.Draw(img)
    font = _font(14)
    for gx in range(0, TILE, 100):
        d.line([(gx, 0), (gx, TILE)], fill=(170, 200, 255), width=1)
        d.text((gx + 2, 2), str(x0 + gx), fill=(60, 90, 220), font=font)
    for gy in range(0, TILE, 100):
        d.line([(0, gy), (TILE, gy)], fill=(170, 200, 255), width=1)
        d.text((2, gy + 2), str(y0 + gy), fill=(60, 90, 220), font=font)
    for b in load(s):
        if b["status"] == "rej":
            continue
        x, y, w, h = b["bbox"]
        if x + w < x0 or y + h < y0 or x > x0 + TILE or y > y0 + TILE:
            continue
        color = "green" if b["status"] == "ok" else "red"
        d.rectangle([x - x0, y - y0, x + w - x0, y + h - y0], outline=color, width=2)
        d.text((x - x0, y - y0 - 16), f"#{b['id']}:{b['legend_id']}", fill=color, font=font)
    VIEW.mkdir(parents=True, exist_ok=True)
    out = VIEW / f"tile_{Path(s.file_name).stem}_{r}_{c}.png"
    img.save(out)
    print(out, f"x {x0}..{x0 + TILE}, y {y0}..{y0 + TILE}")


def cmd_add(stem: str, cls: str, x: float, y: float) -> None:
    s = sheet_by_stem(stem)
    tpl = legend_templates(s.family)[cls]
    r = int(max(tpl.shape) * 1.6) + 20
    gray = s.load_gray()
    x0, y0 = max(0, int(x) - r), max(0, int(y) - r)
    win = np.ascontiguousarray(gray[y0 : int(y) + r, x0 : int(x) + r])
    cands = m_template.detect(win, {cls: tpl}, score_thr=0.0, workers=1)
    best = None
    for p in cands:
        bx, by, bw, bh = p["bbox"]
        cx, cy = bx + bw / 2 + x0, by + bh / 2 + y0
        dist = np.hypot(cx - x, cy - y)
        if dist <= max(bw, bh) and (best is None or p["score"] > best["score"]):
            best = {**p, "bbox": [bx + x0, by + y0, bw, bh]}
    if best is None:
        print("не нашел совпадения рядом с точкой: укажи точку точнее или используй addbox")
        return
    boxes = load(s)
    nb = dict(id=_next_id(boxes), bbox=[round(v, 1) for v in best["bbox"]], legend_id=cls,
              status="ok", source="manual", score=round(best["score"], 3))
    boxes.append(nb)
    save(s, boxes)
    print(f"добавлен #{nb['id']} {cls} bbox={nb['bbox']} score={nb['score']}")


def cmd_addbox(stem: str, cls: str, x: float, y: float, w: float, h: float) -> None:
    s = sheet_by_stem(stem)
    boxes = load(s)
    nb = dict(id=_next_id(boxes), bbox=[x, y, w, h], legend_id=cls, status="ok",
              source="manual", score=1.0)
    boxes.append(nb)
    save(s, boxes)
    print(f"добавлен #{nb['id']}")


def cmd_delete(stem: str, ids: list[int]) -> None:
    s = sheet_by_stem(stem)
    boxes = [b for b in load(s) if b["id"] not in ids]
    save(s, boxes)
    print("удалено")


def cmd_stats(stem: str) -> None:
    from collections import Counter

    s = sheet_by_stem(stem)
    boxes = load(s)
    c = Counter((b["legend_id"], b["status"]) for b in boxes)
    by_cls: dict[str, dict] = {}
    for (lid, st), n in c.items():
        by_cls.setdefault(lid, {})[st] = n
    for lid in sorted(by_cls):
        print(lid, by_cls[lid])
    print("всего", Counter(b["status"] for b in boxes))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("legend").add_argument("family")
    p = sub.add_parser("init")
    p.add_argument("stem")
    p.add_argument("--thr", type=float, default=0.6)
    p.add_argument("--skip", default="")
    p = sub.add_parser("grid")
    p.add_argument("stem")
    p.add_argument("--cls")
    p.add_argument("--status")
    p.add_argument("--page", type=int, default=0)
    p = sub.add_parser("set")
    p.add_argument("stem")
    p.add_argument("status", choices=["ok", "rej", "cand"])
    p.add_argument("ids", type=int, nargs="+")
    p = sub.add_parser("relabel")
    p.add_argument("stem")
    p.add_argument("cls")
    p.add_argument("ids", type=int, nargs="+")
    sub.add_parser("tiles").add_argument("stem")
    p = sub.add_parser("tile")
    p.add_argument("stem")
    p.add_argument("r", type=int)
    p.add_argument("c", type=int)
    p = sub.add_parser("add")
    p.add_argument("stem")
    p.add_argument("cls")
    p.add_argument("x", type=float)
    p.add_argument("y", type=float)
    p = sub.add_parser("addbox")
    p.add_argument("stem")
    p.add_argument("cls")
    for k in "xywh":
        p.add_argument(k, type=float)
    p = sub.add_parser("delete")
    p.add_argument("stem")
    p.add_argument("ids", type=int, nargs="+")
    sub.add_parser("stats").add_argument("stem")
    a = ap.parse_args()
    if a.cmd == "legend":
        cmd_legend(a.family)
    elif a.cmd == "init":
        cmd_init(a.stem, a.thr, [x for x in a.skip.split(",") if x])
    elif a.cmd == "grid":
        cmd_grid(a.stem, a.cls, a.status, a.page)
    elif a.cmd == "set":
        cmd_set(a.stem, a.status, a.ids)
    elif a.cmd == "relabel":
        cmd_relabel(a.stem, a.cls, a.ids)
    elif a.cmd == "tiles":
        cmd_tiles(a.stem)
    elif a.cmd == "tile":
        cmd_tile(a.stem, a.r, a.c)
    elif a.cmd == "add":
        cmd_add(a.stem, a.cls, a.x, a.y)
    elif a.cmd == "addbox":
        cmd_addbox(a.stem, a.cls, a.x, a.y, a.w, a.h)
    elif a.cmd == "delete":
        cmd_delete(a.stem, a.ids)
    elif a.cmd == "stats":
        cmd_stats(a.stem)


if __name__ == "__main__":
    main()
