"""Нарезка листа на плитки (куски фиксированного размера) вместе с рамками.

Что происходит, простыми словами: лист режется на прямоугольные куски, а рамки пересчитываются в
координаты каждого куска. Рамка, разрезанная краем плитки, обрезается по краю или отбрасывается.

Режимы (`legend_mode`):
- `keep`: остаются все плитки, где есть хотя бы одна рамка. У каждого `object` ставится атрибут
  `legend_in_tile`: есть ли в этой плитке образец (`legend_item`) с тем же `legend_id`.
- `require`: остаются только плитки с хотя бы одним `object`, у каждого из которых образец в плитке есть.
  Образец вместе с экземплярами нужен, чтобы учиться сопоставлению. Для листов, где легенда лежит на
  отдельном листе (группа SVG), `require` не даст ни одной плитки.
- `with-legend`: как `keep`, но для объектов, чей образец остался вне плитки, образец вырезается из
  целого листа и приклеивается полосой под плиткой (плитка становится выше). Приклеенному образцу
  ставится `pasted = True`, а `legend_in_tile` у объектов становится True. Образец, который не
  помещается по ширине плитки, не приклеивается (у объекта остаётся `legend_in_tile = False`).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from coco_io import LEGEND_ITEM, OBJECT, Box, Sheet

LEGEND_MODES = ("keep", "require", "with-legend")
STRIP_MARGIN = 8  # отступ между приклеенными образцами и от краёв полосы, px


def tile_positions(length: int, tile: int, step: int) -> list[int]:
    """Левые (или верхние) края плиток вдоль одной оси; последняя прижата к краю листа."""
    if tile >= length:
        return [0]
    positions = list(range(0, length - tile + 1, step))
    if positions[-1] + tile < length:
        positions.append(length - tile)
    return positions


def _cut_box(box: Box, x0: int, y0: int, w: int, h: int, min_visible: float) -> Box | None:
    """Рамка в координатах плитки или None, если видимая доля площади меньше `min_visible`."""
    x, y, bw, bh = box.bbox
    left, top = max(x, x0), max(y, y0)
    right, bottom = min(x + bw, x0 + w), min(y + bh, y0 + h)
    if right <= left or bottom <= top or bw <= 0 or bh <= 0:
        return None
    visible = (right - left) * (bottom - top) / (bw * bh)
    if visible < min_visible:
        return None
    attributes = dict(box.attributes)
    if visible < 1.0:
        attributes["truncated"] = True  # рамка обрезана краем плитки
    return Box(box.category, (left - x0, top - y0, right - left, bottom - top), attributes)


def plan_legend_strip(
    tile: Sheet, full: Sheet
) -> tuple[list[tuple[Box, int, int, int, int]], int]:
    """Раскладывает недостающие образцы полками слева направо под плиткой (картинку не трогает).

    Возвращает список `(исходная рамка образца, x, y в полосе, ширина, высота)` и высоту полосы.
    Пустой список — приклеивать нечего (полосы нет, высота 0).
    """
    present = {b.legend_id for b in tile.boxes if b.category == LEGEND_ITEM}
    wanted: list[str] = []
    for box in tile.boxes:
        lid = box.legend_id
        if box.category == OBJECT and lid is not None and lid not in present and lid not in wanted:
            wanted.append(lid)
    sources = {}
    for box in full.boxes:
        if box.category == LEGEND_ITEM and box.legend_id in wanted:
            sources.setdefault(box.legend_id, box)
    m = STRIP_MARGIN
    placed = []
    x, y, row_h = m, m, 0
    for lid in wanted:
        src = sources.get(lid)
        if src is None:
            continue
        w, h = (int(round(v)) for v in src.bbox[2:])
        if w < 1 or h < 1 or w + 2 * m > tile.width:
            continue
        if x + w + m > tile.width:  # не влезло в строку: новая полка
            x, y, row_h = m, y + row_h + m, 0
        placed.append((src, x, y, w, h))
        x += w + m
        row_h = max(row_h, h)
    if not placed:
        return [], 0
    return placed, y + row_h + m


def with_legend_strip(
    tile: Sheet, placed: list[tuple[Box, int, int, int, int]], strip_h: int
) -> Sheet:
    """Лист плитки с полосой образцов: выше на `strip_h`, образцы добавлены с `pasted = True`."""
    boxes = list(tile.boxes)
    for src, px, py, w, h in placed:
        boxes.append(Box(LEGEND_ITEM, (px, tile.height + py, w, h), {**src.attributes, "pasted": True}))
    ids = {b.legend_id for b in boxes if b.category == LEGEND_ITEM}
    for b in boxes:
        if b.category == OBJECT:
            b.attributes["legend_in_tile"] = b.legend_id in ids
    history = [*tile.history, f"with_legend:{','.join(str(p[0].legend_id) for p in placed)}"]
    return Sheet(tile.file_name, tile.width, tile.height + strip_h, boxes, tile.source, history)


def _attach_legends(
    tile_image: Image.Image, tile: Sheet, full_image: Image.Image, full: Sheet
) -> tuple[Image.Image, Sheet]:
    """Приклеивает под плитку недостающие образцы, нужные её объектам (полка за полкой слева направо)."""
    placed, strip_h = plan_legend_strip(tile, full)
    if not placed:
        return tile_image, tile
    white = 255 if tile_image.mode == "L" else (255, 255, 255)
    canvas = Image.new(tile_image.mode, (tile.width, tile.height + strip_h), white)
    canvas.paste(tile_image, (0, 0))
    for src, px, py, w, h in placed:
        sx, sy = int(round(src.bbox[0])), int(round(src.bbox[1]))
        canvas.paste(full_image.crop((sx, sy, sx + w, sy + h)), (px, tile.height + py))
    return canvas, with_legend_strip(tile, placed, strip_h)


def plan_tiles(
    sheet: Sheet,
    tile_size: tuple[int, int],
    overlap: int = 0,
    min_visible: float = 0.5,
    legend_min_visible: float = 1.0,
    legend_mode: str = "keep",
    keep_empty: bool = False,
) -> list[tuple[int, int, Sheet]]:
    """Выбирает плитки и пересчитывает рамки; картинку не трогает.

    Возвращает `(x0, y0, лист плитки)`; размер плитки в листе — без полосы образцов
    (режим `with-legend` её добавляет тот, кто режет картинку: `tile_sheet` или `pdf_ops`).
    Параметры те же, что у `tile_sheet`.
    """
    tile_w, tile_h = tile_size
    if tile_w <= 0 or tile_h <= 0:
        raise ValueError("Размер плитки должен быть положительным")
    if not 0 <= overlap < min(tile_w, tile_h):
        raise ValueError("Перекрытие должно быть от 0 и меньше размера плитки")
    if legend_mode not in LEGEND_MODES:
        raise ValueError(f"legend_mode должен быть одним из {LEGEND_MODES}, получено {legend_mode!r}")
    tile_w, tile_h = min(tile_w, sheet.width), min(tile_h, sheet.height)
    path = Path(sheet.file_name)
    result = []
    for y0 in tile_positions(sheet.height, tile_h, tile_h - overlap):
        for x0 in tile_positions(sheet.width, tile_w, tile_w - overlap):
            boxes = []
            for box in sheet.boxes:
                needed = legend_min_visible if box.category == LEGEND_ITEM else min_visible
                cut = _cut_box(box, x0, y0, tile_w, tile_h, needed)
                if cut is not None:
                    boxes.append(cut)
            legend_ids = {b.legend_id for b in boxes if b.category == LEGEND_ITEM}
            objects = [b for b in boxes if b.category == OBJECT]
            for obj in objects:
                obj.attributes["legend_in_tile"] = obj.legend_id in legend_ids
            if not boxes and not keep_empty:
                continue
            if legend_mode == "require" and (
                not objects or not all(o.attributes["legend_in_tile"] for o in objects)
            ):
                continue
            name = f"{path.stem}__tile_x{x0}_y{y0}{path.suffix}"
            history = [*sheet.history, f"tile:{x0},{y0},{tile_w},{tile_h}"]
            result.append((x0, y0, Sheet(name, tile_w, tile_h, boxes, source=sheet.source,
                                         history=history)))
    return result


def tile_sheet(
    image: Image.Image,
    sheet: Sheet,
    tile_size: tuple[int, int],
    overlap: int = 0,
    min_visible: float = 0.5,
    legend_min_visible: float = 1.0,
    legend_mode: str = "keep",
    keep_empty: bool = False,
) -> list[tuple[Image.Image, Sheet]]:
    """Режет лист на плитки `tile_size = (ширина, высота)` с перекрытием `overlap` пикселей.

    min_visible: какая доля площади `object` должна остаться в плитке, чтобы рамка сохранилась.
    legend_min_visible: то же для образцов; по умолчанию образец берётся только целиком, потому что
        обрезанный образец плохой эталон. Если образец отброшен, объектам ставится `legend_in_tile = False`.
    keep_empty: оставлять ли плитки совсем без рамок (по умолчанию они пропускаются).
    """
    result = []
    for x0, y0, tile in plan_tiles(sheet, tile_size, overlap, min_visible, legend_min_visible,
                                   legend_mode, keep_empty):
        tile_image = image.crop((x0, y0, x0 + tile.width, y0 + tile.height))
        if legend_mode == "with-legend":
            tile_image, tile = _attach_legends(tile_image, tile, image, sheet)
        result.append((tile_image, tile))
    return result
