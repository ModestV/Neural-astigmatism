"""Чтение эталонов и разметки: COCO из CVAT, простой JSON, GT синтетики. Задача T3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from symspot.core.types import BBox, GroundTruth, QuerySpec

LEGEND_CATEGORY = "legend_item"
OBJECT_CATEGORY = "object"


# ---------------------------------------------------------------------------
# Общие помощники
# ---------------------------------------------------------------------------


def _read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _detect_format(data: Any) -> str:
    """Тип файла по содержимому: ``coco``, ``synth`` или ``simple``."""
    if isinstance(data, list):
        return "simple"
    if not isinstance(data, dict):
        raise ValueError("ожидался JSON-объект или список")
    if "annotations" in data and ("images" in data or "categories" in data):
        return "coco"
    if "instances" in data or "queries" in data:
        return "synth"
    if "legend_items" in data:
        return "simple"
    raise ValueError("неизвестный формат разметки (нет annotations, instances, legend_items)")


def _legend_id(raw: Any, ctx: str) -> str:
    """``legend_id`` как строка без пробелов; пустой -> ValueError."""
    if isinstance(raw, float) and raw.is_integer():
        raw = int(raw)
    text = "" if raw is None else str(raw).strip()
    if not text:
        raise ValueError(f"пустой legend_id: {ctx}")
    return text


def _scale_factors(
    drawing_size: tuple[float, float] | None, src_w: Any, src_h: Any
) -> tuple[float, float]:
    """Множители (sx, sy); (1, 1), если размеры неизвестны."""
    if drawing_size is None or not src_w or not src_h:
        return 1.0, 1.0
    return float(drawing_size[0]) / float(src_w), float(drawing_size[1]) / float(src_h)


def _bbox_xyxy(box: Any, sx: float, sy: float, ctx: str) -> BBox:
    """Рамка (x0, y0, x1, y1) с упорядочением координат и масштабом."""
    if box is None or len(box) != 4:
        raise ValueError(f"некорректный bbox: {ctx}")
    xa, ya, xb, yb = (float(v) for v in box)
    return (min(xa, xb) * sx, min(ya, yb) * sy, max(xa, xb) * sx, max(ya, yb) * sy)


def _bbox_xywh(box: Any, sx: float, sy: float, ctx: str) -> BBox:
    """COCO [x, y, w, h] -> (x0, y0, x1, y1) с масштабом."""
    if box is None or len(box) != 4:
        raise ValueError(f"некорректный bbox: {ctx}")
    x, y, w, h = (float(v) for v in box)
    return _bbox_xyxy((x, y, x + w, y + h), sx, sy, ctx)


def _stem(name: str) -> str:
    """Базовое имя без каталогов и расширения (поддерживает ``/`` и ``\\``)."""
    return Path(name.replace("\\", "/")).stem


def _label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


Item = tuple[str, BBox, str | None]  # (legend_id, bbox, label)


# ---------------------------------------------------------------------------
# Форматы
# ---------------------------------------------------------------------------


def _coco_select_image(data: dict, image_name: str | None) -> dict | None:
    images = data.get("images") or []
    if not images:
        return None
    if len(images) == 1:
        return images[0]
    names = [str(im.get("file_name", "")) for im in images]
    if image_name:
        key = _stem(image_name)
        found = [im for im, n in zip(images, names, strict=True) if _stem(n) == key]
        if len(found) == 1:
            return found[0]
        reason = "несколько изображений совпали с" if found else "не найдено изображение"
    else:
        reason = "в COCO несколько изображений, image_name не задан:"
    raise ValueError(f"{reason} {image_name!r}; имена в файле: {names}")


def _coco_items(
    data: dict,
    category: str,
    drawing_size: tuple[float, float] | None,
    image_name: str | None,
) -> list[Item]:
    """Аннотации категории ``category`` выбранного изображения."""
    image = _coco_select_image(data, image_name)
    sx, sy = 1.0, 1.0
    if image is not None:
        sx, sy = _scale_factors(drawing_size, image.get("width"), image.get("height"))
    cat_ids = {
        c["id"]
        for c in data.get("categories", [])
        if str(c.get("name", "")).strip().lower() == category
    }
    out: list[Item] = []
    for ann in data.get("annotations", []):
        if ann.get("category_id") not in cat_ids:
            continue
        if image is not None and ann.get("image_id") != image.get("id"):
            continue
        ctx = f"аннотация id={ann.get('id', '?')}"
        attrs = ann.get("attributes") or {}
        out.append(
            (
                _legend_id(attrs.get("legend_id"), ctx),
                _bbox_xywh(ann.get("bbox"), sx, sy, ctx),
                _label(attrs.get("meaning")),
            )
        )
    return out


def _simple_items(data: Any, drawing_size: tuple[float, float] | None) -> list[Item]:
    if isinstance(data, dict):
        items = data.get("legend_items", [])
        sx, sy = _scale_factors(drawing_size, data.get("width"), data.get("height"))
    else:
        items = data
        sx, sy = 1.0, 1.0
    return _xyxy_items(items, sx, sy, "legend_items")


def _xyxy_items(items: list[dict], sx: float, sy: float, name: str) -> list[Item]:
    out: list[Item] = []
    for k, it in enumerate(items):
        ctx = f"{name}[{k}]"
        out.append(
            (
                _legend_id(it.get("legend_id"), ctx),
                _bbox_xyxy(it.get("bbox"), sx, sy, ctx),
                _label(it.get("label")),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Публичный интерфейс
# ---------------------------------------------------------------------------


def load_queries(
    path: str | Path,
    drawing_size: tuple[float, float] | None = None,
    image_name: str | None = None,
) -> list[QuerySpec]:
    """Эталоны из файла разметки.

    Форматы (определяются по содержимому):
    - COCO из CVAT: аннотации категории ``legend_item``; ``legend_id`` из
      ``attributes.legend_id``; ``label`` из ``attributes.meaning`` (если есть).
      bbox COCO = [x, y, w, h] -> (x0, y0, x1, y1).
    - простой JSON: список ``[{legend_id, bbox: [x0,y0,x1,y1], label?}]`` или
      ``{"legend_items": [...], "width"?, "height"?}``.
    - GT синтетики (``{"width", "height", "queries": [...], "instances": [...]}``).
    Масштабирование: если ``drawing_size=(W, H)`` задан и в файле известен размер исходного
    изображения (COCO ``images[].width/height``, или ``width/height`` в JSON), координаты
    умножаются на W / w_src и H / h_src. Если в COCO несколько изображений, берется то, у
    которого базовое имя файла совпадает с базовым именем ``image_name`` (без каталогов и без
    расширения), иначе единственное; при неоднозначности — ValueError.
    Несколько эталонов с одним ``legend_id`` допустимы и сохраняются все.
    """
    data = _read_json(path)
    fmt = _detect_format(data)
    if fmt == "coco":
        items = _coco_items(data, LEGEND_CATEGORY, drawing_size, image_name)
    elif fmt == "synth":
        sx, sy = _scale_factors(drawing_size, data.get("width"), data.get("height"))
        items = _xyxy_items(data.get("queries", []), sx, sy, "queries")
    else:
        items = _simple_items(data, drawing_size)
    return [QuerySpec(legend_id=lid, bbox=bb, label=lb) for lid, bb, lb in items]


def load_ground_truth(
    path: str | Path,
    drawing_size: tuple[float, float] | None = None,
    image_name: str | None = None,
) -> list[GroundTruth]:
    """Разметка экземпляров: COCO ``object`` (``legend_id`` из атрибутов) или GT синтетики.

    В GT синтетики экземпляры-эталоны помечены ``is_query``. Масштабирование как в
    ``load_queries``. ``parent_ids`` не заполняется. Файл простого JSON с одними эталонами
    не содержит разметки экземпляров -> ValueError.
    """
    data = _read_json(path)
    fmt = _detect_format(data)
    if fmt == "coco":
        return [
            GroundTruth(legend_id=lid, bbox=bb)
            for lid, bb, _ in _coco_items(data, OBJECT_CATEGORY, drawing_size, image_name)
        ]
    if fmt == "synth":
        sx, sy = _scale_factors(drawing_size, data.get("width"), data.get("height"))
        raw = data.get("instances", [])
        items = _xyxy_items(raw, sx, sy, "instances")
        return [
            GroundTruth(legend_id=lid, bbox=bb, is_query=bool(r.get("is_query", False)))
            for (lid, bb, _), r in zip(items, raw, strict=True)
        ]
    raise ValueError("файл простого JSON содержит только эталоны, разметки экземпляров нет")


def load_similar_groups(path: str | Path) -> list[set[str]]:
    """Группы похожих символов из GT синтетики (``similar_groups``); для остальных форматов []."""
    data = _read_json(path)
    if _detect_format(data) != "synth":
        return []
    return [{str(x).strip() for x in g} for g in data.get("similar_groups", []) if g]
