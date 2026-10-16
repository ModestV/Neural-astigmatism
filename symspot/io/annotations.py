"""Чтение эталонов и разметки: COCO из CVAT, простой JSON, GT синтетики. Задача T3."""

from __future__ import annotations

from pathlib import Path

from symspot.core.types import GroundTruth, QuerySpec


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
    """
    raise NotImplementedError


def load_ground_truth(
    path: str | Path,
    drawing_size: tuple[float, float] | None = None,
    image_name: str | None = None,
) -> list[GroundTruth]:
    """Разметка экземпляров: COCO ``object`` (``legend_id`` из атрибутов) или GT синтетики.

    В GT синтетики экземпляры-эталоны помечены ``is_query``. Масштабирование как в
    ``load_queries``.
    """
    raise NotImplementedError
