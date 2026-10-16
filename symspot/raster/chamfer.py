"""Chamfer-поиск эталона по растру. Фаза 5."""

from __future__ import annotations

from symspot.core.config import Config
from symspot.core.types import Hypothesis, Query
from symspot.raster.preprocess import RasterPrep


def chamfer_search(query: Query, prep: RasterPrep, cfg: Config) -> list[Hypothesis]:
    """Гипотезы T (нормализованный эталон -> пиксели) перебором углов (шаг
    ``raster_angle_step``, при ``allow_mirror`` и с отражением) и масштабов
    (``scale_min..scale_max``, шаг ``raster_scale_step``) относительно ``query.diag``.

    Для каждой пары (угол, масштаб) сэмплы эталона ``query.samples`` растеризуются в бинарную
    маску-шаблон; средний chamfer = свертка ``prep.dist`` с маской (``cv2.filter2D``) / число
    пикселей шаблона. Тайлы ``raster_tile`` с перекрытием >= 2 размера шаблона. Локальные
    минимумы ниже ``raster_chamfer_max`` * диагональ экземпляра -> гипотезы
    (``votes`` = -chamfer, ``source="chamfer"``), NMS по центру, топ ``raster_topk``.
    """
    raise NotImplementedError
