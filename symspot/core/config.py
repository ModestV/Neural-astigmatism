"""Все пороги SymSpot. Значения по умолчанию = ``configs/default.yaml`` (PLAN.md, раздел 5).

Единицы:
- ``sample_step``, ``cover_tol``, ``exact_tol`` — доли диагонали эталона;
- ``flatten_tol``, ``arc_fit_tol``, ``merge_gap``, ``snap_tol`` — доли опорного размера
  символа (медиана диагоналей рамок эталонов), пайплайн переводит их в абсолютные;
- углы в градусах, если в имени есть ``deg``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # --- сэмплинг и нормализация ---
    sample_step: float = 0.02
    flatten_tol: float = 0.002
    arc_fit_tol: float = 0.01
    merge_gap: float = 0.005
    merge_angle_deg: float = 1.0
    snap_tol: float = 0.002
    min_piece_len: float = 1e-4  # доля опорного размера: короче — выбрасываем

    # --- эталоны ---
    query_parent_inside: float = 0.8
    query_bbox_pad: float = 0.02  # расширение рамки эталона (доля диагонали) при выборе кусков

    # --- точные копии ---
    exact_tol: float = 0.005

    # --- пары и голосование ---
    pair_radius_factor: float = 1.5
    scale_min: float = 0.7
    scale_max: float = 1.4
    allow_mirror: bool = True
    angle_bin_deg: float = 5.0
    scale_bin_log: float = 0.05
    shift_bin: float = 0.25  # доля диагонали эталона
    ratio_bin_log: float = 0.1  # квантование log-отношений размеров в ключе пары
    key_angle_bin_deg: float = 10.0  # квантование углов в ключе пары
    min_distinct_pairs: int = 3
    max_hypotheses_per_query: int = 2000
    max_pairs_per_piece: int = 64  # ограничение пар на кусок чертежа (плотные места)
    density_norm: bool = True

    # --- проверка ---
    icp_iters: int = 15
    icp_angle_deg: float = 20.0
    cover_tol: float = 0.03
    min_prim_coverage: float = 0.6
    accept_score: float = 0.8
    extra_ink_lambda: float = 0.05
    text_match: bool = True

    # --- отличающие части и соревнование ---
    discriminative_weight: float = 3.0
    similar_cover: float = 0.85  # эталоны похожи, если один покрывает другой на эту долю
    similar_size_ratio: float = 1.5
    compete_iou: float = 0.5
    compete_margin: float = 0.03

    # --- назначение кусков и разрешение ---
    assign_inside: float = 0.7
    resolve_taken_frac: float = 0.5
    dedup_iou: float = 0.5

    # --- расширение запроса ---
    expand: bool = True
    expand_score: float = 0.9
    expand_residual: float = 0.01  # доля диагонали

    # --- растр ---
    raster_angle_step: float = 10.0
    raster_scale_step: float = 0.05
    raster_tile: int = 2048
    raster_topk: int = 200
    sauvola_window: int = 25
    sauvola_k: float = 0.2
    min_blob_px: int = 8
    raster_chamfer_max: float = 0.08  # доля диагонали: порог среднего chamfer для гипотезы
    deskew: bool = False

    # --- производительность ---
    workers: int | str = "auto"

    # --- служебное ---
    seed: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)

    def n_workers(self) -> int:
        if self.workers == "auto":
            return max(1, (os.cpu_count() or 2) - 1)
        return max(1, int(self.workers))


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Читает YAML (если задан) и применяет переопределения из CLI."""
    data: dict[str, Any] = {}
    if path is not None:
        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"config {path}: ожидался словарь")
        data.update(loaded)
    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})
    return Config(**data)
