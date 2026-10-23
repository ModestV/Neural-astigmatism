"""Проверяемый формат входных данных и результатов исследования."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Box(StrictModel):
    """Рамка в координатах отображаемой страницы PDF, начало слева сверху."""

    x: float = Field(ge=0)
    y: float = Field(ge=0)
    w: float = Field(gt=0)
    h: float = Field(gt=0)

    def as_list(self) -> list[float]:
        return [self.x, self.y, self.w, self.h]


class PageSpec(StrictModel):
    file: str
    page: int = Field(default=1, ge=1)


class TemplateSpec(PageSpec):
    id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    bbox: Box


class Truth(StrictModel):
    symbol: str
    bbox: Box
    legend: bool = False


class TargetSpec(PageSpec):
    # None означает отсутствие разметки; [] — размеченная страница без объектов.
    ground_truth: list[Truth] | None = None
    # Области, которые исключаем из поиска и оценки (например, источник образца).
    ignore_regions: list[Box] = Field(default_factory=list)


class Experiment(StrictModel):
    schema_version: Literal[1] = 1
    description: str = ""
    templates: list[TemplateSpec] = Field(min_length=1)
    targets: list[TargetSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_items(self) -> Experiment:
        ids = [t.id for t in self.templates]
        if len(ids) != len(set(ids)):
            raise ValueError("Идентификаторы образцов должны быть уникальны")
        keys = [(t.file, t.page) for t in self.targets]
        if len(keys) != len(set(keys)):
            raise ValueError("Страница не должна повторяться в targets")
        return self


class Settings(StrictModel):
    dpi: int = Field(default=150, ge=36, le=600)
    angles: list[float] = Field(default_factory=lambda: [0, 90, 180, 270], min_length=1)
    scales: list[float] = Field(default_factory=lambda: [1.0], min_length=1)
    template_threshold: float = Field(default=0.85, gt=0, le=1)
    ratio: float = Field(default=0.75, gt=0, lt=1)
    min_inliers: int = Field(default=4, ge=3)
    min_inlier_ratio: float = Field(default=0.5, gt=0, le=1)
    ransac_px: float = Field(default=3, gt=0)
    angle_tolerance: float = Field(default=10, ge=0, le=180)
    scale_tolerance: float = Field(default=0.15, ge=0, lt=1)
    features: int = Field(default=5000, ge=10)
    nms_iou: float = Field(default=0.3, gt=0, le=1)
    evaluation_iou: float = Field(default=0.5, gt=0, le=1)
    max_detections: int = Field(default=200, ge=1)
    max_pixels: int = Field(default=40_000_000, ge=100)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    include_legend: bool = False
    # Первичная версия не проверяет зеркальные экземпляры.

    @model_validator(mode="after")
    def positive_scales(self) -> Settings:
        if any(not math.isfinite(s) or s <= 0 for s in self.scales):
            raise ValueError("Масштабы должны быть конечными положительными числами")
        if any(not math.isfinite(a) for a in self.angles):
            raise ValueError("Углы должны быть конечными числами")
        return self


class Detection(StrictModel):
    symbol: str
    template_id: str
    bbox: Box
    score: float
    angle: float
    scale: float
    inliers: int | None = None
    matches: int | None = None
