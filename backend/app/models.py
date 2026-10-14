"""Модели данных. В этом же виде они лежат в JSON-файлах каталога data/."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator

DrawingFormat = Literal["pdf", "svg", "dxf", "dwg"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class BBox(BaseModel):
    """Прямоугольник в координатах страницы: (x, y) — левый верхний угол, ось Y направлена вниз."""

    x: float
    y: float
    w: float = Field(gt=0)
    h: float = Field(gt=0)


class PageInfo(BaseModel):
    index: int = Field(ge=1, description="Номер страницы, начиная с 1")
    width: float
    height: float


class Drawing(BaseModel):
    id: str
    name: str = Field(description="Имя исходного файла")
    format: DrawingFormat
    size_bytes: int
    created_at: datetime = Field(default_factory=utcnow)
    status: Literal["ready", "error"] = "ready"
    error: str | None = None
    units: str = Field(
        "pt", description="Единицы координат страницы: pt — пункты PDF, svg — единицы viewBox"
    )
    pages: list[PageInfo] = Field(default_factory=list)

    def page(self, index: int) -> PageInfo | None:
        return next((p for p in self.pages if p.index == index), None)


class Symbol(BaseModel):
    """Условное обозначение из библиотеки — образец для поиска."""

    id: str
    name: str
    category: str | None = None
    drawing_id: str = Field(description="Чертёж, на котором выделен образец")
    page: int
    bbox: BBox
    created_at: datetime = Field(default_factory=utcnow)
    has_preview: bool = False


class SearchParams(BaseModel):
    """Параметры поиска. Заглушка их игнорирует, они заготовлены для настоящего алгоритма."""

    threshold: float = Field(0.8, ge=0, le=1, description="Порог сходства")
    rotations: bool = Field(True, description="Искать повёрнутые экземпляры")
    mirror: bool = Field(False, description="Искать зеркальные экземпляры")
    scale: bool = Field(False, description="Искать экземпляры другого масштаба")


class RegionQuery(BaseModel):
    """Образец, выделенный на чертеже без сохранения в библиотеку."""

    drawing_id: str
    page: int = Field(ge=1)
    bbox: BBox


class SearchRequest(BaseModel):
    drawing_id: str = Field(description="Чертёж, на котором ищем")
    pages: list[int] | None = Field(None, description="Номера страниц; null — все страницы")
    query: RegionQuery | None = None
    symbol_ids: list[str] = Field(default_factory=list, description="Образцы из библиотеки")
    params: SearchParams = Field(default_factory=SearchParams)

    @model_validator(mode="after")
    def _require_template(self) -> SearchRequest:
        if self.query is None and not self.symbol_ids:
            raise ValueError("Нужен образец: выделенный фрагмент (query) или symbol_ids")
        return self


class Detection(BaseModel):
    page: int
    bbox: BBox
    score: float = Field(ge=0, le=1)
    symbol_id: str | None = Field(None, description="Найденный образец; null — выделенный фрагмент")
    angle: float = Field(0.0, description="Поворот относительно образца, градусы")
    mirrored: bool = False


class SearchRun(BaseModel):
    """Один запуск поиска — запрос и его результаты."""

    id: str
    created_at: datetime = Field(default_factory=utcnow)
    request: SearchRequest
    detector: str
    status: Literal["done", "failed"]
    message: str | None = None
    duration_ms: float
    detections: list[Detection] = Field(default_factory=list)
