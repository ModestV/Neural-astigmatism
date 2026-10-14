"""Схемы HTTP API: что приходит от клиента и что уходит ему (с готовыми URL файлов)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .models import BBox, Drawing, PageInfo, SearchRun, Symbol
from .storage import Storage


class PageOut(PageInfo):
    image_url: str
    thumb_url: str


class DrawingOut(Drawing):
    pages: list[PageOut]
    original_url: str


class SymbolOut(Symbol):
    preview_url: str | None
    drawing_name: str | None = Field(description="null — исходный чертёж удалён")


class SymbolCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    category: str | None = Field(None, max_length=100)
    drawing_id: str
    page: int = Field(ge=1)
    bbox: BBox
    preview_png: str | None = Field(None, description="Превью фрагмента: data:image/png;base64,…")


class SymbolUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=200)
    category: str | None = Field(None, max_length=100)


class RunSummary(BaseModel):
    id: str
    created_at: datetime
    drawing_id: str
    drawing_name: str | None
    detector: str
    status: Literal["done", "failed"]
    message: str | None
    duration_ms: float
    detections_count: int
    has_region_query: bool
    symbols_count: int


class DetectorInfo(BaseModel):
    name: str
    description: str


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    detector: DetectorInfo
    dwg_converter: str | None = Field(description="Конвертер DWG; null — DWG не обрабатывается")


class Stats(BaseModel):
    drawings: int
    symbols: int
    runs: int


def drawing_out(drawing: Drawing, storage: Storage) -> DrawingOut:
    base = f"/api/drawings/{drawing.id}"
    pages = []
    for page in drawing.pages:
        image_url = f"{base}/pages/{page.index}/image"
        has_thumb = storage.page_thumb_path(drawing.id, page.index).is_file()
        thumb_url = f"{base}/pages/{page.index}/thumb" if has_thumb else image_url
        pages.append(PageOut(**page.model_dump(), image_url=image_url, thumb_url=thumb_url))
    return DrawingOut(
        **drawing.model_dump(exclude={"pages"}), pages=pages, original_url=f"{base}/original"
    )


def symbol_out(symbol: Symbol, drawing_names: dict[str, str]) -> SymbolOut:
    return SymbolOut(
        **symbol.model_dump(),
        preview_url=f"/api/symbols/{symbol.id}/preview" if symbol.has_preview else None,
        drawing_name=drawing_names.get(symbol.drawing_id),
    )


def run_summary(run: SearchRun, drawing_names: dict[str, str]) -> RunSummary:
    return RunSummary(
        id=run.id,
        created_at=run.created_at,
        drawing_id=run.request.drawing_id,
        drawing_name=drawing_names.get(run.request.drawing_id),
        detector=run.detector,
        status=run.status,
        message=run.message,
        duration_ms=run.duration_ms,
        detections_count=len(run.detections),
        has_region_query=run.request.query is not None,
        symbols_count=len(run.request.symbol_ids),
    )
