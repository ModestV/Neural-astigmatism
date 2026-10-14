from __future__ import annotations

import base64
import binascii

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..deps import StorageDep, drawing_names
from ..models import BBox, PageInfo, Symbol
from ..schemas import SymbolCreate, SymbolOut, SymbolUpdate, symbol_out
from ..storage import Storage, new_id

router = APIRouter(prefix="/api/symbols", tags=["symbols"])

PNG_DATA_URL_PREFIX = "data:image/png;base64,"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_PREVIEW_BYTES = 5 * 1024 * 1024


@router.get("", response_model=list[SymbolOut])
def list_symbols(
    storage: StorageDep, drawing_id: str | None = None, category: str | None = None
) -> list[SymbolOut]:
    """Библиотека условных обозначений — «заданный набор» образцов для поиска."""
    symbols = storage.list_symbols()
    if drawing_id is not None:
        symbols = [s for s in symbols if s.drawing_id == drawing_id]
    if category is not None:
        symbols = [s for s in symbols if s.category == category]
    names = drawing_names(storage)
    return [symbol_out(s, names) for s in symbols]


@router.post("", response_model=SymbolOut, status_code=201)
def create_symbol(body: SymbolCreate, storage: StorageDep) -> SymbolOut:
    """Сохранить выделенный на чертеже фрагмент в библиотеку."""
    drawing = storage.get_drawing(body.drawing_id)
    if drawing is None:
        raise HTTPException(404, "Чертёж не найден")
    page = drawing.page(body.page)
    if page is None:
        raise HTTPException(422, f"На чертеже нет страницы {body.page}")
    name = body.name.strip()
    if not name:
        raise HTTPException(422, "Название не может быть пустым")

    preview = _decode_png(body.preview_png) if body.preview_png else None
    symbol = Symbol(
        id=new_id(),
        name=name,
        category=_clean_category(body.category),
        drawing_id=drawing.id,
        page=page.index,
        bbox=_clip_to_page(body.bbox, page),
        has_preview=preview is not None,
    )
    storage.save_symbol(symbol, preview)
    return symbol_out(symbol, {drawing.id: drawing.name})


@router.get("/{symbol_id}", response_model=SymbolOut)
def get_symbol(symbol_id: str, storage: StorageDep) -> SymbolOut:
    return symbol_out(_get_or_404(storage, symbol_id), drawing_names(storage))


@router.patch("/{symbol_id}", response_model=SymbolOut)
def update_symbol(symbol_id: str, body: SymbolUpdate, storage: StorageDep) -> SymbolOut:
    symbol = _get_or_404(storage, symbol_id)
    changes = body.model_dump(exclude_unset=True)
    if changes.get("name") is not None:
        name = changes["name"].strip()
        if not name:
            raise HTTPException(422, "Название не может быть пустым")
        symbol.name = name
    if "category" in changes:
        symbol.category = _clean_category(changes["category"])
    storage.save_symbol(symbol)
    return symbol_out(symbol, drawing_names(storage))


@router.delete("/{symbol_id}", status_code=204)
def delete_symbol(symbol_id: str, storage: StorageDep) -> None:
    if not storage.delete_symbol(symbol_id):
        raise HTTPException(404, "Обозначение не найдено")


@router.get("/{symbol_id}/preview")
def symbol_preview(symbol_id: str, storage: StorageDep) -> FileResponse:
    symbol = _get_or_404(storage, symbol_id)
    path = storage.symbol_preview_path(symbol.id)
    if not path.is_file():
        raise HTTPException(404, "У обозначения нет превью")
    return FileResponse(path, media_type="image/png")


def _get_or_404(storage: Storage, symbol_id: str) -> Symbol:
    symbol = storage.get_symbol(symbol_id)
    if symbol is None:
        raise HTTPException(404, "Обозначение не найдено")
    return symbol


def _clean_category(value: str | None) -> str | None:
    return (value or "").strip() or None


def _clip_to_page(bbox: BBox, page: PageInfo) -> BBox:
    x0, y0 = max(bbox.x, 0.0), max(bbox.y, 0.0)
    x1, y1 = min(bbox.x + bbox.w, page.width), min(bbox.y + bbox.h, page.height)
    if x1 <= x0 or y1 <= y0:
        raise HTTPException(422, "Выделенная область лежит за пределами страницы")
    return BBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0)


def _decode_png(data_url: str) -> bytes:
    if not data_url.startswith(PNG_DATA_URL_PREFIX):
        raise HTTPException(422, "preview_png должен быть data URL с картинкой PNG")
    try:
        png = base64.b64decode(data_url[len(PNG_DATA_URL_PREFIX) :], validate=True)
    except binascii.Error:
        raise HTTPException(422, "preview_png: некорректный base64") from None
    if not png.startswith(PNG_SIGNATURE):
        raise HTTPException(422, "preview_png: это не PNG")
    if len(png) > MAX_PREVIEW_BYTES:
        raise HTTPException(413, "preview_png слишком большой")
    return png
