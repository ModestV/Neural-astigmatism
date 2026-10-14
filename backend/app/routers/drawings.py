from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..converters import ConversionError, detect_format, prepare
from ..deps import SettingsDep, StorageDep
from ..models import Drawing, PageInfo
from ..schemas import DrawingOut, drawing_out
from ..storage import Storage, new_id

router = APIRouter(prefix="/api/drawings", tags=["drawings"])

# Загруженный SVG может содержать скрипты. Внутри <img> они не выполняются,
# а CSP защищает и тогда, когда файл открывают напрямую по ссылке.
SVG_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; style-src 'unsafe-inline'; img-src data:; font-src data:"
    ),
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "no-cache",
}


@router.get("", response_model=list[DrawingOut])
def list_drawings(storage: StorageDep) -> list[DrawingOut]:
    return [drawing_out(d, storage) for d in storage.list_drawings()]


@router.post("", response_model=DrawingOut, status_code=201)
def upload_drawing(file: UploadFile, storage: StorageDep, settings: SettingsDep) -> DrawingOut:
    """Загрузить чертёж (PDF, SVG, DXF или DWG). Страницы для просмотра готовятся сразу.

    Если файл не удалось обработать, чертёж всё равно сохраняется — со статусом error
    и текстом ошибки, чтобы его можно было обработать повторно.
    """
    filename = Path(file.filename or "").name
    fmt = detect_format(filename)
    if fmt is None:
        raise HTTPException(415, "Поддерживаются файлы PDF, SVG, DXF и DWG")

    drawing_id = new_id()
    original = storage.drawing_dir(drawing_id) / f"original.{fmt}"
    original.parent.mkdir(parents=True)
    try:
        size = _save_upload(file, original, settings.max_upload_mb)
    except HTTPException:
        storage.delete_drawing(drawing_id)
        raise

    drawing = Drawing(id=drawing_id, name=filename, format=fmt, size_bytes=size)
    _process(storage, drawing)
    storage.save_drawing(drawing)
    return drawing_out(drawing, storage)


@router.get("/{drawing_id}", response_model=DrawingOut)
def get_drawing(drawing_id: str, storage: StorageDep) -> DrawingOut:
    return drawing_out(_get_or_404(storage, drawing_id), storage)


@router.delete("/{drawing_id}", status_code=204)
def delete_drawing(drawing_id: str, storage: StorageDep) -> None:
    """Удалить чертёж. Выделенные на нём обозначения остаются в библиотеке, но без исходника."""
    if not storage.delete_drawing(drawing_id):
        raise HTTPException(404, "Чертёж не найден")


@router.post("/{drawing_id}/reprocess", response_model=DrawingOut)
def reprocess_drawing(drawing_id: str, storage: StorageDep) -> DrawingOut:
    """Подготовить страницы заново — например, после установки конвертера DWG."""
    drawing = _get_or_404(storage, drawing_id)
    _process(storage, drawing)
    storage.save_drawing(drawing)
    return drawing_out(drawing, storage)


@router.get("/{drawing_id}/original")
def download_original(drawing_id: str, storage: StorageDep) -> FileResponse:
    drawing = _get_or_404(storage, drawing_id)
    return FileResponse(storage.original_path(drawing), filename=drawing.name)


@router.get("/{drawing_id}/pages/{page}/image")
def page_image(drawing_id: str, page: int, storage: StorageDep) -> FileResponse:
    """Страница в SVG — то, что показывает просмотрщик."""
    path = _page_file(storage, drawing_id, page, storage.page_svg_path)
    return FileResponse(path, media_type="image/svg+xml", headers=SVG_HEADERS)


@router.get("/{drawing_id}/pages/{page}/thumb")
def page_thumb(drawing_id: str, page: int, storage: StorageDep) -> FileResponse:
    path = _page_file(storage, drawing_id, page, storage.page_thumb_path)
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-cache"})


def _get_or_404(storage: Storage, drawing_id: str) -> Drawing:
    drawing = storage.get_drawing(drawing_id)
    if drawing is None:
        raise HTTPException(404, "Чертёж не найден")
    return drawing


def _page_file(
    storage: Storage, drawing_id: str, page: int, path_of: Callable[[str, int], Path]
) -> Path:
    drawing = _get_or_404(storage, drawing_id)
    path = path_of(drawing.id, page)
    if drawing.page(page) is None or not path.is_file():
        raise HTTPException(404, "Страница не найдена")
    return path


def _save_upload(file: UploadFile, target: Path, max_mb: int) -> int:
    size = 0
    with target.open("wb") as out:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > max_mb * 1024 * 1024:
                raise HTTPException(413, f"Файл больше {max_mb} МБ")
            out.write(chunk)
    if size == 0:
        raise HTTPException(400, "Файл пустой")
    return size


def _process(storage: Storage, drawing: Drawing) -> None:
    """Готовит страницы для просмотра и записывает результат в drawing."""
    try:
        prepared = prepare(
            storage.original_path(drawing),
            drawing.format,
            storage.pages_dir(drawing.id),
            storage.converted_pdf_path(drawing.id),
        )
    except ConversionError as exc:
        drawing.status, drawing.error, drawing.pages = "error", str(exc), []
        return
    drawing.status, drawing.error, drawing.units = "ready", None, prepared.units
    drawing.pages = [
        PageInfo(index=i, width=p.width, height=p.height)
        for i, p in enumerate(prepared.pages, start=1)
    ]
