from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Query

from ..deps import DetectorDep, StorageDep, drawing_names
from ..detection import PageRef, Template
from ..models import Detection, Drawing, SearchRequest, SearchRun
from ..schemas import RunSummary, run_summary
from ..storage import Storage, new_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["search"])


@router.post("/search", response_model=SearchRun)
def run_search(req: SearchRequest, storage: StorageDep, detector: DetectorDep) -> SearchRun:
    """Найти на чертеже выделенный фрагмент и/или обозначения из библиотеки.

    Поиск выполняется синхронно. Когда настоящий алгоритм начнёт работать долго,
    запуск стоит перенести в фоновую задачу.
    """
    target = _ready_drawing(storage, req.drawing_id)
    page_numbers = req.pages or [p.index for p in target.pages]
    for number in page_numbers:
        if target.page(number) is None:
            raise HTTPException(422, f"На чертеже нет страницы {number}")
    templates, skipped = _collect_templates(storage, req)

    started = time.perf_counter()
    detections: list[Detection] = []
    try:
        for number in page_numbers:
            page = page_ref(storage, target, number)
            detections.extend(detector.detect(page, templates, req.params))
        status, message = "done", detector.description or None
    except Exception as exc:
        logger.exception("Алгоритм %s завершился с ошибкой", detector.name)
        status, message, detections = "failed", f"Ошибка алгоритма: {exc}", []
    if skipped:
        note = f"Пропущены образцы без исходного чертежа: {', '.join(skipped)}."
        message = f"{message} {note}" if message else note

    run = SearchRun(
        id=new_id(),
        request=req,
        detector=detector.name,
        status=status,
        message=message,
        duration_ms=(time.perf_counter() - started) * 1000,
        detections=detections,
    )
    storage.save_run(run)
    return run


@router.get("/runs", response_model=list[RunSummary])
def list_runs(
    storage: StorageDep,
    drawing_id: str | None = None,
    limit: int = Query(50, ge=1, le=500),
) -> list[RunSummary]:
    """История запусков поиска, новые сверху."""
    runs = storage.list_runs()
    if drawing_id is not None:
        runs = [r for r in runs if r.request.drawing_id == drawing_id]
    names = drawing_names(storage)
    return [run_summary(r, names) for r in runs[:limit]]


@router.get("/runs/{run_id}", response_model=SearchRun)
def get_run(run_id: str, storage: StorageDep) -> SearchRun:
    run = storage.get_run(run_id)
    if run is None:
        raise HTTPException(404, "Запуск не найден")
    return run


def page_ref(storage: Storage, drawing: Drawing, index: int) -> PageRef:
    page = drawing.page(index)
    assert page is not None
    if drawing.format == "pdf":
        pdf_path = storage.original_path(drawing)
    elif drawing.format in ("dxf", "dwg"):
        pdf_path = storage.converted_pdf_path(drawing.id)
    else:
        pdf_path = None
    return PageRef(
        drawing=drawing,
        index=index,
        width=page.width,
        height=page.height,
        svg_path=storage.page_svg_path(drawing.id, index),
        pdf_path=pdf_path,
        source_path=storage.original_path(drawing),
    )


def _ready_drawing(storage: Storage, drawing_id: str) -> Drawing:
    drawing = storage.get_drawing(drawing_id)
    if drawing is None:
        raise HTTPException(404, "Чертёж не найден")
    if drawing.status != "ready":
        raise HTTPException(409, f"Чертёж «{drawing.name}» не обработан: {drawing.error}")
    return drawing


def _collect_templates(storage: Storage, req: SearchRequest) -> tuple[list[Template], list[str]]:
    templates: list[Template] = []
    skipped: list[str] = []
    if req.query is not None:
        drawing = _ready_drawing(storage, req.query.drawing_id)
        if drawing.page(req.query.page) is None:
            raise HTTPException(422, f"На чертеже нет страницы {req.query.page}")
        page = page_ref(storage, drawing, req.query.page)
        templates.append(Template(page=page, bbox=req.query.bbox, name="Выделенный фрагмент"))

    for symbol_id in req.symbol_ids:
        symbol = storage.get_symbol(symbol_id)
        if symbol is None:
            raise HTTPException(404, f"Обозначение {symbol_id} не найдено")
        drawing = storage.get_drawing(symbol.drawing_id)
        if drawing is None or drawing.status != "ready" or drawing.page(symbol.page) is None:
            skipped.append(symbol.name)
            continue
        page = page_ref(storage, drawing, symbol.page)
        templates.append(
            Template(page=page, bbox=symbol.bbox, symbol_id=symbol.id, name=symbol.name)
        )

    if not templates:
        raise HTTPException(409, "Ни у одного образца нет доступного исходного чертежа")
    return templates, skipped
