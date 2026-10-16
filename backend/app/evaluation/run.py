"""Прогон алгоритма поиска по чертежам из разметки — без веб-сервиса.

Образцы для поиска берутся из самой разметки: для каждого обозначения — первый его экземпляр
в таблице условных обозначений (legend). Так ведёт себя пользователь: обводит символ в легенде
и ищет его по чертежу.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from ..converters import detect_format, prepare
from ..detection import Detector, PageRef, Template
from ..models import BBox, Drawing, PageInfo, SearchParams
from ..storage import new_id
from .dataset import Annotations, Instance


class RunError(RuntimeError):
    """Прогон невозможен: нет чертежа, нет образцов и т. п."""


def run_detector(
    detector: Detector,
    truth: Annotations,
    drawings_dir: Path,
    params: SearchParams | None = None,
    progress: Callable[[str], None] | None = None,
) -> Annotations:
    """Ищет обозначения из разметки на всех её чертежах и возвращает ответы алгоритма."""
    params = params or SearchParams()
    result = Annotations(symbol_names=dict(truth.symbol_names), detector=detector.name)
    with tempfile.TemporaryDirectory(prefix="symsearch-eval-") as tmp:
        for file in truth.files:
            pages = _prepare_drawing(drawings_dir / file, Path(tmp) / new_id())
            templates = _templates(truth, file, pages)
            started = time.perf_counter()
            for page in pages.values():
                for det in detector.detect(page, templates, params):
                    symbol = det.symbol_id or "?"
                    bbox = (det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h)
                    result.instances.append(
                        Instance(file, det.page, symbol, bbox, score=det.score, angle=det.angle)
                    )
            result.times[file] = time.perf_counter() - started
            if progress:
                progress(f"{file}: {len(pages)} лист(ов), {result.times[file]:.3f} с")
    return result


def _prepare_drawing(source: Path, workdir: Path) -> dict[int, PageRef]:
    if not source.is_file():
        raise RunError(f"Нет чертежа {source}")
    fmt = detect_format(source.name)
    if fmt is None:
        raise RunError(f"Неизвестный формат файла {source.name}")
    pages_dir, converted_pdf = workdir / "pages", workdir / "converted.pdf"
    prepared = prepare(source, fmt, pages_dir, converted_pdf)
    drawing = Drawing(
        id=new_id(),
        name=source.name,
        format=fmt,
        size_bytes=source.stat().st_size,
        units=prepared.units,
        pages=[
            PageInfo(index=n, width=p.width, height=p.height)
            for n, p in enumerate(prepared.pages, start=1)
        ],
    )
    pdf_path = {"pdf": source, "svg": None}.get(fmt, converted_pdf)
    return {
        p.index: PageRef(
            drawing=drawing,
            index=p.index,
            width=p.width,
            height=p.height,
            svg_path=pages_dir / f"{p.index}.svg",
            pdf_path=pdf_path,
            source_path=source,
        )
        for p in drawing.pages
    }


def _templates(truth: Annotations, file: str, pages: dict[int, PageRef]) -> list[Template]:
    """По одному образцу на обозначение — первый экземпляр в легенде этого чертежа."""
    templates: dict[str, Template] = {}
    for inst in sorted(truth.instances, key=lambda i: i.page):
        if inst.file == file and inst.legend and inst.symbol not in templates:
            x, y, w, h = inst.bbox
            templates[inst.symbol] = Template(
                page=pages[inst.page],
                bbox=BBox(x=x, y=y, w=w, h=h),
                symbol_id=inst.symbol,
                name=truth.symbol_names.get(inst.symbol, inst.symbol),
            )
    if not templates:
        raise RunError(
            f"{file}: в разметке нет экземпляров с legend=true, не из чего взять образцы"
        )
    return list(templates.values())
