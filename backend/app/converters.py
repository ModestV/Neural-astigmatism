"""Подготовка чертежа к просмотру: исходный файл → постраничные SVG (+ PNG-миниатюры).

Система координат страницы («единицы страницы») совпадает с координатами SVG страницы:
(0, 0) — левый верхний угол, ось Y вниз, размер — PageInfo.width × PageInfo.height.

  * PDF      — пункты PDF (1/72 дюйма), как page.rect в PyMuPDF;
  * DXF, DWG — сначала рендерятся в PDF (ezdxf, только пространство модели), дальше как PDF;
               промежуточный PDF сохраняется в converted.pdf;
  * SVG      — единицы viewBox исходного файла, отсчитанные от начала viewBox.

Миниатюры SVG не делаем: MuPDF не понимает CSS внутри SVG и рисует такие файлы неверно,
поэтому для SVG фронтенд показывает в качестве миниатюры саму страницу.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = ("pdf", "svg", "dxf", "dwg")
THUMB_MAX_SIDE = 480


class ConversionError(Exception):
    """Файл не удалось подготовить. Текст ошибки показывается пользователю."""


@dataclass
class PreparedPage:
    width: float
    height: float


@dataclass
class Prepared:
    units: str
    pages: list[PreparedPage]


def detect_format(filename: str) -> str | None:
    ext = Path(filename).suffix.lower().lstrip(".")
    return ext if ext in SUPPORTED_FORMATS else None


def dwg_converter() -> str | None:
    """Какой конвертер DWG → DXF доступен (None — никакой)."""
    from ezdxf.addons import odafc

    if odafc.is_installed():
        return "ODA File Converter"
    if shutil.which("dwg2dxf"):
        return "LibreDWG (dwg2dxf)"
    return None


def prepare(source: Path, fmt: str, pages_dir: Path, converted_pdf: Path) -> Prepared:
    """Создаёт в pages_dir файлы <n>.svg и <n>.png (n — номер страницы с 1)."""
    if pages_dir.exists():
        shutil.rmtree(pages_dir)
    pages_dir.mkdir(parents=True)
    try:
        if fmt == "pdf":
            return _prepare_pdf(_open_pdf(source.read_bytes()), pages_dir)
        if fmt == "svg":
            return _prepare_svg(source, pages_dir)
        if fmt == "dxf":
            pdf = _dxf_to_pdf(source)
        elif fmt == "dwg":
            with tempfile.TemporaryDirectory() as tmp:
                pdf = _dxf_to_pdf(_dwg_to_dxf(source, Path(tmp)))
        else:
            raise ConversionError(f"Формат «{fmt}» не поддерживается")
        converted_pdf.write_bytes(pdf)
        return _prepare_pdf(_open_pdf(pdf), pages_dir)
    except ConversionError:
        raise
    except Exception as exc:  # файл от пользователя может сломать что угодно
        logger.exception("Не удалось подготовить %s", source)
        raise ConversionError(f"Не удалось обработать файл: {exc}") from exc


# --- PDF -------------------------------------------------------------------


def _open_pdf(data: bytes) -> pymupdf.Document:
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise ConversionError(f"Файл не похож на корректный PDF: {exc}") from exc
    if doc.needs_pass:
        raise ConversionError("PDF защищён паролем")
    if doc.page_count == 0:
        raise ConversionError("В PDF нет страниц")
    return doc


def _prepare_pdf(doc: pymupdf.Document, pages_dir: Path) -> Prepared:
    pages = []
    with doc:
        for index, page in enumerate(doc, start=1):
            rect = page.rect
            (pages_dir / f"{index}.svg").write_text(page.get_svg_image(), encoding="utf-8")
            scale = THUMB_MAX_SIDE / max(rect.width, rect.height)
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
            pixmap.save(pages_dir / f"{index}.png")
            pages.append(PreparedPage(rect.width, rect.height))
    return Prepared(units="pt", pages=pages)


# --- SVG -------------------------------------------------------------------

_SVG_ROOT_RE = re.compile(rb"<svg\b[^>]*>")
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_PX_PER_UNIT = {"": 1.0, "px": 1.0, "pt": 96 / 72, "pc": 16.0, "mm": 96 / 25.4, "cm": 96 / 2.54, "in": 96.0}


def _prepare_svg(source: Path, pages_dir: Path) -> Prepared:
    """Сам SVG браузер покажет как есть. Здесь мы только узнаём размер страницы и
    дописываем в корневой <svg> недостающие viewBox/width/height: без viewBox картинка
    в <img> не масштабируется, а без width/height Firefox не рисует её на canvas."""
    data = source.read_bytes()
    match = _SVG_ROOT_RE.search(data, 0, 1_000_000)
    if not match:
        raise ConversionError("В файле не найден корневой элемент <svg>")
    tag = match.group(0).decode("utf-8", errors="replace")

    viewbox = _parse_viewbox(_attr(tag, "viewBox"))
    width = _parse_length(_attr(tag, "width"))
    height = _parse_length(_attr(tag, "height"))
    if viewbox:
        page_w, page_h = viewbox[2], viewbox[3]
    elif width and height:
        page_w, page_h = width, height  # без viewBox пользовательские единицы — это px
    else:
        raise ConversionError("У SVG нет ни viewBox, ни width/height — не удаётся определить размер")

    new_tag = tag
    if not viewbox:
        new_tag = _set_attr(new_tag, "viewBox", f"0 0 {page_w:g} {page_h:g}")
    if not (width and height):
        new_tag = _set_attr(new_tag, "width", f"{page_w:g}")
        new_tag = _set_attr(new_tag, "height", f"{page_h:g}")
    if new_tag != tag:
        data = data[: match.start()] + new_tag.encode("utf-8") + data[match.end() :]

    (pages_dir / "1.svg").write_bytes(data)
    return Prepared(units="svg", pages=[PreparedPage(page_w, page_h)])


def _attr(tag: str, name: str) -> str | None:
    match = re.search(rf'\s{name}\s*=\s*(["\'])(.*?)\1', tag, re.S)
    return match.group(2).strip() if match else None


def _set_attr(tag: str, name: str, value: str) -> str:
    tag = re.sub(rf'\s{name}\s*=\s*(["\']).*?\1', "", tag, flags=re.S)
    end = -2 if tag.endswith("/>") else -1
    return f'{tag[:end]} {name}="{value}"{tag[end:]}'


def _parse_viewbox(value: str | None) -> tuple[float, float, float, float] | None:
    if not value:
        return None
    numbers = [float(n) for n in _NUMBER_RE.findall(value)]
    if len(numbers) != 4 or numbers[2] <= 0 or numbers[3] <= 0:
        return None
    return numbers[0], numbers[1], numbers[2], numbers[3]


def _parse_length(value: str | None) -> float | None:
    """Длина в px; проценты и неизвестные единицы — None."""
    if not value:
        return None
    match = re.fullmatch(r"\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([a-z]*)\s*", value)
    if not match or match.group(2) not in _PX_PER_UNIT:
        return None
    length = float(match.group(1)) * _PX_PER_UNIT[match.group(2)]
    return length if length > 0 else None


# --- DXF / DWG -------------------------------------------------------------


def _dxf_to_pdf(source: Path) -> bytes:
    import ezdxf
    from ezdxf import recover
    from ezdxf.addons.drawing import Frontend, RenderContext, config, layout
    from ezdxf.addons.drawing.pymupdf import PyMuPdfBackend

    try:
        doc, _auditor = recover.readfile(source)
    except (OSError, ezdxf.DXFStructureError) as exc:
        raise ConversionError(f"Не удалось прочитать DXF: {exc}") from exc
    msp = doc.modelspace()
    if len(msp) == 0:
        raise ConversionError("В пространстве модели DXF нет объектов")

    backend = PyMuPdfBackend()
    cfg = config.Configuration(
        background_policy=config.BackgroundPolicy.WHITE,
        color_policy=config.ColorPolicy.BLACK,
    )
    Frontend(RenderContext(doc), backend, config=cfg).draw_layout(msp, finalize=True)
    # Размер страницы 0×0 — подобрать по содержимому в масштабе 1:1 (единицы чертежа — мм).
    page = layout.Page(0, 0, layout.Units.mm, margins=layout.Margins.all(5))
    return backend.get_pdf_bytes(page)


def _dwg_to_dxf(source: Path, workdir: Path) -> Path:
    from ezdxf.addons import odafc

    target = workdir / "converted.dxf"
    if odafc.is_installed():
        odafc.convert(source, target, version="R2018", replace=True)
        return target
    dwg2dxf = shutil.which("dwg2dxf")
    if dwg2dxf:
        subprocess.run(
            [dwg2dxf, "-y", "-o", str(target), str(source)],
            check=True,
            capture_output=True,
            timeout=300,
        )
        return target
    raise ConversionError(
        "Для DWG нужен конвертер: ODA File Converter или LibreDWG (dwg2dxf). "
        "Установите любой из них и нажмите «Повторить обработку» либо загрузите DXF или PDF."
    )
