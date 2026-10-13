#!/usr/bin/env python3
"""Конвертация и открытие SVG, PNG и PDF для чертежей.

PNG остаётся растровым форматом: преобразование PNG -> SVG создаёт SVG-контейнер
со встроенным исходным изображением, а не имитацию векторизации.
"""

from __future__ import annotations

import argparse
import base64
import importlib
import os
import platform
import shutil
import subprocess
import sys
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

VALID_FORMATS = {".svg", ".png", ".pdf"}


def require_module(name: str):
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise RuntimeError(
            f"Нужен пакет {name}. Установите зависимости: python3 -m pip install -r requirements.txt"
        ) from error


def optional_module(name: str):
    try:
        return importlib.import_module(name)
    except (ImportError, OSError):
        return None


def svg_pixel_size(source: Path, dpi: int) -> tuple[int, int]:
    """Возвращает размер SVG в пикселях для целевого DPI."""
    root = ET.parse(source).getroot()
    def to_pixels(dimension: str) -> int | None:
        match = re.fullmatch(r"\s*([0-9.]+)\s*(mm|cm|in|pt|px)?\s*", dimension)
        if not match:
            return None
        value = float(match.group(1))
        unit = match.group(2) or "px"
        inches = {
            "mm": value / 25.4,
            "cm": value / 2.54,
            "in": value,
            "pt": value / 72,
            "px": value / 96,
        }[unit]
        return max(1, round(inches * dpi))
    width = to_pixels(root.get("width", ""))
    height = to_pixels(root.get("height", ""))
    if width and height:
        return width, height
    view_box = root.get("viewBox", "").split()
    if len(view_box) == 4:
        return (
            max(1, round(float(view_box[2]) * dpi / 96)),
            max(1, round(float(view_box[3]) * dpi / 96)),
        )
    raise RuntimeError("У SVG не указаны размеры. Добавьте width/height или viewBox.")


def chrome_binary() -> str | None:
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
    ]
    return next((candidate for candidate in candidates if candidate and Path(candidate).exists()), None)


def chrome_svg_to_png(source: Path, target: Path, dpi: int) -> None:
    chrome = chrome_binary()
    if not chrome:
        raise RuntimeError("Нужен CairoSVG или Chrome/Chromium для SVG -> PNG.")
    width, height = svg_pixel_size(source, dpi)
    with tempfile.TemporaryDirectory(prefix="drawing-converter-") as profile:
        subprocess.run(
            [chrome, "--headless=new", "--disable-gpu", "--no-sandbox", "--allow-file-access-from-files",
             f"--user-data-dir={profile}", f"--window-size={width},{height}",
             f"--screenshot={target}", source.resolve().as_uri()],
            check=True, capture_output=True, text=True,
        )


def chrome_svg_to_pdf(source: Path, target: Path) -> None:
    chrome = chrome_binary()
    if not chrome:
        raise RuntimeError("Нужен CairoSVG или Chrome/Chromium для SVG -> PDF.")
    # Временная обёртка фиксирует физический формат A4 и не добавляет поля браузера.
    html = (
        "<!doctype html><style>@page{size:A4 landscape;margin:0}"
        "html,body,img{display:block;margin:0;width:297mm;height:210mm}</style>"
        f'<img src="{source.resolve().as_uri()}">'
    )
    with tempfile.TemporaryDirectory(prefix="drawing-converter-") as temporary:
        page = Path(temporary) / "drawing.html"
        page.write_text(html, encoding="utf-8")
        subprocess.run(
            [chrome, "--headless=new", "--disable-gpu", "--no-sandbox", "--allow-file-access-from-files",
             "--no-pdf-header-footer", f"--user-data-dir={Path(temporary) / 'profile'}",
             f"--print-to-pdf={target}", page.as_uri()],
            check=True, capture_output=True, text=True,
        )


def ensure_paths(source: Path, target: Path) -> tuple[str, str]:
    source_suffix = source.suffix.lower()
    target_suffix = target.suffix.lower()
    if not source.is_file():
        raise ValueError(f"Файл не найден: {source}")
    if source_suffix not in VALID_FORMATS or target_suffix not in VALID_FORMATS:
        raise ValueError("Поддерживаются только SVG, PNG и PDF.")
    if source_suffix == target_suffix:
        raise ValueError("Исходный и целевой форматы совпадают.")
    target.parent.mkdir(parents=True, exist_ok=True)
    return source_suffix, target_suffix


def svg_to_png(source: Path, target: Path, dpi: int) -> None:
    cairosvg = optional_module("cairosvg")
    if cairosvg:
        cairosvg.svg2png(url=str(source), write_to=str(target), scale=dpi / 96)
        return
    chrome_svg_to_png(source, target, dpi)


def svg_to_pdf(source: Path, target: Path) -> None:
    cairosvg = optional_module("cairosvg")
    if cairosvg:
        cairosvg.svg2pdf(url=str(source), write_to=str(target))
        return
    chrome_svg_to_pdf(source, target)


def png_to_svg(source: Path, target: Path) -> None:
    pillow_image = require_module("PIL.Image")
    with pillow_image.open(source) as image:
        width, height = image.size
    encoded = base64.b64encode(source.read_bytes()).decode("ascii")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'  <image width="{width}" height="{height}" href="data:image/png;base64,{encoded}"/>\n'
        "</svg>\n"
    )
    target.write_text(svg, encoding="utf-8")


def png_to_pdf(source: Path, target: Path) -> None:
    image_reader_module = require_module("reportlab.lib.utils")
    canvas_module = require_module("reportlab.pdfgen.canvas")
    reader = image_reader_module.ImageReader(str(source))
    width_px, height_px = reader.getSize()
    # 72 points/inch: не меняем число пикселей и не пережимаем исходный PNG.
    width_pt = width_px * 72 / 300
    height_pt = height_px * 72 / 300
    canvas = canvas_module.Canvas(str(target), pagesize=(width_pt, height_pt))
    canvas.drawImage(reader, 0, 0, width=width_pt, height=height_pt, mask="auto")
    canvas.showPage()
    canvas.save()


def pdf_to_png(source: Path, target: Path, dpi: int) -> None:
    renderer = shutil.which("pdftoppm")
    if not renderer:
        raise RuntimeError("Нужен pdftoppm из Poppler. macOS: brew install poppler")
    subprocess.run(
        [renderer, "-r", str(dpi), "-f", "1", "-l", "1", "-png", "-singlefile", str(source), str(target.with_suffix(""))],
        check=True,
        capture_output=True,
        text=True,
    )


def pdf_to_svg(source: Path, target: Path) -> None:
    converter = shutil.which("pdftocairo")
    if not converter:
        raise RuntimeError("Нужен pdftocairo из Poppler. macOS: brew install poppler")
    subprocess.run(
        [converter, "-f", "1", "-l", "1", "-svg", str(source), str(target)],
        check=True,
        capture_output=True,
        text=True,
    )


def convert(source: Path, target: Path, dpi: int) -> None:
    source_suffix, target_suffix = ensure_paths(source, target)
    handlers = {
        (".svg", ".png"): lambda: svg_to_png(source, target, dpi),
        (".svg", ".pdf"): lambda: svg_to_pdf(source, target),
        (".png", ".svg"): lambda: png_to_svg(source, target),
        (".png", ".pdf"): lambda: png_to_pdf(source, target),
        (".pdf", ".png"): lambda: pdf_to_png(source, target, dpi),
        (".pdf", ".svg"): lambda: pdf_to_svg(source, target),
    }
    handlers[(source_suffix, target_suffix)]()
    print(f"Готово: {target}")


def open_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"Файл не найден: {path}")
    system = platform.system()
    if system == "Darwin":
        command = ["open", str(path)]
    elif system == "Windows":
        os.startfile(path)  # type: ignore[attr-defined]
        return
    else:
        command = ["xdg-open", str(path)]
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Конвертер SVG, PNG и PDF для чертежей")
    commands = parser.add_subparsers(dest="command", required=True)
    convert_parser = commands.add_parser("convert", help="конвертировать файл")
    convert_parser.add_argument("source", type=Path)
    convert_parser.add_argument("target", type=Path)
    convert_parser.add_argument("--dpi", type=int, default=300, help="разрешение растрового результата, по умолчанию 300")
    open_parser = commands.add_parser("open", help="открыть файл системным просмотрщиком")
    open_parser.add_argument("file", type=Path)
    arguments = parser.parse_args()
    try:
        if arguments.command == "convert":
            if arguments.dpi < 72:
                raise ValueError("Для чертежей укажите DPI не ниже 72.")
            convert(arguments.source, arguments.target, arguments.dpi)
        else:
            open_file(arguments.file)
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
