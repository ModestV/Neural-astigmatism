"""Хранилище на файлах: метаданные — JSON, рядом — сами файлы.

    data/
      drawings/<id>/meta.json        Drawing
      drawings/<id>/original.<ext>   исходный файл
      drawings/<id>/converted.pdf    промежуточный PDF (только для DXF/DWG)
      drawings/<id>/pages/<n>.svg    страница для просмотра
      drawings/<id>/pages/<n>.png    миниатюра (для SVG не создаётся)
      symbols/<id>/meta.json         Symbol
      symbols/<id>/preview.png       превью образца
      runs/<id>.json                 SearchRun

Формат простой и читается глазами — удобно, чтобы собирать из библиотеки датасет.
Если данных станет много, этот модуль можно заменить на БД, не трогая остальной код.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .models import Drawing, SearchRun, Symbol

logger = logging.getLogger(__name__)

_ID_RE = re.compile(r"^[0-9a-f]{12}$")
M = TypeVar("M", bound=BaseModel)


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def is_valid_id(value: str) -> bool:
    """Защита от обхода путей: в путь к файлу попадают только id нашего формата."""
    return bool(_ID_RE.match(value))


class Storage:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.drawings_dir = root / "drawings"
        self.symbols_dir = root / "symbols"
        self.runs_dir = root / "runs"
        for directory in (self.drawings_dir, self.symbols_dir, self.runs_dir):
            directory.mkdir(parents=True, exist_ok=True)

    # --- чертежи ---------------------------------------------------------

    def drawing_dir(self, drawing_id: str) -> Path:
        return self.drawings_dir / drawing_id

    def original_path(self, drawing: Drawing) -> Path:
        return self.drawing_dir(drawing.id) / f"original.{drawing.format}"

    def converted_pdf_path(self, drawing_id: str) -> Path:
        return self.drawing_dir(drawing_id) / "converted.pdf"

    def pages_dir(self, drawing_id: str) -> Path:
        return self.drawing_dir(drawing_id) / "pages"

    def page_svg_path(self, drawing_id: str, page: int) -> Path:
        return self.pages_dir(drawing_id) / f"{page}.svg"

    def page_thumb_path(self, drawing_id: str, page: int) -> Path:
        return self.pages_dir(drawing_id) / f"{page}.png"

    def get_drawing(self, drawing_id: str) -> Drawing | None:
        if not is_valid_id(drawing_id):
            return None
        return self._read(self.drawing_dir(drawing_id) / "meta.json", Drawing)

    def list_drawings(self) -> list[Drawing]:
        items = self._read_all(self.drawings_dir.glob("*/meta.json"), Drawing)
        return sorted(items, key=lambda d: d.created_at, reverse=True)

    def save_drawing(self, drawing: Drawing) -> None:
        self._write(self.drawing_dir(drawing.id) / "meta.json", drawing)

    def delete_drawing(self, drawing_id: str) -> bool:
        if not is_valid_id(drawing_id) or not self.drawing_dir(drawing_id).is_dir():
            return False
        shutil.rmtree(self.drawing_dir(drawing_id))
        return True

    # --- библиотека обозначений ------------------------------------------

    def symbol_dir(self, symbol_id: str) -> Path:
        return self.symbols_dir / symbol_id

    def symbol_preview_path(self, symbol_id: str) -> Path:
        return self.symbol_dir(symbol_id) / "preview.png"

    def get_symbol(self, symbol_id: str) -> Symbol | None:
        if not is_valid_id(symbol_id):
            return None
        return self._read(self.symbol_dir(symbol_id) / "meta.json", Symbol)

    def list_symbols(self) -> list[Symbol]:
        items = self._read_all(self.symbols_dir.glob("*/meta.json"), Symbol)
        return sorted(items, key=lambda s: s.created_at, reverse=True)

    def save_symbol(self, symbol: Symbol, preview_png: bytes | None = None) -> None:
        if preview_png is not None:
            path = self.symbol_preview_path(symbol.id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(preview_png)
        self._write(self.symbol_dir(symbol.id) / "meta.json", symbol)

    def delete_symbol(self, symbol_id: str) -> bool:
        if not is_valid_id(symbol_id) or not self.symbol_dir(symbol_id).is_dir():
            return False
        shutil.rmtree(self.symbol_dir(symbol_id))
        return True

    # --- запуски поиска --------------------------------------------------

    def get_run(self, run_id: str) -> SearchRun | None:
        if not is_valid_id(run_id):
            return None
        return self._read(self.runs_dir / f"{run_id}.json", SearchRun)

    def list_runs(self) -> list[SearchRun]:
        items = self._read_all(self.runs_dir.glob("*.json"), SearchRun)
        return sorted(items, key=lambda r: r.created_at, reverse=True)

    def save_run(self, run: SearchRun) -> None:
        self._write(self.runs_dir / f"{run.id}.json", run)

    # --- служебное -------------------------------------------------------

    @staticmethod
    def _write(path: Path, model: BaseModel) -> None:
        # Пишем во временный файл и атомарно подменяем, чтобы не оставить «полуфайл».
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(model.model_dump_json(indent=2))
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    @staticmethod
    def _read(path: Path, cls: type[M]) -> M | None:
        try:
            return cls.model_validate_json(path.read_bytes())
        except FileNotFoundError:
            return None
        except ValidationError as exc:
            logger.warning("Повреждённый файл %s: %s", path, exc)
            return None

    @staticmethod
    def _read_all(paths, cls: type[M]) -> list[M]:
        items = []
        for path in paths:
            try:
                items.append(cls.model_validate_json(path.read_bytes()))
            except (OSError, ValidationError) as exc:
                logger.warning("Пропускаю повреждённый файл %s: %s", path, exc)
        return items
