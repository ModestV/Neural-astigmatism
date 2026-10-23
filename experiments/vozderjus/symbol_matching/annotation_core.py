"""Рабочая разметка, импорт COCO и экспорт конфигурации исследования."""

from __future__ import annotations

import json
import math
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, model_validator

from models import Box, Experiment, StrictModel, TemplateSpec, TargetSpec, Truth
from pdf_io import checksum, pdf_path, validate_box


class SymbolClass(StrictModel):
    id: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=200)
    enabled: bool = False


class Mark(StrictModel):
    id: str = Field(min_length=1)
    symbol: str
    bbox: Box
    legend: bool = False
    template: bool = False
    source: dict[str, Any] = Field(default_factory=dict)


class AnnotationPage(StrictModel):
    id: str
    file: str
    page: int = Field(ge=1)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    rotation: Literal[0, 90, 180, 270] = 0
    sha256: str
    annotations: list[Mark] = Field(default_factory=list)
    reviewed_symbols: list[str] = Field(default_factory=list)


class Workspace(StrictModel):
    schema_version: Literal[1] = 1
    revision: int = Field(default=0, ge=0)
    symbols: list[SymbolClass] = Field(default_factory=list)
    pages: list[AnnotationPage] = Field(default_factory=list)
    imports: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistency(self) -> Workspace:
        ids = [s.id for s in self.symbols]
        if len(ids) != len(set(ids)):
            raise ValueError("Повтор идентификатора класса")
        pages = [p.id for p in self.pages]
        if len(pages) != len(set(pages)):
            raise ValueError("Повтор идентификатора страницы")
        pairs = [(p.file, p.page) for p in self.pages]
        if len(pairs) != len(set(pairs)):
            raise ValueError("Повтор PDF-страницы")
        annotations = [a.id for p in self.pages for a in p.annotations]
        if len(annotations) != len(set(annotations)):
            raise ValueError("Повтор идентификатора рамки")
        for page in self.pages:
            if not set(page.reviewed_symbols).issubset(ids):
                raise ValueError("Неизвестный класс в отметке проверки страницы")
            for mark in page.annotations:
                if mark.symbol not in ids:
                    raise ValueError(f"Неизвестный класс {mark.symbol}")
                validate_box(mark.bbox, page.model_dump())
        return self


def discover_workspace(drawings: Path) -> Workspace:
    """Перечислить страницы PDF; загруженную разметку пока не считаем проверенной."""
    import pymupdf

    pages = []
    for path in sorted(drawings.rglob("*")):
        if path.suffix.lower() != ".pdf" or not path.is_file():
            continue
        relative = path.relative_to(drawings).as_posix()
        pdf_path(drawings, relative)
        digest = checksum(path)
        with pymupdf.open(path) as document:
            if document.needs_pass:
                raise ValueError(f"PDF защищён паролем: {relative}")
            for index, page in enumerate(document, 1):
                pages.append(
                    AnnotationPage(
                        id=f"page_{len(pages) + 1}",
                        file=relative,
                        page=index,
                        width=page.rect.width,
                        height=page.rect.height,
                        rotation=page.rotation,
                        sha256=digest,
                    )
                )
    if not pages:
        raise ValueError("В drawings не найдены PDF")
    return Workspace(pages=pages)


def image_page(image_name: str, pages: list[AnnotationPage]) -> AnnotationPage:
    """Сопоставить имена CVAT, включая [1] и хвост ...000000000.jpeg-1.jpg."""
    name = Path(image_name.replace("\\", "/")).name
    name = re.sub(
        r"0{9}\.(?:jpeg|jpg|png)(?:-\d+\.(?:jpg|jpeg|png))?$", "", name, flags=re.IGNORECASE
    )
    name = re.sub(r"\.(?:jpg|jpeg|png)$", "", name, flags=re.IGNORECASE)
    match = re.fullmatch(r"(.+?)\[(\d+)\]", name)
    stem, page_number = (match[1], int(match[2])) if match else (name, 1)
    candidates = [p for p in pages if Path(p.file).stem == stem and p.page == page_number]
    if len(candidates) != 1:
        raise ValueError(f"Неоднозначное или отсутствующее соответствие PDF для {image_name}")
    return candidates[0]


def import_coco(workspace: Workspace, path: Path, drawings: Path) -> Workspace:
    """Импортировать копию COCO, переводя пиксельные bbox в координаты видимой PDF-страницы."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not all(key in data for key in ("images", "annotations", "categories")):
        raise ValueError("Ожидается COCO с images, annotations и categories")
    digest = checksum(path)
    if any(record["sha256"] == digest for record in workspace.imports):
        return workspace.model_copy(deep=True)
    result = workspace.model_copy(deep=True)
    categories = {item["id"]: item["name"] for item in data["categories"]}
    images = {item["id"]: item for item in data["images"]}
    if len(images) != len(data["images"]):
        raise ValueError("Повтор image_id в COCO")
    annotation_ids = [a["id"] for a in data["annotations"]]
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("Повтор annotation_id в COCO")
    mapped = {key: image_page(image["file_name"], result.pages) for key, image in images.items()}
    used_pages = [p.id for p in mapped.values()]
    if len(set(used_pages)) != len(used_pages):
        raise ValueError("Несколько изображений COCO ссылаются на одну PDF-страницу")
    for key, page in mapped.items():
        image = images[key]
        width, height = float(image["width"]), float(image["height"])
        if not math.isfinite(width + height) or width <= 0 or height <= 0:
            raise ValueError("Некорректный размер изображения COCO")
        # Только равномерный рендер: растяжение или обрезка не должны маскироваться масштабом.
        if abs((width / height) / (page.width / page.height) - 1) > 0.005:
            raise ValueError(f"Не совпадает ориентация/пропорции COCO и PDF: {image['file_name']}")
        if page.annotations:
            raise ValueError(
                f"На {page.file} уже есть рамки; повторный импорт не объединяется молча"
            )
    known = {s.id for s in result.symbols}
    for annotation in data["annotations"]:
        if annotation["image_id"] not in images:
            raise ValueError("Аннотация ссылается на неизвестное изображение")
        page = mapped[annotation["image_id"]]
        image = images[annotation["image_id"]]
        category = categories.get(annotation["category_id"])
        if category is None:
            raise ValueError("Аннотация ссылается на неизвестную категорию")
        if annotation.get("iscrowd", 0):
            raise ValueError("COCO iscrowd требует отдельного правила оценки и не импортируется")
        attributes = annotation.get("attributes", {})
        legend_id = str(attributes.get("legend_id") or "").strip()
        if category in ("legend_item", "object"):
            if not legend_id:
                raise ValueError(f"У {category} отсутствует legend_id")
            namespace = f"{Path(page.file).stem}_p{page.page}"
            symbol_id = f"{namespace}__legend_{legend_id}"
            symbol_name = f"{Path(page.file).stem} · ИУО {legend_id}"
        else:
            # Неизвестные/временные категории не выдаём за определённый тип ИУО.
            raise ValueError(f"Категория {category!r} требует явного сопоставления классов")
        if symbol_id not in known:
            result.symbols.append(SymbolClass(id=symbol_id, name=symbol_name))
            known.add(symbol_id)
        x, y, width, height = annotation["bbox"]
        box = Box(
            x=x * page.width / image["width"],
            y=y * page.height / image["height"],
            w=width * page.width / image["width"],
            h=height * page.height / image["height"],
        )
        validate_box(box, page.model_dump())
        page.annotations.append(
            Mark(
                id=f"coco_{digest[:12]}_{annotation['id']}",
                symbol=symbol_id,
                bbox=box,
                legend=category == "legend_item",
                source={
                    "coco_sha256": digest,
                    "annotation_id": annotation["id"],
                    "image_name": image["file_name"],
                    "attributes": attributes,
                },
            )
        )
    empty = [p.file for p in mapped.values() if not p.annotations]
    for name in empty:
        result.warnings.append(
            f"{name}: импортированное изображение не содержит рамок; "
            "это не подтверждение отсутствия ИУО"
        )
    rotations = sum(bool(a.get("attributes", {}).get("rotation", 0)) for a in data["annotations"])
    if rotations:
        result.warnings.append(
            f"{path.name}: у {rotations} рамок есть атрибут rotation; "
            "сохранён в source, используются осевые bbox из COCO"
        )
    result.imports.append(
        {
            "file": path.relative_to(drawings).as_posix(),
            "sha256": digest,
            "images": len(images),
            "annotations": len(data["annotations"]),
        }
    )
    return Workspace.model_validate(result.model_dump())


def export_experiment(
    workspace: Workspace,
    page_id: str | None = None,
) -> tuple[Experiment, list[str]]:
    """Экспортировать выбранные классы; непроверенные страницы остаются без метрик."""
    pages = [p for p in workspace.pages if page_id is None or p.id == page_id]
    if not pages:
        raise ValueError("Страница не найдена")
    page_symbols = {a.symbol for p in pages for a in p.annotations}
    page_symbols.update(s for p in pages for s in p.reviewed_symbols)
    enabled = {s.id for s in workspace.symbols if s.enabled and s.id in page_symbols}
    if not enabled:
        raise ValueError("Выберите хотя бы один исследуемый класс")
    templates = [
        TemplateSpec(id=a.id, symbol=a.symbol, file=p.file, page=p.page, bbox=a.bbox)
        for p in workspace.pages
        for a in p.annotations
        if a.template and a.symbol in enabled
    ]
    missing = enabled - {t.symbol for t in templates}
    if missing:
        raise ValueError("Не выбран образец для классов: " + ", ".join(sorted(missing)))
    targets, warnings = [], []
    for page in pages:
        relevant = enabled & ({a.symbol for a in page.annotations} | set(page.reviewed_symbols))
        if not relevant:
            continue
        complete = relevant.issubset(page.reviewed_symbols)
        ground_truth = (
            [
                Truth(symbol=a.symbol, bbox=a.bbox, legend=a.legend)
                for a in page.annotations
                if a.symbol in enabled
            ]
            if complete
            else None
        )
        if not complete:
            warnings.append(
                f"{page.file}, лист {page.page}: разметка выбранных классов "
                "не подтверждена полностью; метрики будут null"
            )
        # Все известные элементы легенды исключаются, даже для невыбранных классов.
        ignore_regions = [a.bbox for a in page.annotations if a.legend]
        targets.append(
            TargetSpec(
                file=page.file,
                page=page.page,
                ground_truth=ground_truth,
                ignore_regions=ignore_regions,
            )
        )
    if not targets:
        raise ValueError("Нет страниц с выбранными классами")
    return Experiment(
        description="Выбрано в локальном редакторе разметки; непроверенные страницы не оцениваются",
        templates=templates,
        targets=targets,
    ), warnings


class ConflictError(ValueError):
    """Изменения другой вкладки нельзя перезаписывать без перечитывания."""


class AnnotationStore:
    """Атомарное сохранение, резервные копии и контроль версии документа."""

    def __init__(self, path: Path, drawings: Path):
        self.path = path
        self.drawings = drawings
        self.lock = threading.RLock()

    def load(self) -> Workspace:
        with self.lock:
            return Workspace.model_validate_json(self.path.read_text(encoding="utf-8"))

    def validate_sources(self, workspace: Workspace) -> None:
        import pymupdf

        documents = {}
        for page in workspace.pages:
            path = pdf_path(self.drawings, page.file)
            if path not in documents:
                with pymupdf.open(path) as doc:
                    documents[path] = (
                        checksum(path),
                        [(p.rect.width, p.rect.height, p.rotation) for p in doc],
                    )
            digest, pages = documents[path]
            if digest != page.sha256 or not 1 <= page.page <= len(pages):
                raise ValueError(f"PDF изменился или страница удалена: {page.file}")
            width, height, rotation = pages[page.page - 1]
            if (
                abs(width - page.width) > 0.01
                or abs(height - page.height) > 0.01
                or rotation != page.rotation
            ):
                raise ValueError(f"Не совпадают размеры или поворот PDF: {page.file}")

    def save(self, workspace: Workspace) -> Workspace:
        with self.lock:
            workspace = Workspace.model_validate(workspace.model_dump())
            if self.path.exists():
                current = self.load()
                if current.revision != workspace.revision:
                    raise ConflictError(
                        "Разметка изменена в другой вкладке. "
                        "Скачайте свою копию и перезагрузите страницу."
                    )
            self.validate_sources(workspace)
            result = workspace.model_copy(update={"revision": workspace.revision + 1})
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                backups = self.path.parent / ".annotation_backups"
                backups.mkdir(exist_ok=True)
                backup = backups / (
                    self.path.stem + f".r{workspace.revision}.{uuid4().hex[:8]}.json"
                )
                shutil.copyfile(self.path, backup)
                for old in sorted(
                    backups.glob(self.path.stem + ".r*.json"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )[20:]:
                    old.unlink()
            temporary = self.path.with_name(self.path.name + f".{uuid4().hex}.tmp")
            temporary.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            temporary.replace(self.path)
            return result

    def export(self, output_dir: Path, page_id: str) -> tuple[Path, list[str]]:
        with self.lock:
            workspace = self.load()
            self.validate_sources(workspace)
            config, warnings = export_experiment(workspace, page_id)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output = output_dir / f"experiment_{stamp}_{uuid4().hex[:8]}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(config.model_dump_json(indent=2), encoding="utf-8")
            return output, warnings
