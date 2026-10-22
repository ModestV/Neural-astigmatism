"""Чтение и запись COCO-разметки размеченного набора чертежей.

Что делает модуль, простыми словами: читает JSON-файлы разметки из CVAT, приводит их к одному виду
(классы сопоставляются по имени, а не по номеру) и умеет записать результат обратно в COCO.

Рамки везде `(x, y, w, h)` в пикселях картинки, начало слева сверху, ось Y вниз.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

LEGEND_ITEM = "legend_item"
OBJECT = "object"

# Номера классов в выходном файле всегда одинаковые, как в pdf_part3.json / svg_supertest.json.
CATEGORY_IDS = {LEGEND_ITEM: 1, OBJECT: 2}

# Остальные классы (например, пустой `temporary`) пропускаем.
KNOWN_CATEGORIES = frozenset(CATEGORY_IDS)


@dataclass
class Box:
    """Одна рамка: образец легенды или экземпляр обозначения."""

    category: str  # LEGEND_ITEM или OBJECT
    bbox: tuple[float, float, float, float]  # x, y, w, h
    attributes: dict = field(default_factory=dict)  # legend_id, rotation, occluded и т. д.

    @property
    def legend_id(self) -> str | None:
        value = self.attributes.get("legend_id")
        return None if value is None else str(value)


@dataclass
class Sheet:
    """Одна картинка (лист чертежа) с рамками."""

    file_name: str
    width: int
    height: int
    boxes: list[Box] = field(default_factory=list)
    source: str = ""  # из какого файла разметки лист пришёл
    history: list[str] = field(default_factory=list)  # какие операции применены, по порядку


def load_sheets(annotation_paths: list[Path], skip_empty: bool = True) -> list[Sheet]:
    """Читает COCO-файлы и объединяет их в один список листов.

    Классы сопоставляются по имени (`name`), поэтому разные номера в файлах не мешают.
    Листы без рамок (например, неразмеченный 03_washoe_house) по умолчанию пропускаются.
    """
    sheets: list[Sheet] = []
    for path in annotation_paths:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        names = {c["id"]: c["name"] for c in data["categories"]}
        by_image: dict[int, Sheet] = {}
        for image in data["images"]:
            by_image[image["id"]] = Sheet(
                file_name=image["file_name"],
                width=int(image["width"]),
                height=int(image["height"]),
                source=path.name,
                history=list(image.get("augmentations", [])),
            )
        for ann in data["annotations"]:
            name = names.get(ann["category_id"])
            if name not in KNOWN_CATEGORIES:
                continue
            x, y, w, h = ann["bbox"]
            by_image[ann["image_id"]].boxes.append(
                Box(name, (float(x), float(y), float(w), float(h)), dict(ann.get("attributes", {})))
            )
        for sheet in by_image.values():
            if sheet.boxes or not skip_empty:
                sheets.append(sheet)
    return sheets


def collect_annotation_paths(paths: list[Path]) -> list[Path]:
    """Разворачивает папки в отсортированные списки их *.json, файлы оставляет как есть."""
    result: list[Path] = []
    for path in map(Path, paths):
        result.extend(sorted(path.glob("*.json")) if path.is_dir() else [path])
    return result


def find_problems(
    sheets: list[Sheet], images_dir: Path | None = None, tolerance: float = 1e-3
) -> list[str]:
    """Проверяет разметку и возвращает список найденных проблем (пустой список — всё хорошо).

    Проверяется: рамки с положительным размером и внутри картинки, известные классы, уникальные
    имена файлов, а если дана папка с картинками — что файлы есть и их размеры совпадают с разметкой.
    Флаг `legend_in_tile` у объектов должен совпадать с тем, что реально есть в той же картинке.
    """
    from PIL import Image  # импорт здесь, чтобы чтение разметки работало без Pillow

    problems: list[str] = []
    seen: set[str] = set()
    for sheet in sheets:
        name = sheet.file_name
        if name in seen:
            problems.append(f"{name}: имя файла встречается больше одного раза")
        seen.add(name)
        if sheet.width <= 0 or sheet.height <= 0:
            problems.append(f"{name}: неположительный размер {sheet.width}x{sheet.height}")
        legend_ids = {b.legend_id for b in sheet.boxes if b.category == LEGEND_ITEM}
        for number, box in enumerate(sheet.boxes):
            x, y, w, h = box.bbox
            where = f"{name}, рамка {number} ({box.category})"
            if box.category not in KNOWN_CATEGORIES:
                problems.append(f"{where}: неизвестный класс")
            if w <= 0 or h <= 0:
                problems.append(f"{where}: неположительный размер рамки {w}x{h}")
            if (x < -tolerance or y < -tolerance
                    or x + w > sheet.width + tolerance or y + h > sheet.height + tolerance):
                problems.append(f"{where}: рамка {box.bbox} выходит за картинку "
                                f"{sheet.width}x{sheet.height}")
            flag = box.attributes.get("legend_in_tile")
            if box.category == OBJECT and flag is not None and flag != (box.legend_id in legend_ids):
                problems.append(f"{where}: legend_in_tile={flag} не совпадает с содержимым картинки")
        if images_dir is not None:
            path = Path(images_dir) / name
            if not path.exists():
                problems.append(f"{name}: файла нет в {images_dir}")
                continue
            with Image.open(path) as image:
                if image.size != (sheet.width, sheet.height):
                    problems.append(f"{name}: размер файла {image.size} не совпадает с разметкой "
                                    f"{(sheet.width, sheet.height)}")
    return problems


def save_coco(sheets: list[Sheet], path: Path) -> None:
    """Записывает листы в COCO-файл. Номера картинок и рамок назначаются заново с 1."""
    images = []
    annotations = []
    for image_id, sheet in enumerate(sheets, start=1):
        images.append(
            {
                "id": image_id,
                "width": sheet.width,
                "height": sheet.height,
                "file_name": sheet.file_name,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0,
                "augmentations": sheet.history,
            }
        )
        for box in sheet.boxes:
            x, y, w, h = box.bbox
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": CATEGORY_IDS[box.category],
                    "segmentation": [],
                    "area": round(w * h, 2),
                    "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                    "iscrowd": 0,
                    "attributes": box.attributes,
                }
            )
    data = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {"contributor": "", "date_created": "", "description": "", "url": "", "version": "",
                 "year": ""},
        "categories": [{"id": i, "name": n, "supercategory": ""} for n, i in CATEGORY_IDS.items()],
        "images": images,
        "annotations": annotations,
    }
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
