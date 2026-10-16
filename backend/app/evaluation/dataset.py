"""Разметка и ответы алгоритма в едином внутреннем виде.

Метрики (metrics.py) работают только с этими классами. Если формат датасета изменится,
достаточно дописать здесь новый загрузчик — сами метрики трогать не придётся.

Поддерживаемые форматы (определяются по содержимому файла):

1. Наш JSON — разметка и ответы одинаковой формы, см. samples/ground_truth.json:

    {
      "symbols": {"socket": "Розетка", ...},          # необязательно: ключ → название
      "files": {
        "plan.svg": {
          "time_s": 0.42,                              # только в ответах: время на чертёж
          "pages": {
            "1": [
              {"symbol": "socket", "bbox": [x, y, w, h],
               "angle": 0, "legend": false,            # только в разметке
               "score": 0.93}                          # только в ответах
            ]
          }
        }
      }
    }

2. COCO (например, выгрузка из CVAT) — разметка: {"images", "annotations", "categories"}.
   Каждая картинка — отдельный чертёж из одного листа, обозначение — имя категории,
   угол — атрибут CVAT "rotation". Категория legend_item считается таблицей обозначений.

3. COCO results — ответы алгоритма: [{"image_id", "category_id", "bbox", "score"}, ...].
   Номера картинок и категорий берутся из COCO-разметки (load_predictions).

Рамки — [x, y, w, h], ось Y вниз (как во всём проекте). Единицы — как в источнике:
единицы страницы для нашего формата, пиксели для COCO.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

BBoxTuple = tuple[float, float, float, float]


@dataclass(frozen=True)
class Instance:
    """Один экземпляр обозначения: эталонный (из разметки) или найденный алгоритмом."""

    file: str
    page: int
    symbol: str
    bbox: BBoxTuple
    score: float = 1.0
    """Уверенность алгоритма; у эталона — 1."""
    angle: float = 0.0
    legend: bool = False
    """Экземпляр в таблице условных обозначений."""


@dataclass
class Annotations:
    """Набор экземпляров по всем файлам и листам."""

    instances: list[Instance] = field(default_factory=list)
    symbol_names: dict[str, str] = field(default_factory=dict)
    times: dict[str, float] = field(default_factory=dict)
    """Время обработки каждого файла, секунды (только для ответов алгоритма)."""
    detector: str | None = None

    @property
    def files(self) -> list[str]:
        return sorted({i.file for i in self.instances} | set(self.times))

    def pages(self) -> list[tuple[str, int]]:
        return sorted({(i.file, i.page) for i in self.instances})


class DatasetFormatError(ValueError):
    """Файл разметки или ответов не удалось разобрать."""


LEGEND_CATEGORIES = {"legend_item"}
"""Категории COCO, которые обозначают экземпляры в таблице условных обозначений."""


def load(*paths: Path) -> Annotations:
    """Читает разметку (наш формат или COCO) или ответы в нашем формате.

    Несколько файлов объединяются — например, по одному COCO-файлу на чертёж.
    """
    return merge([_parse(_read(p), str(p)) for p in paths])


def load_predictions(path: Path, truth_paths: list[Path]) -> Annotations:
    """Читает ответы алгоритма: наш формат или COCO results.

    Для COCO results нужна COCO-разметка (truth_paths), чтобы перевести номера картинок
    и категорий в имена файлов и обозначений.
    """
    data = _read(path)
    if not isinstance(data, list):
        return _parse(data, str(path))
    images: dict[int, set[str]] = {}
    categories: dict[int, set[str]] = {}
    for truth_path in truth_paths:
        coco = _read(truth_path)
        if not _is_coco(coco):
            continue
        for image in coco["images"]:
            images.setdefault(image["id"], set()).add(image["file_name"])
        for cat in coco["categories"]:
            categories.setdefault(cat["id"], set()).add(cat["name"])
    if not images:
        raise DatasetFormatError(f"{path}: ответы в формате COCO results требуют COCO-разметку")
    result = Annotations()
    for n, item in enumerate(data):
        where = f"{path}: ответ {n + 1}"
        try:
            file = item.get("file_name") or _only(images[item["image_id"]], "картинка", where)
            symbol = _only(categories[item["category_id"]], "категория", where)
        except KeyError as exc:
            raise DatasetFormatError(f"{where}: в разметке нет номера {exc}") from exc
        result.instances.append(_instance(file, "1", {**item, "symbol": symbol}, where))
    return result


def merge(parts: list[Annotations]) -> Annotations:
    result = Annotations()
    for part in parts:
        result.instances += part.instances
        result.symbol_names.update(part.symbol_names)
        result.times.update(part.times)
        result.detector = result.detector or part.detector
    return result


def _read(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetFormatError(f"{path}: {exc}") from exc


def _parse(data, source: str) -> Annotations:
    return from_coco(data, source) if _is_coco(data) else from_dict(data, source)


def _is_coco(data) -> bool:
    return isinstance(data, dict) and {"images", "annotations", "categories"} <= data.keys()


def _only(values: set[str], what: str, where: str) -> str:
    if len(values) != 1:
        raise DatasetFormatError(
            f"{where}: {what} с этим номером неоднозначна ({', '.join(sorted(values))}); "
            "укажите в ответе file_name или оценивайте по одному файлу разметки"
        )
    return next(iter(values))


def from_coco(data: dict, source: str = "<coco>") -> Annotations:
    """COCO-разметка → внутренний вид. Картинка — чертёж из одного листа."""
    categories = {c["id"]: c["name"] for c in data["categories"]}
    images = {i["id"]: i["file_name"] for i in data["images"]}
    result = Annotations()
    for n, ann in enumerate(data["annotations"]):
        where = f"{source}: аннотация {ann.get('id', n + 1)}"
        try:
            file, symbol = images[ann["image_id"]], categories[ann["category_id"]]
        except KeyError as exc:
            raise DatasetFormatError(f"{where}: нет картинки или категории {exc}") from exc
        attributes = ann.get("attributes") or {}
        item = {
            "symbol": symbol,
            "bbox": ann.get("bbox"),
            "angle": attributes.get("rotation", 0.0),
            "legend": symbol in LEGEND_CATEGORIES,
        }
        result.instances.append(_instance(file, "1", item, where))
    return result


def from_dict(data: dict, source: str = "<dict>") -> Annotations:
    if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
        raise DatasetFormatError(f"{source}: нет раздела «files»")
    result = Annotations(
        symbol_names=dict(data.get("symbols") or {}),
        detector=data.get("detector"),
    )
    for file, file_data in data["files"].items():
        if "time_s" in file_data:
            result.times[file] = float(file_data["time_s"])
        for page_key, items in (file_data.get("pages") or {}).items():
            for n, item in enumerate(items):
                where = f"{source}: {file}, лист {page_key}, экземпляр {n + 1}"
                result.instances.append(_instance(file, page_key, item, where))
    return result


def _instance(file: str, page_key: str, item: dict, where: str) -> Instance:
    try:
        x, y, w, h = (float(v) for v in item["bbox"])
        if w <= 0 or h <= 0:
            raise ValueError("у рамки нулевая ширина или высота")
        return Instance(
            file=file,
            page=int(page_key),
            symbol=str(item["symbol"]),
            bbox=(x, y, w, h),
            score=float(item.get("score", 1.0)),
            angle=float(item.get("angle", 0.0)),
            legend=bool(item.get("legend", False)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetFormatError(f"{where}: {exc}") from exc


def to_dict(annotations: Annotations) -> dict:
    """Обратное преобразование — для сохранения ответов алгоритма."""
    files: dict[str, dict] = {}
    for file in annotations.files:
        files[file] = {"pages": {}}
        if file in annotations.times:
            files[file]["time_s"] = round(annotations.times[file], 6)
    for i in annotations.instances:
        pages = files[i.file]["pages"]
        pages.setdefault(str(i.page), []).append(
            {
                "symbol": i.symbol,
                "bbox": [round(v, 4) for v in i.bbox],
                "score": round(i.score, 6),
                "angle": i.angle,
            }
        )
    result: dict = {"files": files}
    if annotations.detector:
        result = {"detector": annotations.detector, **result}
    return result


def save(annotations: Annotations, path: Path) -> None:
    Path(path).write_text(
        json.dumps(to_dict(annotations), ensure_ascii=False, indent=2), encoding="utf-8"
    )
