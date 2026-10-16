"""Разметка и ответы алгоритма в едином внутреннем виде.

Метрики (metrics.py) работают только с этими классами. Если формат датасета изменится,
достаточно дописать здесь новый загрузчик — сами метрики трогать не придётся.

Поддерживаемые форматы (оба — JSON одинаковой формы, см. samples/ground_truth.json):

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

Рамки — в единицах страницы, ось Y вниз (как во всём проекте).
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


def load(path: Path) -> Annotations:
    """Читает разметку или ответы алгоритма в формате, описанном в docstring модуля."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetFormatError(f"{path}: {exc}") from exc
    return from_dict(data, source=str(path))


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
