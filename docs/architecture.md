# Архитектура сервиса

Веб-сервис нужен, чтобы удобно готовить образцы, запускать алгоритм на реальных чертежах и смотреть
результат глазами. Сам алгоритм подключается к нему как отдельный класс и от интерфейса не зависит.

## Как идут данные

```
загрузка файла ──► converters.py ──► постраничные SVG (+ PDF для векторного анализа, миниатюры PNG)
                                                │
выделение рамкой в просмотрщике ──► образец (Symbol) в библиотеке
                                                │
запуск поиска ──► Detector.detect(страница, образцы, параметры) ──► Detection[] ──► SearchRun
                                                                                    │
                                                    просмотрщик рисует найденные рамки
```

- `backend/app/converters.py` — приводит PDF, SVG, DXF и DWG к постраничным SVG.
- `backend/app/detection/` — алгоритмы поиска (интерфейс `Detector`).
- `backend/app/storage.py` — хранилище на файлах в `data/`, формат описан в docstring модуля.
- `backend/app/routers/` — API. Полное описание — Swagger на http://localhost:8000/docs.
- `frontend/src/components/DrawingViewer.tsx` — просмотрщик: масштаб, рука, выделение, рамки результатов.

## Координаты и форматы

Каждый чертёж приводится к постраничным SVG — именно их показывает браузер. Все рамки (выделение,
образцы, результаты поиска) задаются как `{x, y, w, h}` в **единицах страницы**: начало координат —
левый верхний угол листа, ось Y направлена вниз.

| Формат | Как обрабатывается | Единицы страницы |
|---|---|---|
| PDF | каждая страница → SVG (PyMuPDF) | пункты PDF, как `page.rect` в PyMuPDF |
| SVG | показывается как есть, в корень дописываются недостающие `viewBox`/`width`/`height` | единицы `viewBox` минус его начало |
| DXF | пространство модели → PDF (ezdxf, 1:1, поля 5 мм) → как PDF | пункты; промежуточный PDF — `converted.pdf` |
| DWG | → DXF через ODA File Converter или LibreDWG (`dwg2dxf`) → как DXF | как у DXF |

Для DWG нужен внешний конвертер. Если его нет, чертёж сохраняется со статусом «ошибка»;
после установки конвертера нажмите «Повторить обработку».

## Как подключить алгоритм

Алгоритм — это класс-наследник `Detector` (`backend/app/detection/base.py`). Ему передаются
страница, на которой ищем (`PageRef`), и список образцов (`Template` — рамка на странице какого-то чертежа):

```python
# backend/app/detection/my_detector.py
from collections.abc import Sequence

import pymupdf

from ..models import BBox, Detection, SearchParams
from .base import Detector, PageRef, Template


class MyDetector(Detector):
    name = "my"
    description = "Сопоставление векторных примитивов + RANSAC"

    def detect(self, target: PageRef, templates: Sequence[Template], params: SearchParams) -> list[Detection]:
        with pymupdf.open(target.pdf_path) as doc:      # для PDF, DXF и DWG
            primitives = doc[target.index - 1].get_drawings()
        found = []
        for template in templates:
            ...  # template.page — откуда взят образец, template.bbox — где он там
            found.append(Detection(page=target.index, bbox=BBox(x=..., y=..., w=..., h=...),
                                   score=0.93, symbol_id=template.symbol_id, angle=90))
        return found
```

Дальше добавьте класс в `DETECTORS` (`backend/app/detection/__init__.py`) и запустите бэкенд с
`SYMSEARCH_DETECTOR=my`. Интерфейс подхватит результаты сам.

Полезно знать:

- `PageRef` даёт три представления страницы: `svg_path` (то, что видит пользователь), `pdf_path`
  (вектор в тех же координатах; для SVG — `None`) и `source_path` (исходный файл);
- растр нужного разрешения: `page.get_pixmap(dpi=...)`, координаты пересчитываются умножением на `dpi / 72`;
- если у страницы PDF задан поворот (`page.rotation != 0`), координаты `get_drawings()` переводятся
  через `page.rotation_matrix`;
- `params` — порог сходства и флаги «повороты / зеркальные / масштаб» из интерфейса;
- поиск выполняется синхронно в запросе. Если алгоритм будет работать дольше нескольких секунд,
  его стоит перенести в фоновую задачу;
- алгоритм на Julia можно вызывать из `detect` через [juliacall](https://github.com/JuliaPy/PythonCall.jl)
  или как отдельный процесс.

## API

| Метод | Адрес | Назначение |
|---|---|---|
| GET | `/api/health`, `/api/stats` | состояние сервиса, счётчики |
| GET, POST | `/api/drawings` | список чертежей, загрузка (multipart, поле `file`) |
| GET, DELETE | `/api/drawings/{id}` | чертёж, удаление |
| POST | `/api/drawings/{id}/reprocess` | обработать заново |
| GET | `/api/drawings/{id}/original` | исходный файл |
| GET | `/api/drawings/{id}/pages/{n}/image`, `…/thumb` | лист в SVG, миниатюра PNG |
| GET, POST | `/api/symbols` | библиотека, добавить образец |
| GET, PATCH, DELETE | `/api/symbols/{id}` | образец |
| GET | `/api/symbols/{id}/preview` | превью образца |
| POST | `/api/search` | запустить поиск |
| GET | `/api/runs`, `/api/runs/{id}` | история запусков, результаты запуска |

Типы для фронтенда (`frontend/src/api/types.ts`) пока написаны вручную. Меняя схемы на бэкенде,
обновляйте и их.
