# SymSpot MVP — план работ для Claude Code

Поиск обозначений на чертеже по эталонам, размеченным на этом же чертеже.
Результат — раскрашенные линии символов (а не рамки) + JSON.
В первую очередь векторный вход (PDF, SVG), растр (PNG/JPG) упрощённой веткой.

**Роли моделей**
- **Opus 5.5 (главная сессия)** — оркестратор и архитектор: контракты, алгоритмическое ядро
  (голосование, проверка, разрешение конфликтов), интеграция, ревью каждого диффа, отладка на данных.
- **Sonnet 5.5 (сабагенты)** — реализация модулей по готовой спецификации, тесты, рендеринг,
  CLI, генератор синтетики, документация. Агенты описаны в `.claude/agents/`.

---

## 0. Правила делегирования (для Opus)

1. Сначала Opus сам пишет контракты (`symspot/core/types.py`, сигнатуры модулей со
   docstring и `NotImplementedError`). Пока контракты не зафиксированы, никого не запускать.
2. Задача сабагенту = бриф по шаблону ниже. Без брифа не делегировать.
3. Независимые задачи одной фазы запускать параллельно (до 3 одновременно), каждую с
   `isolation: "worktree"`, чтобы агенты не мешали друг другу.
4. После возврата агента Opus обязательно:
   - читает дифф целиком;
   - запускает `uv run pytest -q` и `uv run ruff check .`;
   - проверяет, что не тронуты чужие файлы и `core/types.py`;
   - мержит или отправляет агенту замечания через SendMessage (тот же агент, контекст сохранён).
5. Алгоритмическое ядро (фаза 3: голосование, проверка, соревнование, разрешение) Opus пишет сам.
   Sonnet параллельно пишет тесты к нему **по спецификации**, не видя реализации.
6. Если агент дважды не прошёл приёмку по одной задаче — Opus делает её сам.
7. Каждая фаза заканчивается гейтом: все тесты зелёные + критерии фазы выполнены + коммит.

### Шаблон брифа

```
Агент: sonnet-implementer | sonnet-tester | sonnet-viz
Задача: <одно предложение>
Контекст: <зачем, где в пайплайне, ссылка на раздел PLAN.md>
Файлы, которые можно создавать/менять: <список>
Файлы, которые НЕЛЬЗЯ менять: symspot/core/types.py, всё вне списка
Интерфейс (скопирован из types.py / модуля): <сигнатуры>
Алгоритм / требования: <по пунктам>
Приёмка:
  - команды: uv run pytest tests/<...> -q ; uv run ruff check <...>
  - тесты, которые должны пройти: <список или "напиши сам, минимум N">
  - числовые критерии: <если есть>
Верни: краткий отчёт (что сделано, допущения, что не получилось), без пересказа кода.
```

---

## 1. Объём MVP

**Входит**
- Вход: PDF (векторный), SVG, PNG/JPG.
- Эталоны: COCO-экспорт из CVAT — рамки с label `legend_item` и атрибутом `legend_id`
  (как в имеющемся `instances_Train.json`), либо простой JSON `[{legend_id, bbox}]`.
- Векторный конвейер, шаги 0–8 из схемы (ввод, нормализация, эталоны, точные копии,
  кандидаты, проверка, раскраска, глобальное разрешение, расширение запроса).
- Растровый конвейер: предобработка, chamfer-поиск, та же проверка, маски пикселей.
- Выход: SVG/PDF/PNG с цветным слоем, `result.json`, HTML-отчёт.
- Генератор синтетических чертежей с разметкой (для тестов и метрик).
- Метрики: precision/recall/F1 по экземплярам и по классам, матрица путаницы.
- CLI + простое демо (Streamlit).

**Не входит (после MVP)**
- Обучаемая ветка (SymPoint под запрос), DINOv2, VLM-арбитр.
- DXF `INSERT` и PDF XObject как отдельный быстрый путь (оставить хук).
- Фото с сильной перспективой, рукописные чертежи.

---

## 2. Стек и структура

Python 3.11, `uv`. Зависимости: `numpy scipy shapely>=2 pymupdf svgelements opencv-python-headless
scikit-image pydantic typer rich pytest ruff streamlit cairosvg`.

```
symspot/
  core/        types.py, config.py, geometry_utils.py (Umeyama, трансформы, сэмплинг)
  io/          pdf_reader.py, svg_reader.py, raster_reader.py, annotations.py
  normalize/   bezier.py, merge.py, planarize.py, index.py
  query/       extract.py, descriptors.py, discriminative.py
  exact/       hashing.py
  search/      pairs.py, voting.py
  verify/      icp.py, coverage.py, compete.py
  paint/       assign.py
  resolve/     global_resolve.py
  expand/      query_expansion.py
  raster/      preprocess.py, chamfer.py, verify_raster.py, mask.py
  render/      svg_overlay.py, pdf_overlay.py, png_overlay.py, html_report.py
  synth/       symbols.py, plan_generator.py, export.py
  eval/        metrics.py
  pipeline.py, cli.py
app/demo.py
tests/  (зеркалит пакет) + tests/e2e/
data/   synth/ (генерируется), real/ (руками, в .gitignore)
```

---

## 3. Контракты (пишет Opus в фазе 0)

```python
# symspot/core/types.py  (ориентир, Opus уточняет)
Point = tuple[float, float]

@dataclass(frozen=True)
class Segment:   id: int; p0: Point; p1: Point; parent_id: int; style: Style
@dataclass(frozen=True)
class Arc:       id: int; center: Point; r: float; t0: float; t1: float; parent_id: int; style: Style
                 # t0..t1 в радианах, против часовой; полный круг: t1 - t0 = 2π
@dataclass(frozen=True)
class TextPrim:  id: int; text: str; bbox: tuple[float,float,float,float]; parent_id: int
Primitive = Segment | Arc | TextPrim

@dataclass
class Drawing:
    kind: Literal["vector", "raster"]
    width: float; height: float; units_per_px: float
    primitives: list[Primitive]          # для vector
    parents: dict[int, ParentPath]       # исходный путь файла (для раскраски и отладки)
    image: np.ndarray | None             # для raster (grayscale uint8)
    source_path: str

@dataclass
class QuerySpec:     legend_id: str; bbox: tuple[float,float,float,float]; label: str | None
@dataclass
class Query:         legend_id: str; prims: list[Primitive]; frame: Similarity  # нормализованный эталон
                     samples: np.ndarray; tangents: np.ndarray; weights: np.ndarray
@dataclass
class Similarity:    s: float; theta: float; tx: float; ty: float; mirror: bool   # + apply(), inverse(), compose()
@dataclass
class Hypothesis:    legend_id: str; T: Similarity; votes: int; source: Literal["exact","vote","expand"]
@dataclass
class Instance:      legend_id: str; T: Similarity; score: float; coverage: float
                     piece_ids: list[int]; parent_ids: list[int]
                     mask_rle: dict | None; bbox: tuple; status: Literal["ok","ambiguous"]
@dataclass
class Result:        drawing: str; instances: list[Instance]; timings: dict[str,float]; config: dict
```

`config.py`: pydantic-модель со всеми порогами (раздел 5), читается из YAML/CLI.

---

## 4. Фазы

### Фаза 0 — каркас и контракты · **Opus, сам**
- `uv init`, pyproject, ruff, pytest, `.gitignore`, CLAUDE.md уже лежит.
- `core/types.py`, `core/config.py`, заглушки всех модулей с сигнатурами и docstring.
- `core/geometry_utils.py`: `Similarity`, `umeyama(src, dst, allow_mirror)`, сэмплинг точек и
  касательных по примитивам с шагом `step`. Это используется всеми — пишет Opus.
- Тесты на `geometry_utils` (Opus или sonnet-tester).
- **Гейт:** `pytest` зелёный, все модули импортируются.

### Фаза 1 — периферия · **Sonnet, параллельно (3 агента)**

**T1. Генератор синтетики** · `sonnet-implementer` · `synth/`
- Библиотека ≥12 символов как списки примитивов, включая **почти одинаковые группы**:
  duplex / GFCI (текст) / 220 (текст); выключатель S / S₃ / S_D; встроенный светильник ⊘ /
  накладной ○ с крестом; вентилятор; детектор SD/CO (текст в круге).
- План: внешние и внутренние стены, двери с дугами, размерные линии с текстом, штриховки,
  случайный текст.
- Размещение экземпляров: поворот (90°-кратные и произвольные), масштаб ±15 %, зеркало;
  доля «сложных» случаев: стена через символ, символ у края штриховки, линии разбиты на
  2–3 отрезка, окружность как 4 кривые Безье.
- Экспорт: SVG, PDF (cairosvg), PNG 300 dpi + «грязный» PNG (шум, размытие, лёгкий поворот).
- GT: JSON с экземплярами (legend_id, трансформ, id исходных путей, bbox) + COCO в стиле CVAT,
  где для каждого класса **одна** рамка `legend_item` (эталон) и все рамки `object`.
- CLI: `python -m symspot.synth --n 20 --size large --seed 0 --out data/synth`.
- Приёмка: детерминизм по seed; размеры: small ~2k примитивов, large ~30k; тесты на GT.

**T2. Чтение векторов** · `sonnet-implementer` · `io/pdf_reader.py`, `io/svg_reader.py`
- PDF: `page.get_drawings()` → линии (`l`), кривые (`c`), прямоугольники (`re`), квадрики;
  `page.get_text("dict")` → `TextPrim` по span'ам. Учесть `rotation` страницы.
- SVG: `svgelements` с применением всех transform, `<text>` → `TextPrim`, `<use>` разворачивать.
- Кривые Безье пока не трогать — отдавать как `parent` с исходной геометрией + сэмплированной
  полилинией; фитинг дуг делается в фазе 2.
- Приёмка: на синтетике число путей и bbox совпадают с GT (±1e-3); тест на повёрнутой странице.

**T3. Аннотации и метрики** · `sonnet-implementer` · `io/annotations.py`, `eval/metrics.py`
- Чтение COCO из CVAT: `legend_item` → `QuerySpec`, `object` → GT (по `legend_id`);
  масштабирование координат, если размер изображения в COCO ≠ размеру растеризации PDF.
- Метрики: сопоставление предсказаний и GT по IoU рамок ≥0.5 (Hungarian), P/R/F1
  общие и по классам, матрица путаницы, отдельно доля ошибок внутри групп похожих символов.
- Приёмка: юнит-тесты на ручных примерах; чтение реального `instances_Train.json`.

**T4. Рендеринг** · `sonnet-viz` · `render/` (можно начать после T2, интерфейс `Result` уже есть)
- SVG: исходный чертёж серым (opacity 0.35) + цветной слой с точной геометрией назначенных кусков.
- PDF: копия страницы + рисование кусков через `page.new_shape()` поверх.
- PNG: для растра — окраска пикселей маски; для вектора — растеризация SVG-оверлея.
- HTML-отчёт: картинка с zoom/pan, таблица экземпляров, фильтр по классу, клик → подсветка,
  отдельный блок «ambiguous».
- Палитра: различимые цвета, стабильные по `legend_id`.

**Гейт фазы 1 (Opus):** ревью всех четырёх, сгенерировать `data/synth` (20 small + 3 large),
прогнать чтение всех SVG/PDF без ошибок.

### Фаза 2 — нормализация и точные копии · **Sonnet + ревью Opus**

**T5. Нормализация** · `sonnet-implementer` · `normalize/`, тесты пишет `sonnet-tester` параллельно
- `bezier.py`: Безье → полилиния (допуск `flatten_tol`); распознавание окружности из 4 кривых
  и дуг: фит окружности (алгебраический + уточнение), принять, если остаток < `arc_fit_tol`.
- `merge.py`: склейка коллинеарных соприкасающихся отрезков (угол < 1°, зазор < `merge_gap`),
  удаление дублей и нулевых.
- `planarize.py`: разрез всех отрезков и дуг в точках пересечения (STRtree для кандидатов,
  аналитические пересечения отрезок–отрезок, отрезок–дуга, дуга–дуга). Каждый кусок хранит
  `parent_id`. Сложность ≈ O(n log n + k).
- `index.py`: обёртка STRtree + cKDTree по сэмплам.
- Приёмка: синтетика large (~30k) нормализуется < 10 с; тест «стена через символ» даёт кусок
  внутри круга отдельным примитивом; тест «окружность из 4 Безье» → один `Arc`.

**T6. Точные копии** · `sonnet-implementer` · `exact/hashing.py`
- Канонический хеш набора примитивов эталона (сдвиг в начало, округление до `exact_tol`,
  сортировка). Поиск: якорь = самый редкий тип/размер примитива эталона, окна вокруг каждого
  такого примитива на чертеже, хеш окна.
- Приёмка: на синтетике находит все неповёрнутые немасштабированные копии без ложных.

**Гейт фазы 2 (Opus):** вручную проверить планаризацию на 2–3 картинках (рендер кусков случайными цветами).

### Фаза 3 — алгоритмическое ядро · **Opus пишет, Sonnet тестирует**

Opus реализует сам, в этом порядке. Параллельно `sonnet-tester` пишет тесты по этому разделу.

**3.1 Извлечение эталона** (`query/extract.py`)
- Куски внутри рамки берутся, если **родитель** (до планаризации) лежит в рамке ≥ `query_parent_inside` (0.8).
  Так отбрасывается стена или размерная линия, проходящая через рамку.
- Нормализация: центр рамки → 0, масштаб = диагональ рамки → 1.
- Сэмплы + касательные с шагом `sample_step` (доля диагонали, 0.02).

**3.2 Дескрипторы пар** (`query/descriptors.py`, `search/pairs.py`)
- Якорь примитива: отрезок → середина, направление (неориентированное), длина; дуга →
  центр, радиус, угловой размах; текст → центр bbox, строка.
- Ключ пары (i, j), инвариантный к подобию: типы; `log(size_i/size_j)`; если якоря не
  совпадают — `log(size_i/d)`, углы направлений относительно вектора `d`; если совпадают
  (концентричные, как круг и диагональ ⊘) — отдельный тип ключа: отношение размеров и угол между
  направлениями. Квантование, хеширование.
- Для чертежа: пары только среди кусков, чьи якоря ближе `pair_radius` = 1.5 × max диагональ
  эталонов × max масштаб. Таблица строится один раз на чертёж.

**3.3 Голосование** (`search/voting.py`)
- Совпавший ключ → преобразование подобия из пары (2 варианта для неориентированных отрезков,
  ×2 при `allow_mirror`). Голос в аккумулятор: сдвиг (шаг = 0.25 диагонали эталона), угол (5°),
  log-масштаб (0.05), зеркало.
- Отбрасывать масштабы вне `[scale_min, scale_max]` (по умолчанию 0.7–1.4 от эталона).
- Гипотеза принимается, если голоса пришли от ≥ `min_distinct_pairs` разных пар эталона;
  нормировка на локальную плотность кусков (штриховка не должна побеждать).
- NMS по сдвигу, ограничение `max_hypotheses_per_query`.

**3.4 Проверка** (`verify/icp.py`, `verify/coverage.py`)
- ICP: сэмплы эталона → ближайшие сэмплы чертежа (cKDTree), Umeyama, ≤ 15 итераций,
  матчинг только при согласованном направлении касательных (< 20°).
- Покрытие в одну сторону = взвешенная доля сэмплов эталона, у которых есть сэмпл чертежа
  ближе `cover_tol` (0.03 диагонали + половина толщины линии) с согласованным направлением.
- Покрытие по каждому примитиву эталона; если какой-то примитив < `min_prim_coverage` (0.6) —
  отказ («нет диагонали»).
- Текст: если в эталоне есть `TextPrim`, на трансформированном месте должен быть текст с той же
  нормализованной строкой (для растра — флаг «есть чернила в зоне текста», OCR вне MVP).
- `score = weighted_coverage − λ · extra_ink_ratio_inside` (λ маленький, лишняя тушь почти не штрафуется).

**3.5 Отличающие части и соревнование** (`query/discriminative.py`, `verify/compete.py`)
- Для каждой пары эталонов схожего размера: совместить ICP; если похожи (chamfer < порога) —
  сэмплам одного, не покрытым другим, вес ×`discriminative_weight` (3).
- Все принятые экземпляры разных классов с IoU рамок > 0.5 сравниваются; победитель —
  по score с отрывом ≥ `compete_margin` (0.03), иначе `status="ambiguous"` (оба в отчёт).

**3.6 Назначение кусков** (`paint/assign.py`)
- Кусок чертежа принадлежит экземпляру, если ≥ `assign_inside` (0.7) его сэмплов ближе
  `cover_tol` к трансформированному эталону и направления согласованы.

**3.7 Глобальное разрешение** (`resolve/global_resolve.py`)
- Эталоны обрабатываются по убыванию сложности (число сэмплов).
- Жадно: экземпляр забирает куски; следующий, у которого > 50 % кусков уже заняты, отклоняется.
- Дубли одного класса: IoU множеств кусков > 0.5 → один экземпляр.

**3.8 Расширение запроса** (`expand/query_expansion.py`)
- Экземпляры со score ≥ 0.9, заметно отличающиеся от эталона (остаток ICP > порога), становятся
  доп. эталонами своего класса; один повтор поиска 3.3–3.7 только по незанятым кускам.

**Гейт фазы 3:** все тесты `sonnet-tester` зелёные; ручная проверка на 3 small синтетиках.

### Фаза 4 — сборка векторного пайплайна · **Opus + Sonnet**
- Opus: `pipeline.py` (порядок шагов, тайминги, логирование, кэш нормализации).
- `sonnet-implementer`: `cli.py` (typer):
  `symspot run <drawing> --queries <coco|json> --out <dir> [--config cfg.yaml] [--page N]`,
  `symspot eval <result.json> <gt>`, `symspot synth ...`.
- `sonnet-tester`: e2e-тесты на синтетике.
- **Критерии (синтетика small, вектор):** F1 ≥ 0.92; ошибки внутри групп похожих символов ≤ 5 %;
  кейс «стена через символ» — окрашен только кусок внутри; время small < 5 с.

### Фаза 5 — растровая ветка · **Sonnet по спецификации Opus, ядро проверки Opus**
- `sonnet-implementer`: `raster/preprocess.py` — серое, Sauvola, удаление мелкого мусора,
  опционально deskew (Hough), скелет, карта расстояний, градиент направления.
- `sonnet-implementer`: `raster/chamfer.py` — эталон = скелет кропа из рамки; поиск по углам
  (шаг `raster_angle_step`, 10°) и масштабам (`scale_min..scale_max`, шаг 0.05) через
  свёртку карты расстояний с бинарной маской эталона (`cv2.filter2D`); тайлы с перекрытием
  ≥ 2 × размер эталона; NMS; топ-K гипотез.
- Opus: `raster/verify_raster.py` — те же ICP/покрытие/соревнование на точках скелета
  (переиспользовать фазу 3), `raster/mask.py` — пиксели туши ближе допуска к трансформированному
  эталону и с согласованным направлением градиента → RLE.
- **Критерии (синтетика PNG):** F1 ≥ 0.85 на чистом PNG, ≥ 0.7 на «грязном».

### Фаза 6 — производительность · **Opus профилирует, Sonnet оптимизирует точечно**
- Бенчмарк `tests/bench`: large (~30k примитивов, 25 эталонов) и растр 7200×4800.
- Цель: large вектор < 60 с, растр 7200×4800 с 20 эталонами < 3 мин на CPU.
- Средства: multiprocessing по эталонам/тайлам, numpy-векторизация пар, кэш таблицы пар,
  отсечение пустых областей по плотности.
- Opus: `cProfile` → список горячих мест → бриф на каждое `sonnet-implementer`.

### Фаза 7 — демо, документация, реальные данные · **Sonnet + финальное ревью Opus**
- `sonnet-viz`: `app/demo.py` (Streamlit): загрузка чертежа + COCO, запуск, просмотр оверлея,
  таблица, скачать SVG/PDF/JSON.
- `sonnet-implementer`: README (установка, примеры, формат входа/выхода, ограничения).
- Opus: прогон на реальном чертеже (`data/real/`, JPG 7200×4800 + `instances_Train.json`),
  разбор ошибок, подстройка конфигурации, итоговый отчёт `docs/REPORT.md` (метрики, примеры,
  известные проблемы, что делать дальше).

---

## 5. Конфиг по умолчанию (`configs/default.yaml`)

```yaml
sample_step: 0.02          # доля диагонали эталона
flatten_tol: 0.002
arc_fit_tol: 0.01
merge_gap: 0.005
query_parent_inside: 0.8
pair_radius_factor: 1.5
scale_min: 0.7
scale_max: 1.4
allow_mirror: true
angle_bin_deg: 5
scale_bin_log: 0.05
min_distinct_pairs: 3
max_hypotheses_per_query: 2000
cover_tol: 0.03
min_prim_coverage: 0.6
accept_score: 0.8
discriminative_weight: 3.0
compete_margin: 0.03
assign_inside: 0.7
expand_score: 0.9
raster_angle_step: 10
raster_tile: 2048
workers: auto
```

---

## 6. Порядок запуска и параллельность

```
Ф0  Opus ───────────────┐
Ф1  T1 synth ─┐         │ параллельно, worktree
    T2 readers┼─► T4 viz│
    T3 ann+metrics      │
Ф2  T5 normalize (+tester) ║ T6 exact
Ф3  Opus ядро  ║  sonnet-tester пишет тесты по спецификации
Ф4  Opus pipeline ║ CLI ║ e2e
Ф5  preprocess ║ chamfer  →  Opus verify/mask
Ф6  Opus профиль → точечные оптимизации
Ф7  demo ║ README  →  Opus реальные данные + REPORT
```

## 7. Определение готовности MVP
- `symspot run` работает на PDF, SVG, PNG/JPG с COCO из CVAT.
- Выход: оверлей (SVG/PDF/PNG), `result.json`, `report.html`.
- Метрики фаз 4–5 выполнены, бенчмарк фазы 6 выполнен.
- `pytest` зелёный, `ruff` чистый, README позволяет запустить с нуля.
- `docs/REPORT.md` с результатами на реальном чертеже.
