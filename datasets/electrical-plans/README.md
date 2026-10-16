# Планы электрооборудования (COCO)

Размеченная командой часть датасета: растровые чертежи (JPEG) планов электрооборудования жилых
домов с американскими обозначениями. Разметка — рамки в формате COCO, выгрузка из CVAT.

**Картинок в репозитории нет**, только разметка. Где лежат изображения и откуда взяты чертежи —
TODO: дописать источник и лицензию.

## Состав

| Файл разметки | Чертёж | Размер, px | Экземпляров | Замечание |
|---|---|---|---|---|
| `instances_Train.json` | `01_riga_lighting000000000.jpeg-1.jpg` | 3308 × 2339 | 13 | размечены только светильники (`light`) |
| `instances_Train1.json` | `02_renee_house000000000.jpeg-1.jpg` | 7200 × 4800 | 132 | 16 классов |
| `instances_Train3.json` | `03_washoe_house000000000.jpeg-1.jpg` | 7200 × 4800 | 119 | 10 классов |

Всего 264 экземпляра, 19 из 24 классов встречаются хотя бы раз.

## Классы

`simplex_socket`, `light`, `switch`, `tv_outlet`, `garage_door_button`, `smoke_detector`,
`duplex_socket`, `gfci_socket`, `socket_wp`, `recessed_light`, `vapor_proof_recessed_light`,
`pendant_light`, `wall_light`, `exhaust_fan`, `ceiling_fan`, `garage_door_opener`,
`electrical_panel`, `fluorescent_light`, `equipment_connection`, `surface_ceiling_light`,
`three_way_switch`, `smoke_co_detector`, `object`, `legend_item`.

`legend_item` — экземпляр в таблице условных обозначений: скрипт метрик по умолчанию его
не оценивает. В текущих файлах таких экземпляров нет.

## Что учесть

- Координаты — в пикселях картинки, а не в единицах листа PDF.
- Атрибут CVAT `rotation` везде 0, хотя часть обозначений повёрнута (у рамки поменяны ширина
  и высота). Угол поворота из разметки брать нельзя.
- `instances_Train.json` размечен не полностью: если алгоритм найдёт на этом чертеже что-то,
  кроме светильников, это засчитается как ложное срабатывание.
- В каждом файле картинка имеет номер 1. Ответы в формате COCO results с `image_id` поэтому
  однозначны только при оценке по одному файлу; для нескольких файлов добавляйте в ответ `file_name`.

## Как посчитать метрики

```sh
backend/.venv/bin/python scripts/evaluate.py --gt datasets/electrical-plans/annotations/*.json --pred results.json
backend/.venv/bin/python scripts/evaluate.py --gt datasets/electrical-plans/annotations/*.json --simulate
```

`--simulate` подставляет вместо алгоритма разметку со случайным шумом — только для проверки
скрипта, это не результат алгоритма.
