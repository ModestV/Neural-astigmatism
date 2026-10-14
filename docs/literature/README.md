# Обзор литературы

Здесь собирается обзор современных подходов к поиску условных обозначений на чертежах
(в англоязычной литературе — *symbol spotting*). Это первый критерий оценки проекта: обзор нужен,
чтобы обосновать выбранный подход.

Удобно вести по одному файлу на тему, например `classical.md`, `deep-learning.md`, `vlm.md`,
и сводку в этом README. Для каждой работы: что делает, на каких данных проверена, какие метрики,
чем полезна нам.

## С чего начать

Список для старта, его нужно проверить и дополнить.

**Классические методы**

- Fischler M., Bolles R. *Random Sample Consensus.* Communications of the ACM, 1981 — RANSAC.
- Ballard D. *Generalizing the Hough transform to detect arbitrary shapes.* Pattern Recognition, 1981.
- Работы J. Lladós, M. Rusiñol и соавторов по symbol spotting на векторных и растровых чертежах.

**Глубокое обучение на чертежах**

- Rezvanifar A., Cote M., Albu A. *Symbol spotting on digital architectural floor plans using a deep
  learning-based framework.* CVPR Workshops 2020.
- Fan Z. et al. *FloorPlanCAD: A Large-Scale CAD Drawing Dataset for Panoptic Symbol Spotting.* ICCV 2021.
- Fan Z. et al. *CADTransformer: Panoptic Symbol Spotting Transformer for CAD Drawings.* CVPR 2022.
- Zheng Z. et al. *GAT-CADNet: Graph Attention Network for Panoptic Symbol Spotting in CAD Drawings.* CVPR 2022.
- Liu W. et al. *Symbol as Points: Panoptic Symbol Spotting via Point-based Representation.* ICLR 2024.

**Языково-визуальные модели (VLM).** Нужно проверить, как современные VLM справляются с поиском
мелких обозначений на больших листах, и решить, годятся ли они как «оракул» для проверки кандидатов.

## Аналоги среди продуктов

- **Bluebeam Revu, функция VisualSearch** — пользователь обводит символ на PDF, программа находит похожие.
  По тому же принципу устроен просмотр в нашем сервисе.
- **AutoCAD, Smart Blocks: Search and Convert** — находит повторяющуюся геометрию в DWG и превращает её в блоки.
- **Togal.AI, Kreo и другие сервисы для смет** — автоматически считают обозначения на строительных
  чертежах с помощью машинного обучения.
