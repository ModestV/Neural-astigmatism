# Обзор подходов к поиску условных обозначений на чертежах

Обзор подготовлен для сценария SymSpot: растровые электрические и пожарные планы 7200x4800 и 8400x6000 px, символы 35-80 px, один образец на класс в легенде отдельного листа, 20-40 классов с длинным хвостом, 4 размеченных листа (около 325 объектов), новые комплекты со своими символами.

Пометки достоверности у каждой работы: [А] прочитана аннотация на странице источника в ходе подготовки обзора; [П] подтверждено только библиографической записью или пересказом поисковой выдачи, страница работы напрямую не открывалась; [Н] классическая работа, приведена по памяти составителя, в ходе подготовки не сверялась. Числа приводятся только если они были видны в проверенном источнике. Оценки применимости к проекту являются мнением составителя и экспериментально не проверялись.

## Краткое резюме

Известные подходы делятся на пять семейств. Классические методы (сопоставление с шаблоном, chamfer, обобщенное преобразование Хафа, графовые и векторные сигнатуры) работают по одному образцу, но чувствительны к шуму, соседним линиям и мелким отличиям. Панорамное распознавание на векторных CAD-чертежах (FloorPlanCAD, CADTransformer, GAT-CADNet, SymPoint) дает лучшие числа, однако требует векторного входа и закрытого набора из 30 классов с тысячами размеченных листов. Детекторы семейства YOLO и Faster R-CNN на P&ID, однолинейных схемах и планах показывают хорошие результаты при фиксированном словаре классов и сотнях-тысячах экземпляров, а редкие классы лечат синтетикой и GAN. Few-shot и visual prompting (OWL-ViT/OWLv2, T-Rex2, DINO-X, FS-DETR, DE-ViT) решают задачу «найти похожее по образцу», но обучены на естественных изображениях, а бенчмарки межобластного переноса показывают заметное падение качества; применение этих моделей к чертежам в проверенных источниках не обнаружено. Для чертежей есть работы, прямо использующие легенду или один образец на класс (OSSR-PID, Sarkar и др. 2022, Jamieson и др. 2024), но воспроизводимых результатов на электрических планах среди проверенного материала нет. Для нашего случая из этого следует гибридная схема: генерация кандидатов по образцам легенды, затем верификация обучаемым признаковым описанием, а YOLO разумен как класс-независимый детектор «символ или нет» и как детектор, обучаемый на синтетике из легенды конкретного комплекта, но не как единая закрытая модель на 4 размеченных листах.

## 1. Классические методы и датасет SESYD

**1.1 Rusiñol, Lladós. Symbol Spotting in Technical Drawings Using Vectorial Signatures. GREC 2005, LNCS 3926, 2006.** https://doi.org/10.1007/11767978_4 [П]. Чертеж векторизуется и представляется атрибутированным графом из геометрических и структурных отношений между отрезками, запрос ищется по сигнатурам в многомасштабном описании. Применимость: идея индексации примитивов полезна, но требует чистой векторизации; для растра 7200x4800 с текстом и стенами хрупка.

**1.2 Dutta, Lladós, Pal. Symbol Spotting in Line Drawings Through Graph Paths Hashing. ICDAR 2011; расширенная версия: Pattern Recognition 46(3), 2013.** Запись: https://openresearch.surrey.ac.uk/permalink/44SUR_INST/15d8lgh/alma991109888602346 [П]. Дескрипторы путей графа хешируются (locality sensitive hashing) для приближенного поиска, локализация делается пространственным голосованием. Метод запрос-по-образцу, но испытан на синтетических и простых линейных чертежах, масштабирование на листы с тысячами примитивов не показано в проверенном материале.

**1.3 Delalandre, Valveny, Pridmore, Karatzas. Generation of Synthetic Documents for Performance Evaluation of Symbol Recognition and Spotting Systems. IJDAR 13(3), 2010, с. 187-207.** https://hal.archives-ouvertes.fr/hal-01022594 [П]. Датасет SESYD: символы размещаются на заданных фонах по доменным ограничениям (архитектура, электроника), разметка получается автоматически; коллекции вышли примерно на 284 тыс. изображений. Применимость: методологический прецедент синтетики из набора образцов; сами изображения далеки от реальных планов (чистые, малого размера), поэтому SESYD годится для проверки кода, но не для оценки на наших листах.

**1.4 Rezvanifar. Analyzing symbols in architectural floor plans via traditional computer vision and deep learning approaches. PhD thesis, University of Victoria, 2021.** https://dspace.library.uvic.ca/items/3cc20c2c-8c51-4d7d-a7d9-77c08b9258dc [А]. Гибридный метод: векторизация элементов символа, граф близости, грубая локализация статистикой, уточнение пиксельной кросс-корреляцией; отдельно YOLO с обучением по тайлам. Заявлено превосходство над популярными методами на SESYD и реальных планах, численные метрики на странице не приведены. Применимость: ближайший по сценарию классический пример (архитектурные планы), показывает, что уточнение локализации корреляцией по грубым кандидатам оправдано.

**1.5 Ballard. Generalizing the Hough transform to detect arbitrary shapes. Pattern Recognition 13(2), 1981 [Н]; Borgefors. Hierarchical chamfer matching. IEEE TPAMI 10(6), 1988 [Н].** Основа голосования по R-таблице и сопоставления контуров по карте расстояний. Применимость: chamfer по скелету или контурному изображению уже реализован в проекте (фаза 5), он устойчив к толщине линий и дешев, но не различает символы, отличающиеся одной буквой.

**1.6 Cheng, Wu, Abd-Almageed, Natarajan. QATM: Quality-Aware Template Matching for Deep Learning. CVPR 2019.** https://arxiv.org/abs/1903.07254 [А]. Оценка качества пары «окно, шаблон» через мягкое ранжирование среди всех пар, что учитывает ситуации «один шаблон, много совпадений». Применимость: готовый способ снизить число ложных срабатываний обычной нормированной корреляции в повторяющихся структурах (стены, рамки); на чертежах не проверялся.

## 2. Глубокое обучение на чертежах

### 2.1 Векторные CAD-чертежи (панорамное распознавание)

**2.1.1 Fan, Zhu, Li, Chen, Zhu, Tan. FloorPlanCAD: A Large-Scale CAD Drawing Dataset for Panoptic Symbol Spotting. ICCV 2021.** https://arxiv.org/abs/2105.07147 [А]. Более 10 000 планов, 30 категорий объектов, векторная разметка; предложена базовая модель PanCADNet (CNN-GCN, с головами классификации и детекции). Применимость: PanCADNet отдельной публикации не имеет, это базовая модель этой работы; закрытый словарь 30 классов архитектурных элементов (двери, окна, санузлы), классы нашей легенды не совпадают.

**2.1.2 Fan, Chen, Wang, Wang. CADTransformer: Panoptic Symbol Spotting Transformer for CAD Drawings. CVPR 2022.** https://openaccess.thecvf.com/content/CVPR2022/html/Fan_CADTransformer_Panoptic_Symbol_Spotting_Transformer_for_CAD_Drawings_CVPR_2022_paper.html [П] (страница при прямом обращении вернула 403, сведения из пересказа поиска). ViT-основа над графическими примитивами, соседская самовнимательность, аугментация «Random Layer»; PQ на FloorPlanCAD с 0,595 до 0,685. Применимость: тот же ограничитель, закрытый набор классов и векторный вход.

**2.1.3 Zheng, Li, Zhu, Li, Petzold, Tan. GAT-CADNet: Graph Attention Network for Panoptic Symbol Spotting in CAD Drawings. 2022.** https://arxiv.org/abs/2201.00625 [А]. Чертеж как граф, экземпляры предсказываются как подграфы через матрицу смежности. Применимость: как предыдущая.

**2.1.4 Liu, Yang, Wang, Yu, Zhang. Symbol as Points (SymPoint). ICLR 2024.** https://arxiv.org/abs/2401.10556 [А]. Примитивы как точки, point transformer и голова в стиле Mask2Former; выигрыш над GAT-CADNet 9,6 пункта PQ. Продолжения: SymPoint-V2 (https://arxiv.org/abs/2407.01928, [А], добавляет информацию о слоях и ускоряет сходимость) и CADSpotting (https://arxiv.org/abs/2412.07377, [А], скользящее окно с голосованием и NMS для больших чертежей, датасет LS-CAD из 45 планов). Применимость: сопоставимое число PQ (по таблице CADSpotting, PanCADNet 59,5, CADTransformer 68,9, SymPoint 83,3, [П] как чужая таблица) относится к FloorPlanCAD. Если исходный PDF векторный, эти модели теоретически применимы, но их обучение требует тысяч размеченных листов и фиксированных классов, поэтому как основа для новых комплектов они не подходят.

### 2.2 Детекторы на растре: P&ID, схемы, планы

**2.2.1 Rezvanifar, Cote, Branzan Albu. Symbol Spotting on Digital Architectural Floor Plans Using a Deep Learning-Based Framework. CVPR Workshops 2020.** https://arxiv.org/abs/2006.00684 [А]. YOLO, обучаемый по тайлам, решает проблему малого размера символов на больших планах; устойчивость на SESYD с шумом. Применимость: прямой прецедент тайлинга и YOLO на планах; численных результатов в аннотации нет.

**2.2.2 Elyan, Jamieson, Ali-Gombe. Deep learning for symbols detection and classification in engineering drawings. Neural Networks 129, 2020.** Запись: https://rgu-repository.worktribe.com/OutputFile/928735 [П]. YOLO на P&ID промышленного партнера, более 94% символов распознано; дисбаланс классов смягчается GAN. Применимость: подтверждает, что дисбаланс является главной проблемой и что синтетика для редких классов помогает, но на данных с большим числом экземпляров на класс.

**2.2.3 Rahul, Paliwal, Sharma, Vig. Automatic Information Extraction from Piping and Instrumentation Diagrams. ICPRAM 2019.** https://arxiv.org/abs/1901.11383 [А]. FCN-классификация символов, линии и коды труб классическими методами; «многообещающие результаты» без чисел. Применимость: низкая, показательна гибридность классики и CNN.

**2.2.4 Mani, Haddad, Constantini, Douhard, Li, Poirier. Automatic Digitization of Engineering Diagrams Using Deep Learning and Graph Search. CVPR Workshops 2020.** https://mlanthology.org/cvprw/2020/mani2020cvprw-automatic [А]. CNN классифицирует обычные символы с точностью и полнотой выше 90%, далее граф линий. Применимость: закрытый набор типовых символов.

**2.2.5 Paliwal, Jain, Sharma, Vig. Digitize-PID. PAKDD 2021 workshops.** https://arxiv.org/abs/2109.03794 [А]. Двухэтапное распознавание сложных символов, публичный синтетический набор Dataset-P&ID из 500 листов. Применимость: единственный публично найденный набор для P&ID; домен (трубопроводы) отличается от электрических планов.

**2.2.6 Dosi, Meena, Rajpura, Meena. SkeySpot. IEEE SMC 2025.** https://arxiv.org/abs/2508.10449 [А]. Датасет DELP: 45 сканов электрических планов, 2 450 экземпляров, 34 класса; лучший детектор YOLOv8, mAP 82,5%. Применимость: наиболее близкий по домену результат, показывает порядок достижимого качества для YOLO на 45 листах и закрытом словаре; наши условия жестче (4 листа, новые комплекты).

**2.2.7 Работы по однолинейным схемам (SLED, статьи Applied Sciences 2023 и IJACSA 2023)** [П]. YOLOv5 и YOLOv8 с добавлением GAN-синтетики при дисбалансе, заявлена точность около 95-96%. Страницы статей не открывались, значения приведены по пересказу поиска и требуют сверки с текстом.

## 3. Few-shot, visual prompting, легенда и синтетика

### 3.1 Общие few-shot детекторы и визуальные подсказки

**3.1.1 Minderer и др. Simple Open-Vocabulary Object Detection with Vision Transformers (OWL-ViT). ECCV 2022.** https://arxiv.org/abs/2205.06230 [А]. Поддерживает текстовую детекцию и одношаговую детекцию по изображению-образцу. **Minderer, Gritsenko, Houlsby. Scaling Open-Vocabulary Object Detection (OWLv2). 2023.** https://arxiv.org/abs/2306.09683 [А]. Самообучение на псевдоразметке, LVIS rare AP с 31,2 до 44,6. Применимость: образец легенды можно подать как запрос; ожидается слабая различимость символов, отличающихся буквой, и разрыв домена (фото против линейной графики), оба предположения не проверены на чертежах.

**3.1.2 Jiang и др. T-Rex2.** https://arxiv.org/abs/2403.14610 [А]; **Ren и др. DINO-X.** https://arxiv.org/abs/2411.14347 [А]. Обе модели принимают визуальные подсказки (образец объекта), DINO-X заявляет 63,3 AP на редких классах LVIS minival при нулевом обучении. Применимость: удобны для быстрой проверки гипотезы на 4 листах; условия доступа (API, лицензии) не проверялись, на чертежах результатов в источниках нет.

**3.1.3 Liu и др. Grounding DINO.** https://arxiv.org/abs/2303.05499 [А]. Детекция по тексту, 52,5 AP на COCO без обучения. Применимость: низкая, так как классы чертежа текстом не описываются (буква внутри круга).

**3.1.4 Bulat, Guerrero, Martinez, Tzimiropoulos. FS-DETR. ICCV 2023.** https://arxiv.org/abs/2210.04845 [А]. Образцы подаются как визуальные подсказки без дообучения, число новых классов и образцов произвольно. **Zhang, Liu, Wang, Boularias. DE-ViT (Detect Everything with Few Examples). 2023.** https://arxiv.org/abs/2309.12969 [А]. Без дообучения, лучшие результаты few-shot на COCO и LVIS. Применимость: архитектурно близки нашей постановке (класс задается образцом), но обучены на естественных изображениях.

**3.1.5 Fu и др. Cross-Domain Few-Shot Object Detection via Enhanced Open-Set Object Detector (CD-ViTO). ECCV 2024.** https://arxiv.org/abs/2402.03094 [А]. Бенчмарк показывает, что большинство современных подходов плохо переносятся между областями, падение связано со стилем, межклассовой дисперсией и неопределенными границами. Применимость: хотя бы косвенное предупреждение (чертежи в бенчмарк не входили, по крайней мере в аннотации это не заявлено).

**3.1.6 Oquab и др. DINOv2.** https://arxiv.org/abs/2304.07193 [А]. Универсальные признаки без дообучения. Применимость: основа для сопоставления патчей и верификации кандидатов по эмбеддингу образца; пригодность к линейной графике требует проверки.

### 3.2 Работы на чертежах по одному образцу или по легенде

**3.2.1 Paliwal, Sharma, Vig. OSSR-PID: One-Shot Symbol Recognition in P&ID Sheets using Path Sampling and GCN. IJCNN 2021.** https://arxiv.org/abs/2109.03849 [А]. Один прототип на класс, точки вдоль контуров образуют граф, DGCNN с эмбеддингами ResNet-34 и ArcFace; синтетика 100 листов, реальных 12. Применимость: точнейшая по постановке работа (один образец на класс, классы различаются тонко); численные метрики на странице аннотации не приведены.

**3.2.2 Jamieson, Elyan, Moreno-García. Few-Shot Symbol Detection in Engineering Drawings. Applied Artificial Intelligence 38(1), 2024, e2406712.** https://rgu-repository.worktribe.com/output/2480905 [П] (страницы заблокированы, сведения из пересказа). Менее десяти, иногда один экземпляр на класс, на промышленных чертежах; заявлено превосходство над современными методами. Детали архитектуры не проверены.

**3.2.3 Sarkar, Pandey, Kar. Automatic Detection and Classification of Symbols in Engineering Drawings. 2022.** https://arxiv.org/abs/2204.13277 [А]. Несколько сетей находят объекты из таблицы легенды, не требуя дообучения на новых шаблонах. Результатов в аннотации нет; это единственная найденная работа, прямо ориентированная на легенду.

**3.2.4 Автоматическое обнаружение шахт на ситуационных планах (TUHH, 2024).** https://tore.tuhh.de/bitstreams/3e49592b-0833-4376-b871-cdbdd2d7cd4a/download [П]. Из файла удалось прочитать только название; по пересказу поиска: поиск области легенды OCR, выбор нужного символа, детекция на плане. Предварительная работа.

**3.2.5 Odqvist. Zero-Shot Detection and Classification of Symbols in P&IDs. Master's thesis, Chalmers, 2026.** https://odr.chalmers.se/items/fb27a1a9-9f0b-416e-a920-15fbf0af398a [А]. Детекция сводится к классификации визуально-языковой моделью из списка категорий; численных результатов на странице нет.

### 3.3 Синтетика вставкой символов

**3.3.1 Dwibedi, Misra, Hebert. Cut, Paste and Learn. ICCV 2017.** https://arxiv.org/abs/1708.01642 [А]. Вырезание и вставка экземпляров на случайные фоны; достаточен реализм на уровне патча, детектор учат игнорировать артефакты границ; с 10% реальных данных превосходит обучение на всех реальных в межобластной постановке. **Ghiasi и др. Simple Copy-Paste. CVPR 2021.** https://arxiv.org/abs/2012.07177 [А]. Случайная вставка улучшает сильные базовые модели, на редких категориях LVIS даны заметные приросты.

**3.3.2 Gupta, Wei, Czerniawski. Automated Valve Detection in P&ID Diagrams. ISARC 2022.** https://www.iaarc.org/publications/2022_proceedings_of_the_39th_isarc_bogota_colombia/automated_valve_detection_in_piping_and_instrumentation_pid_diagrams.html [А]. Синтетические данные случайным вырезанием символов, исследуется влияние относительного размера символа и фона; результаты названы обнадеживающими. Применимость: прямой прецедент для нашего случая, количественных значений нет.

## 4. Метрики и особенности

**4.1 Akyon, Altinuc, Temizel. SAHI. ICIP 2022.** https://arxiv.org/abs/2202.06934 [А]. Инференс по срезам повышает AP на мелких объектах на 5-7 пунктов, вместе с дообучением по срезам до 12-14 пунктов (VisDrone, xView). Применимость: символ 35-80 px на листе 8400x6000 занимает около 0,5-1% ширины, тайлинг с перекрытием обязателен независимо от выбранного детектора; перекрытие должно быть не меньше размера символа, чтобы объект не разрезался.

**4.2 Скользящее окно с голосованием и NMS** в CADSpotting (см. 2.1.4) [А] и обучение по тайлам у Rezvanifar (2.2.1) [А] подтверждают тот же подход для чертежей.

**4.3 Lin и др. Focal Loss.** https://arxiv.org/abs/1708.02002 [А] и **Gupta, Dollár, Girshick. LVIS.** https://arxiv.org/abs/1908.03195 [А]. Первая работа лечит дисбаланс «фон против объекта», вторая формулирует проблему длинного хвоста (редкие категории с малым числом примеров). Применимость: на листе фон составляет почти все пиксели, а классы с 1-5 экземплярами не дадут стабильной оценки AP; предлагается отчитываться по классам и по частотным группам, а не одним mAP.

**4.4 Повороты.** Среди проверенного материала отдельной работы по поворотам символов на чертежах не найдено. Предлагается (оценка составителя) учитывать повороты на 90/180/270 градусов аугментацией при обучении и перебором четырех поворотов образца при сопоставлении; симметричные символы и «буквенные» символы (текст в круге) при повороте дают неоднозначность, которую нужно разрешать по контексту.

## Применимость YOLO к нашему случаю

Аргументы за. Зрелая реализация, быстрый обучаемый детектор, есть прецеденты на планах (Rezvanifar 2020), P&ID (Elyan 2020) и электрических планах (SkeySpot 2025, mAP 82,5% при YOLOv8). Тайлинг с перекрытием решает проблему размера. Синтетика вставкой образцов хорошо изучена (Dwibedi 2017, Ghiasi 2021, Gupta 2022).

Аргументы против. Закрытый словарь: новый комплект приносит новые символы, а модель по умолчанию не умеет «класс по образцу». На 4 листах и 325 объектах хвост из 1-5 экземпляров не обучается; в прецедентах выше десятки листов и сотни-тысячи экземпляров. Классы, отличающиеся одной буквой, требуют разрешения, близкого к исходному (35-80 px это допускает, но класс определяется мелкой деталью), а стандартный YOLO с сильным даунсэмплингом и аугментациями (mosaic, масштаб) может их смешивать. Результаты по числам SkeySpot получены на другом домене и данных; переносить их на наши листы нельзя.

Разумные постановки (оценки составителя).
1. Класс-независимый детектор «символ или не символ» (один или несколько грубых классов: круг, квадрат, ромб, прочее). Обучение на 4 размеченных листах плюс синтетика; различение классов выносится на второй этап сопоставлением с образцом легенды. Риск: детектор должен переноситься на второе семейство листов; проверка отложенным семейством обязательна.
2. Обучение на синтетике из легенды конкретного комплекта: вставка образцов (повороты 0/90/180/270, варьирование толщины и масштаба) на фоны планов этого же комплекта с предварительно стертыми символами, плюс реальные листы. Закрытый словарь здесь пересоздается для каждого комплекта, что допустимо, если время дообучения невелико. Риск: разрыв между видом символа в легенде и на плане (размер, толщина, пересечения со стенами и текстом).
3. Закрытый набор классов, обученный только на размеченных листах: по имеющимся данным не рекомендуется, кроме как база сравнения.

## Кандидатные пайплайны (по убыванию ожидаемой пользы)

1. **Кандидаты по образцам легенды плюс обучаемая верификация.** Многомасштабный и четырехповоротный chamfer или корреляция по скелету дает гипотезы, затем малая сеть-верификатор (эмбеддинги DINOv2 или метрическое обучение с ArcFace, как в OSSR-PID), дообученная на синтетических вариантах образцов и 4 листах, подтверждает или отклоняет и различает классы с разницей в букву (дополнительно OCR внутри рамки). Опора: проектная фаза 5, OSSR-PID, QATM. Риски: полнота кандидатов на сильно зашумленных участках, смешение классов по букве, время работы на листе 8400x6000.
2. **Двухстадийная схема: класс-независимый YOLO и классификация по прототипам легенды.** Детектор из постановки 1, классификатор по ближайшему прототипу (образец легенды, расширенный аугментациями) с порогом «неизвестный класс». Риски: перенос детектора на новое семейство листов, калибровка порога для редких классов.
3. **Синтетика из легенды комплекта и дообучение YOLO или RT-DETR на лету.** Постановка 2 выше. Риски: домен легенды против плана, затраты на подготовку фонов, переобучение на артефакты вставки, неустойчивая оценка при 1-5 экземплярах реальной проверки.
4. **Визуальные подсказки готовых моделей (OWLv2 image-guided, T-Rex2/DINO-X, DE-ViT) как генератор кандидатов и контрольная точка.** Быстро проверяется на 4 листах без обучения. Риски: разрыв домена (CD-FSOD показывает деградацию), невидимость разницы в букву, ограничения доступа и лицензий, обработка тайлами.
5. **Few-shot трансформер, дообученный на синтетике и реальных листах (FS-DETR, CD-ViTO).** Теоретически лучше всего соответствует постановке «класс по образцу». Риски: высокая сложность, неизвестная доступность кода и весов в проверенных источниках, вычислительная стоимость на больших листах.
6. **Векторный путь, если PDF сохранил примитивы.** Группировка примитивов и сопоставление подграфов (в духе Dutta 2011, SymPoint) с образцом из векторной легенды, без растрового шума. Риски: PDF может содержать растр или рисованные в кривых тексты, различные способы построения символов у двух проектировщиков, обучаемые модели FloorPlanCAD неприменимы напрямую.

## Список источников

1. Rusiñol M., Lladós J. Symbol Spotting in Technical Drawings Using Vectorial Signatures // Graphics Recognition (GREC 2005), LNCS 3926. 2006. https://doi.org/10.1007/11767978_4
2. Dutta A., Lladós J., Pal U. Symbol Spotting in Line Drawings Through Graph Paths Hashing // ICDAR. 2011. P. 982-986. https://openresearch.surrey.ac.uk/permalink/44SUR_INST/15d8lgh/alma991109888602346
3. Delalandre M., Valveny E., Pridmore T., Karatzas D. Generation of Synthetic Documents for Performance Evaluation of Symbol Recognition & Spotting Systems // IJDAR. 2010. Vol. 13(3). P. 187-207. https://hal.archives-ouvertes.fr/hal-01022594
4. Rezvanifar A. Analyzing symbols in architectural floor plans via traditional computer vision and deep learning approaches. PhD thesis, University of Victoria, 2021. https://dspace.library.uvic.ca/items/3cc20c2c-8c51-4d7d-a7d9-77c08b9258dc
5. Ballard D. H. Generalizing the Hough transform to detect arbitrary shapes // Pattern Recognition. 1981. Vol. 13(2). (не сверялась)
6. Borgefors G. Hierarchical chamfer matching // IEEE TPAMI. 1988. Vol. 10(6). (не сверялась)
7. Cheng J., Wu Y., Abd-Almageed W., Natarajan P. QATM: Quality-Aware Template Matching for Deep Learning // CVPR. 2019. https://arxiv.org/abs/1903.07254
8. Fan Z., Zhu L., Li H., Chen X., Zhu S., Tan P. FloorPlanCAD // ICCV. 2021. https://arxiv.org/abs/2105.07147
9. Fan Z., Chen T., Wang P., Wang Z. CADTransformer // CVPR. 2022. https://openaccess.thecvf.com/content/CVPR2022/html/Fan_CADTransformer_Panoptic_Symbol_Spotting_Transformer_for_CAD_Drawings_CVPR_2022_paper.html
10. Zheng Z. et al. GAT-CADNet // 2022. https://arxiv.org/abs/2201.00625
11. Liu W., Yang T., Wang Y., Yu Q., Zhang L. Symbol as Points // ICLR. 2024. https://arxiv.org/abs/2401.10556
12. Liu W., Yang T., Yu Q., Zhang L. SymPoint Revolutionized // 2024. https://arxiv.org/abs/2407.01928
13. Yang F. et al. CADSpotting // 2024. https://arxiv.org/abs/2412.07377
14. Rezvanifar A., Cote M., Branzan Albu A. Symbol Spotting on Digital Architectural Floor Plans Using a Deep Learning-Based Framework // CVPR Workshops. 2020. https://arxiv.org/abs/2006.00684
15. Elyan E., Jamieson L., Ali-Gombe A. Deep learning for symbols detection and classification in engineering drawings // Neural Networks. 2020. Vol. 129. P. 91-102. https://rgu-repository.worktribe.com/OutputFile/928735
16. Rahul R., Paliwal S., Sharma M., Vig L. Automatic Information Extraction from Piping and Instrumentation Diagrams // ICPRAM. 2019. https://arxiv.org/abs/1901.11383
17. Mani S. et al. Automatic Digitization of Engineering Diagrams Using Deep Learning and Graph Search // CVPR Workshops. 2020. https://mlanthology.org/cvprw/2020/mani2020cvprw-automatic
18. Paliwal S., Jain A., Sharma M., Vig L. Digitize-PID // PAKDD workshops. 2021. https://arxiv.org/abs/2109.03794
19. Dosi D., Meena R., Rajpura P., Meena Y. K. SkeySpot // IEEE SMC. 2025. https://arxiv.org/abs/2508.10449
20. Minderer M. et al. Simple Open-Vocabulary Object Detection with Vision Transformers // ECCV. 2022. https://arxiv.org/abs/2205.06230
21. Minderer M., Gritsenko A., Houlsby N. Scaling Open-Vocabulary Object Detection // 2023. https://arxiv.org/abs/2306.09683
22. Jiang Q. et al. T-Rex2 // 2024. https://arxiv.org/abs/2403.14610
23. Ren T. et al. DINO-X // 2024. https://arxiv.org/abs/2411.14347
24. Liu S. et al. Grounding DINO // 2023. https://arxiv.org/abs/2303.05499
25. Bulat A. et al. FS-DETR // ICCV. 2023. https://arxiv.org/abs/2210.04845
26. Zhang X. et al. Detect Everything with Few Examples (DE-ViT) // 2023. https://arxiv.org/abs/2309.12969
27. Fu Y. et al. Cross-Domain Few-Shot Object Detection via Enhanced Open-Set Object Detector // ECCV. 2024. https://arxiv.org/abs/2402.03094
28. Oquab M. et al. DINOv2 // 2023. https://arxiv.org/abs/2304.07193
29. Paliwal S., Sharma M., Vig L. OSSR-PID // IJCNN. 2021. https://arxiv.org/abs/2109.03849
30. Jamieson L., Elyan E., Moreno-García C. F. Few-Shot Symbol Detection in Engineering Drawings // Applied Artificial Intelligence. 2024. Vol. 38(1). e2406712. https://rgu-repository.worktribe.com/output/2480905
31. Sarkar S., Pandey P., Kar S. Automatic Detection and Classification of Symbols in Engineering Drawings // 2022. https://arxiv.org/abs/2204.13277
32. Automatic Detection of Shaft Locations in Site Plans. TUHH, 2024 (авторы не установлены). https://tore.tuhh.de/bitstreams/3e49592b-0833-4376-b871-cdbdd2d7cd4a/download
33. Odqvist C. Zero-Shot Detection and Classification of Symbols in P&IDs. Master's thesis, Chalmers, 2026. https://odr.chalmers.se/items/fb27a1a9-9f0b-416e-a920-15fbf0af398a
34. Dwibedi D., Misra I., Hebert M. Cut, Paste and Learn // ICCV. 2017. https://arxiv.org/abs/1708.01642
35. Ghiasi G. et al. Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation // CVPR. 2021. https://arxiv.org/abs/2012.07177
36. Gupta M., Wei C., Czerniawski T. Automated Valve Detection in Piping and Instrumentation (P&ID) Diagrams // ISARC. 2022. P. 630-637. https://www.iaarc.org/publications/2022_proceedings_of_the_39th_isarc_bogota_colombia/automated_valve_detection_in_piping_and_instrumentation_pid_diagrams.html
37. Akyon F. C., Altinuc S. O., Temizel A. Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection (SAHI) // ICIP. 2022. https://arxiv.org/abs/2202.06934
38. Lin T.-Y. et al. Focal Loss for Dense Object Detection // 2017. https://arxiv.org/abs/1708.02002
39. Gupta A., Dollár P., Girshick R. LVIS: A Dataset for Large Vocabulary Instance Segmentation // 2019. https://arxiv.org/abs/1908.03195
