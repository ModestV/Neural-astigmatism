# Drawing converter

Утилита для преобразований между SVG, PNG и PDF с приоритетом качества, важного для обработки чертежей моделью.

## Установка

```bash
python3 -m pip install -r requirements.txt
```

Для `PDF -> PNG` и `PDF -> SVG` нужен Poppler. На macOS: `brew install poppler`. Если CairoSVG недоступен, на macOS утилита использует установленный Google Chrome для преобразований `SVG -> PNG/PDF`.

## Примеры

```bash
python3 src/drawing_converter.py convert samples/test_drawing.svg samples/test_drawing.png --dpi 300
python3 src/drawing_converter.py convert samples/test_drawing.svg samples/test_drawing.pdf
python3 src/drawing_converter.py open samples/test_drawing.pdf
```

## Ограничения форматов

- SVG -> PDF сохраняет векторные линии и текст.
- PDF -> SVG возможен, если в системе есть `pdftocairo` из Poppler; обрабатывается первая страница.
- PDF -> PNG рендерит первую страницу с выбранным DPI.
- PNG -> SVG не может честно восстановить векторные объекты. Утилита создаёт SVG-контейнер с исходными пикселями без потери качества.
- PNG -> PDF помещает исходное изображение в PDF без масштабирования числа пикселей.

Для анализа чертежей храните канонический SVG/PDF и создавайте PNG только для подачи модели. Практичный старт: 300 DPI; для мелких размерных линий и текста — 600 DPI. Не конвертируйте PNG туда-обратно перед анализом.
