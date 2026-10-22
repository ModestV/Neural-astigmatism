"""Проверка разметки глазами: рисует рамки на картинках и сохраняет превью.

Работает и с результатом аугментации (`<out>/annotations.json`), и с исходной разметкой.
Что нарисовано:
- рамка `object` — тонкая линия, рамка `legend_item` — толстая;
- цвет зависит от `legend_id`: объект и его образец в легенде одного цвета, так связь видно сразу;
- подпись — `legend_id`; у образца спереди `L`; звёздочка `*` у объекта означает, что образца этого
  объекта в этой картинке нет (так бывает на плитках);
- рамки, обрезанные краем плитки (`truncated`), подписаны знаком `~`.

Пример:

    python visualize.py ../../../data/aug_try/annotations.json --images ../../../data/aug_try/images \\
        --out ../../../data/aug_try/preview --max 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from coco_io import LEGEND_ITEM, OBJECT, Sheet, collect_annotation_paths, find_problems, load_sheets

PALETTE = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
    (240, 50, 230), (210, 245, 60), (0, 128, 128), (170, 110, 40), (128, 0, 0), (0, 0, 128),
]


def color_for(legend_id: str | None) -> tuple[int, int, int]:
    """Цвет по legend_id; у рамок без legend_id серый."""
    if legend_id is None:
        return (128, 128, 128)
    try:
        index = int(legend_id)
    except ValueError:
        index = sum(map(ord, legend_id))
    return PALETTE[index % len(PALETTE)]


def draw_sheet(image: Image.Image, sheet: Sheet, max_side: int = 2400) -> Image.Image:
    """Возвращает уменьшенную копию картинки с нарисованными рамками."""
    factor = min(1.0, max_side / max(image.size))
    size = (max(1, round(image.width * factor)), max(1, round(image.height * factor)))
    canvas = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:  # старый Pillow без параметра size
        font = ImageFont.load_default()
    legend_ids = {b.legend_id for b in sheet.boxes if b.category == LEGEND_ITEM}
    for box in sheet.boxes:
        x, y, w, h = (v * factor for v in box.bbox)
        color = color_for(box.legend_id)
        is_legend = box.category == LEGEND_ITEM
        draw.rectangle([x, y, x + w, y + h], outline=color, width=4 if is_legend else 2)
        label = f"L{box.legend_id}" if is_legend else f"{box.legend_id}"
        if box.category == OBJECT and box.legend_id not in legend_ids:
            label += "*"
        if box.attributes.get("truncated"):
            label += "~"
        draw.text((x, max(0, y - 18)), label, fill=color, font=font)
    return canvas


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("annotations", nargs="+", type=Path, help="файлы COCO или папки с ними")
    p.add_argument("--images", type=Path, required=True, help="папка с картинками")
    p.add_argument("--out", type=Path, required=True, help="куда сохранить превью")
    p.add_argument("--max", type=int, default=12, help="сколько картинок нарисовать (по умолчанию 12)")
    p.add_argument("--only", nargs="+", metavar="ЧАСТЬ_ИМЕНИ",
                   help="рисовать только картинки с такой подстрокой в имени")
    p.add_argument("--max-side", type=int, default=2400,
                   help="длинная сторона превью, px (по умолчанию 2400)")
    args = p.parse_args(argv)

    sheets = load_sheets(collect_annotation_paths(args.annotations))
    if args.only:
        sheets = [s for s in sheets if any(part in s.file_name for part in args.only)]
    problems = find_problems(sheets, args.images)
    for line in problems:
        print("ПРОБЛЕМА:", line)

    # Равномерная выборка, чтобы в превью попали и повороты, и масштабы, и плитки.
    if len(sheets) > args.max > 0:
        step = len(sheets) / args.max
        sheets = [sheets[int(i * step)] for i in range(args.max)]
    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    for sheet in sheets:
        path = args.images / sheet.file_name
        if not path.exists():
            continue
        with Image.open(path) as image:
            preview = draw_sheet(image, sheet, args.max_side)
        target = args.out / f"preview_{Path(sheet.file_name).stem}.png"
        preview.save(target)
        written += 1
        print("Превью:", target)
    print(f"Нарисовано: {written}; проблем: {len(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
