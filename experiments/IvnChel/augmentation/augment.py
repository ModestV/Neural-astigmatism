"""Аугментация размеченного набора: картинки листов + COCO-разметка → новые картинки + новая разметка.

Как это работает, простыми словами. Для каждого листа делается несколько «геометрических версий»
(исходный, повороты, масштабы, зеркала — что попросите). Каждая версия при желании режется на плитки.
Затем к каждой картинке, если заданы фильтры внешнего вида (шум, размытие и т. д.), добавляется ещё
одна «испорченная» копия. Рамки и связь legend_id пересчитываются вместе с картинкой.

Пример (бережный запуск на двух листах):

    python augment.py --out ../../../data/aug_try --only 02_renee 04_lakeview \\
        --rotate 90 270 --scale 0.5 --tile 1600x1600 --overlap 200 --noise 12 --blur 0.8

Порядок фильтров внешнего вида: толщина линий, порог, инверсия, пятна, яркость/контраст,
размытие, шум, вырезание квадратов, JPEG. Результат: <out>/images/*.png и <out>/annotations.json.
"""

from __future__ import annotations

import argparse
import sys
import zlib
from dataclasses import replace
from pathlib import Path

from PIL import Image

import ops
import photometric
import cutout as cutout_op
from coco_io import (
    LEGEND_ITEM,
    OBJECT,
    Sheet,
    collect_annotation_paths,
    find_problems,
    load_sheets,
    save_coco,
)
from tiling import LEGEND_MODES, tile_sheet

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ANNOTATIONS = ROOT / "datasets" / "labeled-dataset" / "annotations"
DEFAULT_IMAGES = ROOT / "datasets" / "labeled-dataset" / "images" / "images" / "default"


def parse_tile(text: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("размер плитки задаётся как ШИРИНАxВЫСОТА, например 1600x1600")
    return width, height


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--annotations", nargs="+", type=Path, default=[DEFAULT_ANNOTATIONS],
                   help="файлы COCO или папки с ними (по умолчанию datasets/labeled-dataset/annotations)")
    p.add_argument("--images", type=Path, default=DEFAULT_IMAGES,
                   help="папка с картинками листов (по умолчанию datasets/labeled-dataset/images/images/default)")
    p.add_argument("--out", type=Path, required=True,
                   help="куда писать результат (не кладите в git: картинки большие)")
    p.add_argument("--only", nargs="+", metavar="ЧАСТЬ_ИМЕНИ",
                   help="брать только листы, в имени файла которых есть одна из этих подстрок")
    p.add_argument("--format", choices=["png", "jpg"], default="png",
                   help="формат результата; png без потерь (по умолчанию)")
    p.add_argument("--seed", type=int, default=0, help="для повторяемости случайных операций")

    g = p.add_argument_group("геометрия (рамки пересчитываются)")
    g.add_argument("--rotate", nargs="*", type=int, choices=[90, 180, 270], default=[],
                   help="повороты против часовой стрелки, по версии на каждый угол")
    g.add_argument("--scale", nargs="*", type=float, default=[], metavar="K",
                   help="коэффициенты масштаба, например 0.5 1.5")
    g.add_argument("--mirror", nargs="*", choices=["h", "v"], default=[],
                   help="зеркало: h слева направо, v сверху вниз. ОСТОРОЖНО: меняет смысл знаков")
    g.add_argument("--no-original", action="store_true",
                   help="не включать неповернутую версию")

    t = p.add_argument_group("плитки")
    t.add_argument("--tile", type=parse_tile, metavar="ШxВ", help="резать на плитки, например 1600x1600")
    t.add_argument("--overlap", type=int, default=0, help="перекрытие плиток, px")
    t.add_argument("--min-visible", type=float, default=0.5,
                   help="доля площади объекта, которая должна остаться в плитке")
    t.add_argument("--legend-mode", choices=LEGEND_MODES, default="keep",
                   help="keep: все плитки с флагом legend_in_tile; require: только плитки с образцами; "
                        "with-legend: недостающие образцы приклеиваются полосой под плиткой")
    t.add_argument("--keep-empty", action="store_true", help="оставлять плитки без рамок")

    f = p.add_argument_group("внешний вид (рамки не меняются); если задан хоть один, к каждой "
                             "картинке добавляется копия с этими искажениями")
    f.add_argument("--thickness", type=int, help="толщина линий: >0 толще, <0 тоньше, на |K| px")
    f.add_argument("--threshold", type=int, help="чёрно-белый порог 0..255")
    f.add_argument("--invert", action="store_true", help="инверсия цветов")
    f.add_argument("--stains", type=int, help="число тёмных пятен (вне рамок)")
    f.add_argument("--brightness", type=float, help="яркость, 1.0 без изменений")
    f.add_argument("--contrast", type=float, help="контраст, 1.0 без изменений")
    f.add_argument("--blur", type=float, help="радиус размытия, px")
    f.add_argument("--noise", type=float, help="сила шума, уровней яркости из 255")
    f.add_argument("--jpeg", type=int, help="качество JPEG 1..95 (меньше — хуже)")
    f.add_argument("--cutout", type=int, metavar="N",
                   help="N белых квадратов; закрытые знаки получают occluded или теряют рамку")
    f.add_argument("--cutout-size", type=int, nargs=2, default=[20, 60], metavar=("МИН", "МАКС"),
                   help="сторона квадрата, px")
    f.add_argument("--no-clean", action="store_true",
                   help="не сохранять версии без искажений (нужен хотя бы один фильтр внешнего вида)")
    return p


def geometry_variants(args) -> list[tuple[str, object]]:
    """Список (тег для имени файла, операция или None для исходного листа)."""
    variants: list[tuple[str, object]] = [] if args.no_original else [("", None)]
    for angle in args.rotate:
        variants.append((f"rot{angle}", lambda i, s, a=angle: ops.rotate_90(i, s, a)))
    for factor in args.scale:
        variants.append((f"x{factor:g}", lambda i, s, k=factor: ops.scale(i, s, k)))
    for axis in args.mirror:
        variants.append((f"flip{axis}", lambda i, s, a=axis: ops.flip(i, s, a)))
    return variants


def photo_chain(args):
    """Цепочка фильтров внешнего вида в фиксированном порядке; пустая, если ничего не задано."""
    steps = []
    if args.thickness:
        steps.append(lambda i, s, seed: photometric.line_thickness(i, s, args.thickness))
    if args.threshold is not None:
        steps.append(lambda i, s, seed: photometric.threshold(i, s, args.threshold))
    if args.invert:
        steps.append(lambda i, s, seed: photometric.invert(i, s))
    if args.stains:
        steps.append(lambda i, s, seed: photometric.stains(i, s, args.stains, seed=seed))
    if args.brightness is not None or args.contrast is not None:
        steps.append(lambda i, s, seed: photometric.brightness_contrast(
            i, s, args.brightness or 1.0, args.contrast or 1.0))
    if args.blur:
        steps.append(lambda i, s, seed: photometric.blur(i, s, args.blur))
    if args.noise:
        steps.append(lambda i, s, seed: photometric.add_noise(i, s, args.noise, seed=seed))
    if args.cutout:
        steps.append(lambda i, s, seed: cutout_op.cutout(
            i, s, args.cutout, tuple(args.cutout_size), seed=seed))
    if args.jpeg:
        steps.append(lambda i, s, seed: photometric.jpeg_compress(i, s, args.jpeg))
    return steps


def apply_chain(steps, image: Image.Image, sheet: Sheet, seed: int) -> tuple[Image.Image, Sheet]:
    for step in steps:
        image, sheet = step(image, sheet, seed)
    return image, sheet


def with_name(sheet: Sheet, tag: str, fmt: str) -> Sheet:
    """Новое имя файла: `<имя листа>__<тег>.<формат>`; без тега — только смена формата."""
    stem = Path(sheet.file_name).stem
    return replace(sheet, file_name=f"{stem}__{tag}.{fmt}" if tag else f"{stem}.{fmt}")


def save_image(image: Image.Image, path: Path) -> None:
    if path.suffix == ".jpg":
        image.convert("RGB").save(path, quality=95)
    else:
        image.save(path)


def load_image(path: Path) -> Image.Image:
    image = Image.open(path)
    image.load()
    return image if image.mode in ("L", "RGB") else image.convert("RGB")


def run(args) -> dict:
    """Выполняет аугментацию и возвращает сводку: листы, пропущенные листы, проблемы разметки."""
    variants = geometry_variants(args)
    chain = photo_chain(args)
    if args.no_clean and not chain:
        raise SystemExit("--no-clean имеет смысл только вместе с фильтрами внешнего вида")
    if not variants:
        raise SystemExit("--no-original без поворотов, масштабов и зеркал ничего не оставляет")
    sheets = load_sheets(collect_annotation_paths(args.annotations))
    if args.only:
        sheets = [s for s in sheets if any(part in s.file_name for part in args.only)]
    out_images = args.out / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    results: list[Sheet] = []
    skipped: list[str] = []
    for sheet in sheets:
        path = args.images / sheet.file_name
        if not path.exists():
            skipped.append(f"{sheet.file_name}: нет файла в {args.images}")
            continue
        image = load_image(path)
        if image.size != (sheet.width, sheet.height):
            skipped.append(f"{sheet.file_name}: размер файла {image.size} не совпадает с разметкой "
                           f"{(sheet.width, sheet.height)}")
            continue
        base = with_name(sheet, "", args.format)
        for tag, op in variants:
            version_image, version = (image, base) if op is None else op(image, base)
            version = with_name(version, tag, args.format)
            if args.tile:
                pieces = tile_sheet(version_image, version, args.tile, args.overlap,
                                    args.min_visible, legend_mode=args.legend_mode,
                                    keep_empty=args.keep_empty)
            else:
                pieces = [(version_image, version)]
            for piece_image, piece in pieces:
                if not args.no_clean:
                    save_image(piece_image, out_images / piece.file_name)
                    results.append(piece)
                if chain:
                    seed = (args.seed + zlib.crc32(piece.file_name.encode())) % 2**32
                    bad_image, bad = apply_chain(chain, piece_image, piece, seed)
                    bad = with_name(bad, "photo", args.format)
                    save_image(bad_image, out_images / bad.file_name)
                    results.append(bad)
    save_coco(results, args.out / "annotations.json")
    return {"sheets": results, "skipped": skipped, "problems": find_problems(results, out_images)}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args)
    sheets = summary["sheets"]
    boxes = [b for s in sheets for b in s.boxes]
    print(f"Картинок: {len(sheets)}; рамок: {len(boxes)} "
          f"(object {sum(b.category == OBJECT for b in boxes)}, "
          f"legend_item {sum(b.category == LEGEND_ITEM for b in boxes)})")
    print(f"Результат: {args.out}")
    for line in summary["skipped"]:
        print("ПРОПУЩЕН:", line)
    for line in summary["problems"]:
        print("ПРОБЛЕМА:", line)
    return 1 if summary["problems"] or (summary["skipped"] and not sheets) else 0


if __name__ == "__main__":
    sys.exit(main())
