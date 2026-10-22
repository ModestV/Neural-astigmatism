"""Аугментация в векторе: из PDF-чертежа получаем новые PDF (поворот на 90° и масштаб) и рамки к ним.

Как это работает, простыми словами. Рамки в разметке заданы в пикселях картинки, которую CVAT получил
из PDF при 200 dpi (1 пункт PDF = 200/72 пикселя). Поэтому для новой PDF рамки считаются в пикселях
той же картинки при том же 200 dpi, и пара «PDF + COCO» ведёт себя так же, как «картинка + COCO».

Поворот не трогает содержимое страницы: меняется только флаг `/Rotate`, линии остаются линиями.
Масштаб меняет размер страницы вместе с содержимым, вектор не теряет качества.
Cutout рисует белые квадраты поверх страницы и пересчитывает рамки (как в `cutout.py`).
Толщина линий и инверсия — в `pdf_style.py`.
Плитки — отдельные PDF-страницы с вырезанным куском листа (как в `tiling.py`, включая with-legend).
Если у исходной страницы уже есть `/Rotate` (у 02, 04 и 09 он равен 270), он учитывается.

Пример:

    python pdf_ops.py --pdf "../../../datasets/labeled-dataset/images/pdf_src/чертежи/02_renee_house.pdf" \\
        --out ../../../data/pdf_aug_try --rotate 90 270 --scale 0.5 2 --tile 1600x1600 --overlap 100 --cutout 4 --thickness 1 --invert --check
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import tempfile
import zlib
from dataclasses import replace
from pathlib import Path

from PIL import Image

import ops
from coco_io import Sheet, collect_annotation_paths, load_sheets, save_coco
from cutout import plan_cutout
from pdf_style import invert_page, thicken_page
from tiling import LEGEND_MODES, plan_legend_strip, plan_tiles, with_legend_strip

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ANNOTATIONS = ROOT / "datasets" / "labeled-dataset" / "annotations"

# Картинки для CVAT рендерили при 200 dpi (проверено на размерах всех 10 страниц).
DPI = 200


def page_pixel_size(page, dpi: int = DPI) -> tuple[int, int]:
    """Размер страницы в пикселях при рендере с данным dpi.

    MuPDF округляет вверх с допуском 0.001 px: страница 1190.55 пт даёт 3308 px, а 400 пт при
    пересчёте туда-обратно не должна давать 1601 из-за дробного хвоста.
    """
    k = dpi / 72
    rect = page.rect
    return math.ceil(rect.width * k - 0.001), math.ceil(rect.height * k - 0.001)


def transform_pdf(
    src_path: Path,
    sheet: Sheet,
    out_path: Path,
    rotate: int = 0,
    factor: float = 1.0,
    cutout: int = 0,
    cutout_size: tuple[int, int] = (20, 60),
    seed: int = 0,
) -> Sheet:
    """Делает новую PDF: поворот на `rotate` градусов против часовой (0, 90, 180, 270), масштаб, cutout.

    `sheet` — лист исходной PDF с рамками в пикселях при 200 dpi. Возвращает новый лист: рамки,
    `rotation` у объектов и размер пересчитаны, в `history` записаны операции.

    cutout: сколько белых квадратов поставить; `cutout_size` — их сторона в пикселях готового листа
    (после поворота и масштаба). Квадрат рисуется поверх содержимого, сами линии под ним остаются в
    файле, но на странице их не видно. Правила про долю закрытой рамки — в `cutout.plan_cutout`.
    """
    import pymupdf  # тяжёлую библиотеку импортируем внутри, как велит AGENTS.md

    if rotate not in (0, 90, 180, 270):
        raise ValueError(f"Поворот только на 0, 90, 180 или 270 градусов, получено {rotate}")
    if factor <= 0:
        raise ValueError("Коэффициент масштаба должен быть положительным")

    with pymupdf.open(src_path) as src:
        if len(src) != 1:
            raise ValueError(f"{src_path.name}: ожидается один лист, найдено {len(src)}")
        shown_size = page_pixel_size(src[0])
        if shown_size != (sheet.width, sheet.height):
            raise ValueError(f"{src_path.name}: страница даёт {shown_size} px, в разметке "
                             f"{(sheet.width, sheet.height)}; рамки не совпадут")
        source_rotation = src[0].rotation
        # Снимаем /Rotate исходника: show_pdf_page его не учитывает, мы вернём его в конце.
        src[0].set_rotation(0)
        box = src[0].rect
        out = pymupdf.open()
        page = out.new_page(width=box.width * factor, height=box.height * factor)
        page.show_pdf_page(page.rect, src, 0)
        # /Rotate в PDF идёт по часовой стрелке, а поворот листа у нас против часовой.
        page.set_rotation((source_rotation + (360 - rotate)) % 360)
        new_size = page_pixel_size(page)

        # Рамки: сначала поворот (формулы из ops, картинка там не нужна), потом масштаб по осям.
        current = replace(sheet, history=list(sheet.history))
        if rotate:
            _, current = ops.rotate_90(Image.new("1", (1, 1)), current, rotate)
        sx, sy = new_size[0] / current.width, new_size[1] / current.height
        boxes = [
            replace(b, bbox=(b.bbox[0] * sx, b.bbox[1] * sy, b.bbox[2] * sx, b.bbox[3] * sy))
            for b in current.boxes
        ]
        history = list(current.history)
        if factor != 1:
            history.append(f"pdf_scale:{factor:g}")
        result = replace(current, file_name=out_path.name, width=new_size[0], height=new_size[1],
                         boxes=boxes, history=history)

        if cutout:
            result = _draw_cutout(page, result, cutout, cutout_size, seed)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.save(out_path, garbage=3, deflate=True)
        out.close()
    return result


def _draw_cutout(page, sheet: Sheet, count: int, size: tuple[int, int], seed: int) -> Sheet:
    """Рисует на странице белые квадраты и возвращает лист с пересчитанными рамками."""
    import pymupdf

    squares, result = plan_cutout(sheet, sheet.width, sheet.height, count=count, size=size, seed=seed)
    k = 72 / DPI
    for x0, y0, x1, y1 in squares:
        # Квадраты заданы в пикселях показанной страницы, а рисует PyMuPDF в системе страницы
        # без /Rotate, поэтому переводим через derotation_matrix.
        rect = pymupdf.Rect(x0 * k, y0 * k, x1 * k, y1 * k) * page.derotation_matrix
        page.draw_rect(rect, color=None, fill=(1, 1, 1))
    return result


def cutout_pdf(
    src_path: Path, sheet: Sheet, out_path: Path, count: int,
    size: tuple[int, int] = (20, 60), seed: int = 0,
) -> Sheet:
    """Копия готовой PDF (например, плитки) с белыми квадратами; `sheet` описывает её страницу."""
    import pymupdf

    with pymupdf.open(src_path) as doc:
        if page_pixel_size(doc[0]) != (sheet.width, sheet.height):
            raise ValueError(f"{src_path.name}: размер страницы не совпадает с разметкой")
        result = _draw_cutout(doc[0], sheet, count, size, seed)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(out_path, garbage=3, deflate=True)
    return replace(result, file_name=out_path.name)


def restyle_pdf(
    src_path: Path, sheet: Sheet, out_path: Path, thickness: int = 0, invert: bool = False
) -> Sheet:
    """Копия готовой PDF с другой толщиной линий и/или инверсией цветов; рамки те же.

    thickness: на `2 * thickness` px листа толще (>0) или тоньше (<0), как в растре; только штрихи.
    """
    import pymupdf

    if not thickness and not invert:
        raise ValueError("Нужна толщина линий или инверсия")
    with pymupdf.open(src_path) as doc:
        if page_pixel_size(doc[0]) != (sheet.width, sheet.height):
            raise ValueError(f"{src_path.name}: размер страницы не совпадает с разметкой")
        history = list(sheet.history)
        if thickness:
            thicken_page(doc, doc[0], thickness)
            history.append(f"pdf_thickness:{thickness}")
        if invert:
            invert_page(doc, doc[0])
            history.append("pdf_invert")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(out_path, garbage=3, deflate=True)
    return replace(sheet, file_name=out_path.name, history=history)


def tile_pdf(
    src_path: Path,
    sheet: Sheet,
    out_dir: Path,
    tile_size: tuple[int, int],
    overlap: int = 0,
    min_visible: float = 0.5,
    legend_mode: str = "keep",
    keep_empty: bool = False,
) -> list[Sheet]:
    """Режет PDF на плитки: каждая — отдельная PDF-страница с вырезанным куском листа.

    `sheet` — лист этой PDF с рамками в пикселях при 200 dpi (например, результат `transform_pdf`).
    Плитки выбираются теми же правилами, что в растровом `tiling.tile_sheet`; режим `with-legend`
    приклеивает недостающие образцы полосой под плиткой, тоже вектором. Возвращает листы плиток.
    """
    import pymupdf

    if legend_mode not in LEGEND_MODES:
        raise ValueError(f"legend_mode должен быть одним из {LEGEND_MODES}, получено {legend_mode!r}")
    k = 72 / DPI
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[Sheet] = []
    with pymupdf.open(src_path) as src:
        if len(src) != 1 or page_pixel_size(src[0]) != (sheet.width, sheet.height):
            raise ValueError(f"{src_path.name}: страница не совпадает с разметкой")
        # «Запекаем» /Rotate: дальше работаем в системе показанной страницы без поворота.
        # rotate у show_pdf_page считается против часовой, /Rotate в PDF — по часовой.
        turn = src[0].rotation
        src[0].set_rotation(0)
        flat = pymupdf.open()
        shown = src[0].rect if turn in (0, 180) else pymupdf.Rect(0, 0, src[0].rect.height,
                                                                  src[0].rect.width)
        flat_page = flat.new_page(width=shown.width, height=shown.height)
        flat_page.show_pdf_page(flat_page.rect, src, 0, rotate=(360 - turn) % 360)

        for x0, y0, tile in plan_tiles(sheet, tile_size, overlap, min_visible,
                                       legend_mode=legend_mode, keep_empty=keep_empty):
            placed, strip_h = ([], 0)
            if legend_mode == "with-legend":
                placed, strip_h = plan_legend_strip(tile, sheet)
                if placed:
                    tile = with_legend_strip(tile, placed, strip_h)
            doc = pymupdf.open()
            page = doc.new_page(width=tile.width * k, height=(tile.height) * k)
            body_h = tile.height - strip_h
            page.show_pdf_page(
                pymupdf.Rect(0, 0, tile.width * k, body_h * k), flat, 0,
                clip=pymupdf.Rect(x0 * k, y0 * k, (x0 + tile.width) * k, (y0 + body_h) * k))
            for src_box, px, py, w, h in placed:
                sx, sy = round(src_box.bbox[0]), round(src_box.bbox[1])
                page.show_pdf_page(
                    pymupdf.Rect(px * k, (body_h + py) * k, (px + w) * k, (body_h + py + h) * k),
                    flat, 0, clip=pymupdf.Rect(sx * k, sy * k, (sx + w) * k, (sy + h) * k))
            if page_pixel_size(page) != (tile.width, tile.height):
                raise ValueError(f"{tile.file_name}: страница даёт {page_pixel_size(page)} px, "
                                 f"в разметке {(tile.width, tile.height)}")
            target = out_dir / Path(tile.file_name).with_suffix(".pdf").name
            doc.save(target, garbage=3, deflate=True)
            doc.close()
            results.append(replace(tile, file_name=target.name))
        flat.close()
    return results


def render_overlay(pdf_path: Path, sheet: Sheet, max_side: int = 1800) -> Image.Image:
    """Рендерит PDF при 200 dpi и рисует поверх рамки из разметки (для проверки глазами)."""
    import pymupdf

    import io

    from visualize import draw_sheet

    with pymupdf.open(pdf_path) as doc:
        data = doc[0].get_pixmap(dpi=DPI, alpha=False).tobytes("png")
    image = Image.open(io.BytesIO(data))
    if image.size != (sheet.width, sheet.height):
        raise ValueError(f"{pdf_path.name}: отрендерено {image.size}, в разметке "
                         f"{(sheet.width, sheet.height)}")
    return draw_sheet(image, sheet, max_side)


def output_name(stem: str, rotate: int, factor: float) -> str:
    return f"{stem}__rot{rotate}_x{factor:g}.pdf"


def parse_tile(text: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("размер плитки задаётся как ШИРИНАxВЫСОТА, например 1600x1600")
    return width, height


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--pdf", type=Path, required=True, help="исходная PDF (один лист)")
    p.add_argument("--annotations", nargs="+", type=Path, default=[DEFAULT_ANNOTATIONS],
                   help="файлы COCO или папки с ними (по умолчанию datasets/labeled-dataset/annotations)")
    p.add_argument("--out", type=Path, required=True, help="папка результата: pdf/, annotations.json")
    p.add_argument("--rotate", nargs="*", type=int, default=[], choices=[90, 180, 270],
                   help="углы поворота против часовой; исходная ориентация остаётся всегда")
    p.add_argument("--scale", nargs="*", type=float, default=[],
                   help="коэффициенты масштаба; 1 (без масштаба) остаётся всегда")
    t = p.add_argument_group("плитки (заменяют целый лист)")
    t.add_argument("--tile", type=parse_tile, metavar="ШxВ",
                   help="резать на плитки, размер в px при 200 dpi, например 1600x1600")
    t.add_argument("--overlap", type=int, default=0, help="перекрытие плиток, px")
    t.add_argument("--min-visible", type=float, default=0.5,
                   help="доля площади объекта, которая должна остаться в плитке")
    t.add_argument("--legend-mode", choices=LEGEND_MODES, default="keep",
                   help="keep / require / with-legend: как в augment.py")
    t.add_argument("--keep-empty", action="store_true", help="оставлять плитки без рамок")
    p.add_argument("--cutout", type=int, default=0, metavar="N",
                   help="к каждому листу (плитке) добавить копию с N белыми квадратами")
    p.add_argument("--cutout-size", type=int, nargs=2, default=(20, 60), metavar=("МИН", "МАКС"),
                   help="сторона квадрата, px готового листа (по умолчанию 20 60)")
    p.add_argument("--thickness", type=int, default=0,
                   help="толщина линий: >0 толще, <0 тоньше на 2*K px листа (только штрихи)")
    p.add_argument("--invert", action="store_true", help="инверсия цветов")
    p.add_argument("--no-clean", action="store_true",
                   help="не сохранять версии без искажений (нужны --cutout, --thickness или --invert)")
    p.add_argument("--seed", type=int, default=0, help="seed случайных квадратов")
    p.add_argument("--check", action="store_true",
                   help="сохранить в <out>/check/ превью с рамками для проверки глазами")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    distort = bool(args.cutout or args.thickness or args.invert)
    if args.no_clean and not distort:
        print("--no-clean имеет смысл только вместе с --cutout, --thickness или --invert",
              file=sys.stderr)
        return 2
    stem = args.pdf.stem
    sheets = [s for s in load_sheets(collect_annotation_paths(args.annotations))
              if s.file_name.startswith(stem)]
    if len(sheets) != 1:
        print(f"Для {stem} в разметке найдено листов: {len(sheets)}, нужен ровно один", file=sys.stderr)
        return 2
    base = sheets[0]
    pdf_dir = args.out / "pdf"
    check_dir = args.out / "check"

    def emit(path: Path, sheet: Sheet, results: list[Sheet]) -> None:
        results.append(sheet)
        print(f"PDF: {path.name}  {sheet.width}x{sheet.height} px, рамок {len(sheet.boxes)}")
        if args.check:
            check_dir.mkdir(parents=True, exist_ok=True)
            render_overlay(path, sheet).save(check_dir / f"{path.stem}.png")

    angles = [0, *dict.fromkeys(args.rotate)]
    factors = [1.0, *dict.fromkeys(f for f in args.scale if f != 1)]
    results: list[Sheet] = []
    with tempfile.TemporaryDirectory() as tmp:
        for angle in angles:
            for factor in factors:
                name = output_name(stem, angle, factor)
                whole_dir = Path(tmp) if args.tile else pdf_dir
                whole = transform_pdf(args.pdf, base, whole_dir / name, angle, factor)
                if args.tile:
                    pieces = tile_pdf(whole_dir / name, whole, pdf_dir, args.tile, args.overlap,
                                      args.min_visible, args.legend_mode, args.keep_empty)
                else:
                    pieces = [whole]
                for piece in pieces:
                    path = pdf_dir / piece.file_name
                    if not args.no_clean:
                        emit(path, piece, results)
                    if distort:
                        # Одна «испорченная» копия со всеми искажениями, как в augment.py:
                        # сначала толщина линий и инверсия, потом квадраты (белые и на тёмном).
                        tags = [f"thick{args.thickness}" if args.thickness else "",
                                "inv" if args.invert else "",
                                f"cut{args.cutout}" if args.cutout else ""]
                        target = path.with_name(f"{path.stem}_{'_'.join(t for t in tags if t)}.pdf")
                        current, marked = path, piece
                        if args.thickness or args.invert:
                            current = Path(tmp) / "step.pdf" if args.cutout else target
                            marked = restyle_pdf(path, marked, current, args.thickness, args.invert)
                        if args.cutout:
                            seed = (args.seed + zlib.crc32(piece.file_name.encode())) % 2**32
                            marked = cutout_pdf(current, marked, target, args.cutout,
                                                tuple(args.cutout_size), seed)
                        emit(target, marked, results)
                    if args.no_clean:
                        path.unlink()
    save_coco(results, args.out / "annotations.json")
    print(f"Записано: {len(results)} PDF, {args.out / 'annotations.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
