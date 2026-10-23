"""Автономный запуск исследования, инвентаризация PDF и выбор образцов."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from evaluation import evaluate, filter_detections, ignored_regions, metrics
from models import Box, Experiment, Settings
from pdf_io import (
    checksum,
    crop_template,
    inspect_pdfs,
    pdf_path,
    render_page,
    save_image,
    validate_box,
    write_json,
)

BASE = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "prepare", "select-template", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--drawings-dir", type=Path, default=BASE / "drawings")
        sub.add_argument("--output-dir", type=Path, default=BASE / "results")
        if name != "inspect":
            sub.add_argument("--cache-dir", type=Path, default=BASE / "cache")
            sub.add_argument("--dpi", type=int, default=150)
            sub.add_argument("--max-pixels", type=int, default=40_000_000)
        if name in ("prepare", "select-template"):
            sub.add_argument("--file", required=True, help="Путь PDF относительно drawings")
            sub.add_argument("--page", type=int, default=1)
        if name == "select-template":
            sub.add_argument("--symbol", required=True, help="Идентификатор класса")
            sub.add_argument("--template-id", default="query_1")
            sub.add_argument("--config-out", type=Path, required=True)
            sub.add_argument(
                "--bbox",
                type=float,
                nargs=4,
                metavar=("X", "Y", "W", "H"),
                help="Рамка в единицах PDF; без параметра открывается выбор мышью",
            )
        if name == "run":
            sub.add_argument("--config", type=Path, required=True)
            sub.add_argument(
                "--algorithms",
                nargs="+",
                choices=["template", "orb", "sift"],
                default=["template", "orb", "sift"],
            )
            sub.add_argument("--angles", nargs="+", type=float, default=[0, 90, 180, 270])
            sub.add_argument("--scales", nargs="+", type=float, default=[1])
            sub.add_argument("--template-threshold", type=float, default=0.85)
            sub.add_argument("--ratio", type=float, default=0.75)
            sub.add_argument("--min-inliers", type=int, default=4)
            sub.add_argument("--min-inlier-ratio", type=float, default=0.5)
            sub.add_argument("--ransac-px", type=float, default=3)
            sub.add_argument("--features", type=int, default=5000)
            sub.add_argument("--angle-tolerance", type=float, default=10)
            sub.add_argument("--scale-tolerance", type=float, default=0.15)
            sub.add_argument("--nms-iou", type=float, default=0.3)
            sub.add_argument("--evaluation-iou", type=float, default=0.5)
            sub.add_argument("--max-detections", type=int, default=200)
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--include-legend", action="store_true")
    return parser


def create_output(parent: Path, label: str) -> Path:
    folder = parent / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{label}_{uuid4().hex[:8]}"
    )
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def versions() -> dict:
    result = {"python": sys.version, "platform": platform.platform()}
    for name in ("opencv-python", "numpy", "pymupdf", "pydantic"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def overlay(image, detections, target, scale: float):
    import cv2

    preview = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for gt in target.ground_truth or []:
        x, y, w, h = [round(v * scale) for v in gt.bbox.as_list()]
        cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 120, 0), 2)
    for item in detections:
        x, y, w, h = [round(v * scale) for v in item.bbox.as_list()]
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 0, 220), 2)
        label = f"{item.template_id} {item.score:.2f}"
        cv2.putText(
            preview,
            label.encode("ascii", "replace").decode(),
            (x, max(12, y - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 180),
            1,
        )
    return preview


def run(args: argparse.Namespace) -> Path:
    import cv2

    from algorithms import features, template_matching
    from algorithms.common import nms

    settings = Settings.model_validate(
        {key: getattr(args, key) for key in Settings.model_fields if hasattr(args, key)}
    )
    experiment = Experiment.model_validate_json(args.config.read_text(encoding="utf-8"))
    # Сначала проверяем все пути, чтобы опечатка не дала неполный «успешный» запуск.
    for item in [*experiment.templates, *experiment.targets]:
        pdf_path(args.drawings_dir, item.file)
    known_classes = {t.symbol for t in experiment.templates}
    for target in experiment.targets:
        if any(t.symbol not in known_classes for t in target.ground_truth or []):
            raise ValueError("В ground_truth есть класс без образца в templates")
    cv2.setNumThreads(1)
    cv2.setRNGSeed(settings.seed)
    output = create_output(args.output_dir, "benchmark")
    write_json(
        output / "manifest.json",
        {
            "settings": settings.model_dump(),
            "experiment": experiment.model_dump(),
            "algorithms": args.algorithms,
            "versions": versions(),
            "code_sha256": {
                str(p.relative_to(BASE)): checksum(p)
                for p in sorted(BASE.rglob("*.py"))
                if "cache" not in p.parts and "results" not in p.parts
            },
            "coordinate_system": "видимая страница PDF, x вправо, y вниз, единицы PDF",
            "metric_protocol": "score-descending greedy, same class, max IoU, one-to-one",
            "note": "Без ground_truth метрики null; score разных алгоритмов несопоставимы",
        },
    )
    write_json(output / "status.json", {"status": "running"})
    records = []
    try:
        prepared = []
        for index, specification in enumerate(experiment.templates):
            image, meta = render_page(
                pdf_path(args.drawings_dir, specification.file),
                specification.page,
                settings.dpi,
                args.cache_dir,
                settings.max_pixels,
            )
            patch = crop_template(image, specification.bbox, meta)
            prepared.append((specification, patch, meta))
            save_image(output / f"template_{index + 1}.png", patch)
        for page_index, target in enumerate(experiment.targets):
            print(
                f"Страница {page_index + 1}/{len(experiment.targets)}: "
                f"{target.file}, лист {target.page}",
                flush=True,
            )
            image, meta = render_page(
                pdf_path(args.drawings_dir, target.file),
                target.page,
                settings.dpi,
                args.cache_dir,
                settings.max_pixels,
            )
            for region in [
                *target.ignore_regions,
                *(t.bbox for t in target.ground_truth or []),
            ]:
                validate_box(region, meta)
            zoom = meta["pixels_per_unit"]
            for algorithm in dict.fromkeys(args.algorithms):
                started = perf_counter()
                detections, diagnostics = [], []
                for spec, patch, _ in prepared:
                    if algorithm == "template":
                        found, info = template_matching.detect(
                            image,
                            patch,
                            spec.id,
                            spec.symbol,
                            settings,
                        )
                    else:
                        found, info = features.detect(
                            image,
                            patch,
                            spec.id,
                            spec.symbol,
                            settings,
                            algorithm,
                        )
                    for d in found:
                        d.bbox = Box(
                            x=d.bbox.x / zoom,
                            y=d.bbox.y / zoom,
                            w=d.bbox.w / zoom,
                            h=d.bbox.h / zoom,
                        )
                    detections.extend(found)
                    diagnostics.append(
                        {"template_id": spec.id, "detections_before_exclusions": len(found), **info}
                    )
                detections = nms(detections, settings.nms_iou, settings.max_detections)
                regions = ignored_regions(target, settings.include_legend)
                # Автоматически исключаем источник запроса на той же странице.
                for spec, _, _ in prepared:
                    if spec.file == target.file and spec.page == target.page:
                        regions.append(spec.bbox)
                effective = target.model_copy(update={"ignore_regions": regions})
                before_exclusions = len(detections)
                detections = filter_detections(detections, regions)
                search_seconds = perf_counter() - started
                result = evaluate(
                    detections,
                    effective,
                    settings.evaluation_iou,
                    settings.include_legend,
                )
                record = {
                    "file": target.file,
                    "page": target.page,
                    "algorithm": algorithm,
                    "source": meta,
                    "search_seconds": search_seconds,
                    "template_sources": [m for _, _, m in prepared],
                    "diagnostics": diagnostics,
                    "metrics": result,
                    "detections": [d.model_dump() for d in detections],
                    "ignored_detections": before_exclusions - len(detections),
                    "ignore_regions": [r.model_dump() for r in regions],
                }
                records.append(record)
                prefix = f"{page_index + 1:03d}_{algorithm}"
                write_json(output / f"{prefix}.json", record)
                save_image(output / f"{prefix}.png", overlay(image, detections, target, zoom))
                print(
                    f"  {algorithm}: {len(detections)} обнаружений; "
                    f"{search_seconds:.3f} с; "
                    + ("без эталонной разметки" if result is None else f"F1={result['f1']}"),
                    flush=True,
                )
        summary = {}
        for algorithm in dict.fromkeys(args.algorithms):
            subset = [r for r in records if r["algorithm"] == algorithm]
            labeled = [r for r in subset if r["metrics"] is not None]
            summary[algorithm] = {
                "pages": len(subset),
                "evaluated_pages": len(labeled),
                "micro_metrics": metrics(
                    *(sum(r["metrics"][k] for r in labeled) for k in ("tp", "fp", "fn"))
                )
                if labeled
                else None,
                "search_seconds": sum(r["search_seconds"] for r in subset),
            }
        write_json(output / "summary.json", summary)
        with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
            keys = [
                "file",
                "page",
                "algorithm",
                "detections",
                "search_seconds",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
            ]
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            for record in records:
                row = {k: record[k] for k in ("file", "page", "algorithm", "search_seconds")}
                row["detections"] = len(record["detections"])
                row.update({k: (record["metrics"] or {}).get(k) for k in keys[5:]})
                writer.writerow(row)
        write_json(output / "status.json", {"status": "complete"})
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error": str(error)})
        raise
    return output


def select_template(args: argparse.Namespace) -> Path:
    """Создать конфигурацию поиска по рамке; эталонную разметку не выдумываем."""
    import cv2

    if args.config_out.exists():
        raise ValueError("Конфигурация уже существует; выберите другое имя")
    image, meta = render_page(
        pdf_path(args.drawings_dir, args.file),
        args.page,
        args.dpi,
        args.cache_dir,
        args.max_pixels,
    )
    if args.bbox:
        box = Box(**dict(zip(("x", "y", "w", "h"), args.bbox, strict=True)))
    else:
        display_scale = min(1, 1500 / image.shape[1], 1000 / image.shape[0])
        preview = cv2.resize(image, None, fx=display_scale, fy=display_scale)
        print("Выделите образец мышью; Enter — подтвердить, Esc — отменить.")
        x, y, w, h = cv2.selectROI("Template selection", preview, showCrosshair=True)
        cv2.destroyAllWindows()
        if not w or not h:
            raise ValueError("Выбор образца отменён")
        factor = display_scale * meta["pixels_per_unit"]
        box = Box(x=x / factor, y=y / factor, w=w / factor, h=h / factor)
    crop_template(image, box, meta)
    config = Experiment.model_validate(
        {
            "description": "Образец выбран вручную; ground_truth пока отсутствует",
            "templates": [
                {
                    "id": args.template_id,
                    "symbol": args.symbol,
                    "file": args.file,
                    "page": args.page,
                    "bbox": box.model_dump(),
                }
            ],
            "targets": [{"file": args.file, "page": args.page, "ground_truth": None}],
        }
    )
    write_json(args.config_out, config.model_dump())
    return args.config_out


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "inspect":
            inventory = inspect_pdfs(args.drawings_dir)
            output = create_output(args.output_dir, "inventory")
            write_json(output / "inventory.json", inventory)
            print(f"PDF: {len(inventory)}, страниц: {sum(len(d['pages']) for d in inventory)}")
            if any("error" in item for item in inventory):
                print(f"Есть ошибки чтения. Подробности: {output}")
                return 1
        elif args.command == "prepare":
            settings = Settings(dpi=args.dpi, max_pixels=args.max_pixels)
            image, meta = render_page(
                pdf_path(args.drawings_dir, args.file),
                args.page,
                settings.dpi,
                args.cache_dir,
                settings.max_pixels,
            )
            output = create_output(args.output_dir, "preview")
            save_image(output / "page.png", image)
            write_json(output / "page.json", meta)
        elif args.command == "select-template":
            Settings(dpi=args.dpi, max_pixels=args.max_pixels)
            output = select_template(args)
        else:
            output = run(args)
        print(f"Сохранено: {output}")
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
