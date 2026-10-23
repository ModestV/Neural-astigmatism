"""Воспроизводимое сравнение трёх методов по исходной COCO-разметке пользователя."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from time import perf_counter

from algorithms import features, template_matching
from annotation_core import AnnotationPage, Workspace, discover_workspace, import_coco
from evaluation import evaluate, filter_detections, metrics
from models import Box, Settings, TargetSpec, Truth
from pdf_io import checksum, crop_template, pdf_path, render_page, save_image, write_json
from run_benchmark import create_output, overlay, versions

BASE = Path(__file__).resolve().parent
CALIBRATION = {"01_riga_lighting.pdf", "02_renee_house.pdf", "06_camden_lighting.pdf"}
ALGORITHMS = ["template", "orb", "sift"]
THRESHOLDS = {
    "template": [0.75, 0.8, 0.85, 0.9, 0.95],
    "orb": [0.35, 0.5, 0.65, 0.8, 0.95],
    "sift": [0.35, 0.5, 0.65, 0.8, 0.95],
}


def load_dataset(drawings: Path) -> Workspace:
    """Использовать первичные COCO, не отметки и выборы интерфейса разметки."""
    workspace = discover_workspace(drawings)
    for source in sorted(drawings.rglob("*.json")):
        data = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(data, dict) and {"images", "annotations", "categories"} <= data.keys():
            workspace = import_coco(workspace, source, drawings)
    return workspace


def page_target(page: AnnotationPage) -> TargetSpec:
    return TargetSpec(
        file=page.file,
        page=page.page,
        ground_truth=[
            Truth(symbol=a.symbol, bbox=a.bbox, legend=a.legend) for a in page.annotations
        ],
        ignore_regions=[a.bbox for a in page.annotations if a.legend],
    )


def micro(results: list[dict]) -> dict:
    return metrics(*(sum(r[k] for r in results) for k in ("tp", "fp", "fn")))


def score_record(record: dict, threshold: float, target: TargetSpec) -> tuple[list, dict]:
    from models import Detection

    predictions = [
        Detection.model_validate(d) for d in record["detections"] if d["score"] >= threshold
    ]
    return predictions, evaluate(predictions, target, 0.5, False)


def select_threshold(records: list[dict], targets: dict, algorithm: str) -> tuple[float, list]:
    """Выбрать по micro-F1 только на calibration; при равенстве — precision и строгий порог."""
    rows = []
    for threshold in THRESHOLDS[algorithm]:
        scores = [
            score_record(r, threshold, targets[r["file"]])[1]
            for r in records
            if r["algorithm"] == algorithm and r["split"] == "calibration"
        ]
        rows.append({"algorithm": algorithm, "threshold": threshold, **micro(scores)})
    best = max(rows, key=lambda r: (r["f1"] or 0, r["precision"] or 0, r["threshold"]))
    return best["threshold"], rows


def process_page(
    page: AnnotationPage, args: argparse.Namespace, settings: Settings, output: Path, index: int,
) -> list[dict]:
    """Сохранить кандидатов; оценка порогов не повторяет поиск и не влияет на него."""
    image, metadata = render_page(
        pdf_path(args.drawings_dir, page.file),
        page.page,
        settings.dpi,
        args.cache_dir,
        settings.max_pixels,
    )
    target = page_target(page)
    prototypes = [a for a in page.annotations if a.legend]
    prepared = []
    for query in prototypes:
        try:
            patch = crop_template(image, query.bbox, metadata)
            failure = None
        except ValueError as error:
            patch, failure = None, str(error)
        prepared.append((query, patch, failure))
    records = []
    # Перестановка порядка между страницами снижает систематический эффект прогрева.
    order = ALGORITHMS[index % 3 :] + ALGORITHMS[: index % 3]
    for algorithm in order:
        started = perf_counter()
        target_features = None
        if algorithm != "template":
            target_features = features.prepare_target(image, settings, algorithm)
        extraction_seconds = perf_counter() - started
        detections, diagnostics = [], []
        for query, patch, failure in prepared:
            query_started = perf_counter()
            if failure:
                found, info = [], {"status": "template_too_small", "reason": failure}
            elif algorithm == "template":
                found, info = template_matching.detect(
                    image, patch, query.id, query.symbol, settings
                )
            else:
                found, info = features.detect(
                    image,
                    patch,
                    query.id,
                    query.symbol,
                    settings,
                    algorithm,
                    target_features=target_features,
                )
            zoom = metadata["pixels_per_unit"]
            for item in found:
                item.bbox = Box(
                    x=item.bbox.x / zoom,
                    y=item.bbox.y / zoom,
                    w=item.bbox.w / zoom,
                    h=item.bbox.h / zoom,
                )
            found = filter_detections(found, target.ignore_regions)
            detections.extend(found)
            diagnostics.append(
                {
                    "template_id": query.id,
                    "symbol": query.symbol,
                    "query_seconds": perf_counter() - query_started,
                    "detections": len(found),
                    **info,
                }
            )
        seconds = perf_counter() - started
        record = {
            "file": page.file,
            "page": page.page,
            "algorithm": algorithm,
            "split": "calibration" if Path(page.file).name in CALIBRATION else "test",
            "source": metadata,
            "queries": len(prototypes),
            "search_seconds": seconds,
            "target_feature_seconds": extraction_seconds,
            "diagnostics": diagnostics,
            "detections": [d.model_dump() for d in detections],
        }
        records.append(record)
        name = f"{Path(page.file).stem}_{algorithm}"
        write_json(output / "raw" / f"{name}.json", record)
        print(
            f"  {algorithm}: {len(prototypes)} образцов, {seconds:.2f} с, "
            f"{len(detections)} кандидатов",
            flush=True,
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drawings-dir", type=Path, default=BASE / "drawings")
    parser.add_argument("--cache-dir", type=Path, default=BASE / "cache")
    parser.add_argument("--output-dir", type=Path, default=BASE / "results")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    import cv2

    cv2.setNumThreads(1)
    settings = Settings(
        dpi=args.dpi,
        features=20000,
        scales=[0.5, 0.75, 1, 1.25, 1.5, 2],
        angles=[0, 90, 180, 270],
        template_threshold=0.75,
        min_inlier_ratio=0.35,
        max_detections=200,
    )
    output = create_output(args.output_dir, "study")
    workspace = load_dataset(args.drawings_dir)
    pages = [p for p in workspace.pages if p.annotations]
    # Образцы определяются только из легенды, не из позиций искомых объектов.
    for page in pages:
        legend = {a.symbol for a in page.annotations if a.legend}
        objects = {a.symbol for a in page.annotations if not a.legend}
        if objects - legend:
            raise ValueError(f"Для объектов нет образца в легенде: {page.file}")
    targets = {p.file: page_target(p) for p in pages}
    manifest = {
        "settings": settings.model_dump(),
        "versions": versions(),
        "threshold_grid": THRESHOLDS,
        "calibration_files": sorted(CALIBRATION),
        "dataset": [
            {
                "file": p.file,
                "sha256": p.sha256,
                "split": "calibration" if Path(p.file).name in CALIBRATION else "test",
                "objects": sum(not a.legend for a in p.annotations),
                "queries": sum(a.legend for a in p.annotations),
            }
            for p in pages
        ],
        "excluded_files": [p.file for p in workspace.pages if not p.annotations],
        "coco_sources": workspace.imports,
        "dataset_warnings": workspace.warnings,
        "code_sha256": {
            str(p.relative_to(BASE)): checksum(p)
            for p in sorted(BASE.rglob("*.py"))
            if not {"results", "cache", "drawings"}.intersection(p.parts)
        },
        "protocol": {
            "ground_truth": "исходная COCO-разметка предоставлена пользователем как эталон",
            "queries": "все legend_item, включая классы без object; словарь отдельный для каждого PDF",
            "matching": "один к одному, тот же symbol, IoU >= 0.5, score descending greedy",
            "ignore": "все рамки легенды, по попаданию центра; источник запроса исключён",
            "time": "поиск всех образцов листа; признаки страницы выделяются один раз; "
            "PDF-рендер, оценка и запись файлов не включены",
            "selection": "micro-F1 на calibration, затем precision, затем более строгий порог",
            "limitations": "один проход времени, осевые bbox, 4 угла, 6 масштабов, "
            "кандидаты до 200 на образец; алгоритмы без обучения",
        },
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "status.json", {"status": "running"})
    print(f"Результаты: {output}", flush=True)
    calibration, test = [], []
    calibration_pages = [p for p in pages if Path(p.file).name in CALIBRATION]
    test_pages = [p for p in pages if Path(p.file).name not in CALIBRATION]
    for i, page in enumerate(calibration_pages):
        print(f"CALIBRATION {i + 1}/{len(calibration_pages)} {page.file}", flush=True)
        calibration.extend(process_page(page, args, settings, output, i))
    frozen, sweep = {}, []
    for algorithm in ALGORITHMS:
        threshold, rows = select_threshold(calibration, targets, algorithm)
        frozen[algorithm] = threshold
        sweep.extend(rows)
    write_json(output / "frozen_thresholds.json", frozen)
    write_json(output / "calibration_sweep.json", sweep)
    print(f"Пороги зафиксированы до оценки test: {frozen}", flush=True)
    for i, page in enumerate(test_pages):
        print(f"TEST {i + 1}/{len(test_pages)} {page.file}", flush=True)
        test.extend(process_page(page, args, settings, output, i + len(calibration_pages)))
    records = calibration + test
    rows = []
    for record in records:
        predictions, score = score_record(
            record, frozen[record["algorithm"]], targets[record["file"]]
        )
        evaluated = {
            **record,
            "threshold": frozen[record["algorithm"]],
            "metrics": score,
            "detections": [d.model_dump() for d in predictions],
        }
        stem = f"{Path(record['file']).stem}_{record['algorithm']}"
        write_json(output / "evaluated" / f"{stem}.json", evaluated)
        rows.append(
            {
                "file": Path(record["file"]).name,
                "split": record["split"],
                "algorithm": record["algorithm"],
                "threshold": evaluated["threshold"],
                "queries": record["queries"],
                "search_seconds": record["search_seconds"],
                "predictions": len(predictions),
                **{k: v for k, v in score.items() if k != "per_class"},
            }
        )
        # Изображения нужны для проверки ошибок, не для подсчёта качества.
        image, metadata = render_page(
            pdf_path(args.drawings_dir, record["file"]),
            record["page"],
            settings.dpi,
            args.cache_dir,
            settings.max_pixels,
        )
        save_image(
            output / "overlays" / f"{stem}.png",
            overlay(image, predictions, targets[record["file"]], metadata["pixels_per_unit"]),
        )
    summary = {}
    for split in ("calibration", "test", "all"):
        summary[split] = {}
        for algorithm in ALGORITHMS:
            selected = [
                r
                for r in rows
                if r["algorithm"] == algorithm and (split == "all" or r["split"] == split)
            ]
            if not selected:
                continue
            summary[split][algorithm] = {
                **micro(selected),
                "pages": len(selected),
                "queries": sum(r["queries"] for r in selected),
                "total_seconds": sum(r["search_seconds"] for r in selected),
                "median_seconds_per_page": statistics.median(r["search_seconds"] for r in selected),
                "seconds_per_query": sum(r["search_seconds"] for r in selected)
                / sum(r["queries"] for r in selected),
            }
    with (output / "per_page.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "summary.json", summary)
    write_json(output / "status.json", {"status": "complete"})
    print(json.dumps(summary["test"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
