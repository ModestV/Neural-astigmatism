"""Полный датасет supertest: ручная разметка + предсказания там, где разметки нет.

Ручные аннотации копируются без изменений (включая rotation). Неразмеченные планы
получают предсказания шаблонного метода (outputs/research/fill), неразмеченные легенды
получают автоматический разбор (outputs/research/legend_auto). У машинных аннотаций
заполнен атрибут confidence, у ручных он пуст.

    python build_full_dataset.py            # пишет outputs/research/full_dataset/
    python build_full_dataset.py --upload "supertest full"   # и грузит в CVAT (env CVAT_*)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import common

DST = common.OUT / "full_dataset"


def build() -> dict:
    src = json.loads(common.ANN.read_text())
    cats = {c["name"]: c["id"] for c in src["categories"]}
    images = {im["file_name"]: im for im in src["images"]}
    anns = [
        {**a, "attributes": {**a.get("attributes", {}), "confidence": ""}}
        for a in src["annotations"]
    ]
    manual = {a["image_id"] for a in src["annotations"]}
    next_id = max(a["id"] for a in anns) + 1

    def add(image_id: int, cat: str, bbox: list[float], lid: str, score: float) -> None:
        nonlocal next_id
        anns.append(
            dict(
                id=next_id, image_id=image_id, category_id=cats[cat], bbox=[round(v, 2) for v in bbox],
                area=round(bbox[2] * bbox[3], 2), iscrowd=0, segmentation=[],
                attributes=dict(legend_id=lid, occluded=False, rotation=0.0, confidence=f"{score:.3f}"),
            )
        )
        next_id += 1

    for s in common.load_sheets():
        im = images[s.file_name]
        if im["id"] in manual:
            continue
        if s.is_legend:
            for it in json.loads((common.OUT / "legend_auto" / f"{s.family}.json").read_text()):
                add(im["id"], "legend_item", it["bbox"], it["legend_id"], float(it.get("score", 1.0)))
        else:
            # ручная проверка (label_tool): только принятые рамки
            relabel = common.OUT / "relabel" / f"{Path(s.file_name).stem}.json"
            for b in json.loads(relabel.read_text()):
                if b["status"] == "ok":
                    add(im["id"], "object", b["bbox"], b["legend_id"], b["score"])
    return {**src, "annotations": anns}


def save(coco: dict) -> Path:
    (DST / "annotations").mkdir(parents=True, exist_ok=True)
    (DST / "annotations" / "instances_default.json").write_text(json.dumps(coco, ensure_ascii=False))
    img_dir = DST / "images" / "default"
    img_dir.mkdir(parents=True, exist_ok=True)
    for im in coco["images"]:
        if not (img_dir / im["file_name"]).exists():
            shutil.copy2(common.IMAGES / im["file_name"], img_dir / im["file_name"])
    zip_path = shutil.make_archive(str(common.OUT / "supertest_full"), "zip", DST)
    return Path(zip_path)


def upload(coco: dict, project_name: str) -> None:
    import os
    import tempfile

    from cvat_sdk import make_client
    from cvat_sdk.api_client import models

    from symspot.io.cvat_coco import cvat_labels

    labels = cvat_labels()
    for lab in labels:  # ручная разметка несет атрибуты occluded/rotation: rotation CVAT берет сам
        names = {a["name"] for a in lab["attributes"]}
        for extra in ("legend_id", "confidence"):
            if extra not in names:
                lab["attributes"].append(
                    {"name": extra, "input_type": "text", "mutable": False, "default_value": "", "values": [""]}
                )
    url = os.environ.get("CVAT_URL", "http://localhost:8080")
    with make_client(url, credentials=(os.environ["CVAT_USER"], os.environ["CVAT_PASSWORD"])) as client:
        project = client.projects.create(models.ProjectWriteRequest(name=project_name, labels=labels))
        print("project", project.id)
        for im in sorted(coco["images"], key=lambda i: i["file_name"]):
            spec = models.TaskWriteRequest(name=Path(im["file_name"]).stem, project_id=project.id)
            task = client.tasks.create_from_data(
                spec, resources=[common.IMAGES / im["file_name"]], data_params={"image_quality": 95}
            )
            one = {
                **coco,
                "images": [{**im, "id": 1}],
                "annotations": [
                    {**a, "image_id": 1} for a in coco["annotations"] if a["image_id"] == im["id"]
                ],
            }
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "instances_default.json"
                p.write_text(json.dumps(one, ensure_ascii=False))
                task.import_annotations("COCO 1.0", p)
            print(task.id, im["file_name"], len(one["annotations"]), flush=True)


def replace_plans(coco: dict, project_name: str) -> None:
    """Заменить аннотации неразмеченных пользователем планов в задачах существующего проекта."""
    import os
    import tempfile

    from cvat_sdk import make_client

    src_ids = {a["image_id"] for a in json.loads(common.ANN.read_text())["annotations"]}
    url = os.environ.get("CVAT_URL", "http://localhost:8080")
    with make_client(url, credentials=(os.environ["CVAT_USER"], os.environ["CVAT_PASSWORD"])) as client:
        project = next(p for p in client.projects.list() if p.name == project_name)
        tasks = {t.name: t for t in project.get_tasks()}
        for im in coco["images"]:
            if im["id"] in src_ids or "legend" in im["file_name"]:
                continue
            task = tasks[Path(im["file_name"]).stem]
            one = {
                **coco,
                "images": [{**im, "id": 1}],
                "annotations": [
                    {**a, "image_id": 1} for a in coco["annotations"] if a["image_id"] == im["id"]
                ],
            }
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "instances_default.json"
                p.write_text(json.dumps(one, ensure_ascii=False))
                task.import_annotations("COCO 1.0", p, import_mode="replace")
            print(task.id, im["file_name"], len(one["annotations"]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", default="")
    ap.add_argument("--replace-plans", default="", help="имя существующего проекта CVAT")
    a = ap.parse_args()
    coco = build()
    print("zip", save(coco), "annotations", len(coco["annotations"]))
    if a.upload:
        upload(coco, a.upload)
    if a.replace_plans:
        replace_plans(coco, a.replace_plans)
