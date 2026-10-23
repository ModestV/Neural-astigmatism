"""Импорт COCO, сохранность координат, ревизии и локальный редактор."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

ROOT = Path(__file__).resolve().parents[2] / "experiments/vozderjus/symbol_matching"
sys.path.insert(0, str(ROOT))

from annotation_core import (  # noqa: E402
    AnnotationStore,
    ConflictError,
    Workspace,
    discover_workspace,
    export_experiment,
    import_coco,
)
from annotate import make_server  # noqa: E402


@pytest.fixture
def fixture_data(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    drawings = tmp_path / "drawings"
    drawings.mkdir()
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=200, height=100)
        page.draw_circle((20, 20), 5)
        page.set_rotation(270)
        pdf.save(drawings / "лист.pdf")
    payload = {
        "images": [
            {"id": 1, "file_name": "лист[1]000000000.jpeg-1.jpg", "width": 500, "height": 1000}
        ],
        "categories": [{"id": 5, "name": "legend_item"}, {"id": 9, "name": "object"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 5,
                "bbox": [50, 50, 50, 50],
                "attributes": {"legend_id": "1", "rotation": 0},
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 9,
                "bbox": [250, 250, 50, 50],
                "attributes": {"legend_id": "1", "rotation": 0},
            },
        ],
    }
    coco = drawings / "instances.json"
    coco.write_text(json.dumps(payload), encoding="utf-8")
    workspace = import_coco(discover_workspace(drawings), coco, drawings)
    return drawings, coco, workspace, payload


def test_coco_mapping_rotation_and_reimport(fixture_data):
    drawings, coco, workspace, _ = fixture_data
    page = workspace.pages[0]
    assert page.rotation == 270
    assert (page.width, page.height) == (100, 200)
    assert page.annotations[0].bbox.as_list() == [10, 10, 10, 10]
    assert page.annotations[1].bbox.as_list() == [50, 50, 10, 10]
    assert page.annotations[0].legend
    assert not page.annotations[1].legend
    assert page.annotations[0].symbol == page.annotations[1].symbol
    assert not page.reviewed_symbols
    assert len(import_coco(workspace, coco, drawings).pages[0].annotations) == 2


def test_equal_legend_numbers_on_different_drawings_are_separate(fixture_data):
    drawings, coco, workspace, payload = fixture_data
    import pymupdf

    with pymupdf.open(drawings / "лист.pdf") as pdf:
        pdf.save(drawings / "другой.pdf")
    workspace = discover_workspace(drawings)
    payload["images"].append(
        {"id": 2, "file_name": "другой000000000.jpeg-1.jpg", "width": 500, "height": 1000}
    )
    payload["annotations"].append(
        {
            "id": 3,
            "image_id": 2,
            "category_id": 9,
            "bbox": [250, 250, 50, 50],
            "attributes": {"legend_id": "1"},
        }
    )
    coco.write_text(json.dumps(payload), encoding="utf-8")
    result = import_coco(workspace, coco, drawings)
    assert len(result.symbols) == 2
    assert result.pages[0].annotations[0].symbol != result.pages[1].annotations[0].symbol


@pytest.mark.parametrize("kind", ["dimensions", "unknown_pdf", "missing_link", "crowd"])
def test_coco_rejects_ambiguous_inputs(fixture_data, kind):
    drawings, coco, _, payload = fixture_data
    if kind == "dimensions":
        payload["images"][0]["width"] = 800
    if kind == "unknown_pdf":
        payload["images"][0]["file_name"] = "missing.jpg"
    if kind == "missing_link":
        payload["annotations"][0]["attributes"] = {}
    if kind == "crowd":
        payload["annotations"][0]["iscrowd"] = 1
    coco.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        import_coco(discover_workspace(drawings), coco, drawings)


def test_unreviewed_exports_without_metrics_and_template_is_explicit(fixture_data):
    _, _, workspace, _ = fixture_data
    workspace.symbols[0].enabled = True
    with pytest.raises(ValueError, match="образец"):
        export_experiment(workspace)
    workspace.pages[0].annotations[0].template = True
    config, warnings = export_experiment(workspace, workspace.pages[0].id)
    assert len(config.templates) == 1
    assert config.targets[0].ground_truth is None
    assert warnings
    workspace.pages[0].reviewed_symbols = [workspace.symbols[0].id]
    config, warnings = export_experiment(workspace)
    assert len(config.targets[0].ground_truth) == 2
    assert len(config.targets[0].ignore_regions) == 1
    assert not warnings


def test_revision_backup_and_source_integrity(fixture_data, tmp_path):
    drawings, _, workspace, _ = fixture_data
    store = AnnotationStore(tmp_path / "annotations/workspace.json", drawings)
    first = store.save(workspace)
    assert first.revision == 1
    second = store.save(first)
    assert second.revision == 2
    assert len(list((store.path.parent / ".annotation_backups").glob("*.json"))) == 1
    with pytest.raises(ConflictError):
        store.save(first)
    assert store.load().revision == 2
    (drawings / "лист.pdf").write_bytes((drawings / "лист.pdf").read_bytes() + b"\n")
    with pytest.raises(ValueError, match="изменился"):
        store.save(second)
    assert store.load().revision == 2


def test_out_of_bounds_box_cannot_be_saved(fixture_data):
    _, _, workspace, _ = fixture_data
    data = workspace.model_dump()
    data["pages"][0]["annotations"][0]["bbox"]["w"] = 999
    with pytest.raises(ValueError):
        Workspace.model_validate(data)


def test_editor_http_save_export_and_auth(fixture_data, tmp_path):
    drawings, _, workspace, _ = fixture_data
    store = AnnotationStore(tmp_path / "workspace.json", drawings)
    workspace = store.save(workspace)
    server = make_server(store, tmp_path / "cache", tmp_path / "exports")
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    host, token = server.editor_url.split("#")

    def request(route, body=None, auth=True, origin=None):
        headers = {"Authorization": "Bearer " + token} if auth else {}
        if origin:
            headers["Origin"] = origin
        data = json.dumps(body).encode() if body is not None else None
        with urlopen(
            Request(host.rstrip("/") + route, data=data, headers=headers), timeout=10
        ) as r:
            return r.read()

    try:
        assert b'<html lang="ru">' in request("/", auth=False)
        with pytest.raises(HTTPError) as error:
            request("/api/state", auth=False)
        assert error.value.code == 403
        with pytest.raises(HTTPError) as error:
            request("/api/state", origin="https://example.com")
        assert error.value.code == 403
        body = json.loads(request("/api/state"))["workspace"]
        body["symbols"][0]["enabled"] = True
        body["pages"][0]["annotations"][0]["template"] = True
        body["pages"][0]["reviewed_symbols"] = [body["symbols"][0]["id"]]
        assert json.loads(request("/api/save", body))["revision"] == 2
        with pytest.raises(HTTPError) as error:
            request("/api/save", body)
        assert error.value.code == 409
        exported = json.loads(request("/api/export", {"page_id": body["pages"][0]["id"]}))
        config = json.loads(Path(exported["path"]).read_text())
        assert len(config["targets"][0]["ground_truth"]) == 2
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
