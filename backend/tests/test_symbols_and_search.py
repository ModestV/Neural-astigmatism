from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from helpers import make_pdf, png_data_url, upload

from app.config import Settings
from app.main import create_app

BBOX = {"x": 100, "y": 100, "w": 60, "h": 40}


@pytest.fixture
def drawing(client) -> dict:
    return upload(client, "plan.pdf", make_pdf(pages=2))


def create_symbol(client, drawing, **overrides) -> dict:
    body = {"name": "Розетка", "category": "Электрика", "drawing_id": drawing["id"],
            "page": 1, "bbox": BBOX} | overrides
    response = client.post("/api/symbols", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def test_symbol_crud(client, drawing):
    symbol = create_symbol(client, drawing, preview_png=png_data_url())
    assert symbol["drawing_name"] == "plan.pdf"
    assert symbol["bbox"] == BBOX
    assert client.get(symbol["preview_url"]).content.startswith(b"\x89PNG")

    updated = client.patch(f"/api/symbols/{symbol['id']}",
                           json={"name": " Розетка двойная ", "category": ""}).json()
    assert updated["name"] == "Розетка двойная"
    assert updated["category"] is None

    assert [s["id"] for s in client.get("/api/symbols").json()] == [symbol["id"]]
    assert client.get("/api/symbols", params={"category": "Нет такой"}).json() == []

    assert client.delete(f"/api/symbols/{symbol['id']}").status_code == 204
    assert client.get(f"/api/symbols/{symbol['id']}").status_code == 404


def test_symbol_without_preview(client, drawing):
    symbol = create_symbol(client, drawing)
    assert symbol["preview_url"] is None


def test_symbol_bbox_is_clipped_to_page(client, drawing):
    symbol = create_symbol(client, drawing, bbox={"x": -10, "y": 500, "w": 50, "h": 200})
    assert symbol["bbox"] == {"x": 0, "y": 500, "w": 40, "h": 95}


@pytest.mark.parametrize(
    ("overrides", "status"),
    [
        ({"bbox": {"x": 900, "y": 0, "w": 10, "h": 10}}, 422),  # вне страницы
        ({"bbox": {"x": 0, "y": 0, "w": 0, "h": 10}}, 422),  # нулевая ширина
        ({"page": 3}, 422),
        ({"name": "   "}, 422),
        ({"preview_png": "data:image/png;base64,bm90IGEgcG5n"}, 422),
        ({"drawing_id": "0123456789ab"}, 404),
    ],
)
def test_symbol_validation(client, drawing, overrides, status):
    body = {"name": "X", "drawing_id": drawing["id"], "page": 1, "bbox": BBOX} | overrides
    assert client.post("/api/symbols", json=body).status_code == status


def test_search_by_region_with_stub(client, drawing):
    query = {"drawing_id": drawing["id"], "page": 1, "bbox": BBOX}
    run = client.post("/api/search", json={"drawing_id": drawing["id"], "query": query}).json()

    assert run["status"] == "done"
    assert run["detector"] == "stub"
    # заглушка находит только сам образец — на его же странице
    assert [(d["page"], d["bbox"], d["symbol_id"]) for d in run["detections"]] == [(1, BBOX, None)]

    only_page_2 = client.post(
        "/api/search", json={"drawing_id": drawing["id"], "pages": [2], "query": query}
    ).json()
    assert only_page_2["detections"] == []

    assert client.get(f"/api/runs/{run['id']}").json() == run
    history = client.get("/api/runs").json()
    assert [r["id"] for r in history] == [only_page_2["id"], run["id"]]
    assert history[1]["detections_count"] == 1
    assert history[1]["has_region_query"] is True


def test_search_by_library_symbols(client, drawing):
    symbol = create_symbol(client, drawing)
    run = client.post(
        "/api/search", json={"drawing_id": drawing["id"], "symbol_ids": [symbol["id"]]}
    ).json()
    assert [d["symbol_id"] for d in run["detections"]] == [symbol["id"]]


def test_search_validation(client, drawing):
    no_template = {"drawing_id": drawing["id"]}
    assert client.post("/api/search", json=no_template).status_code == 422

    bad_page = {"drawing_id": drawing["id"], "pages": [5], "symbol_ids": ["0123456789ab"]}
    assert client.post("/api/search", json=bad_page).status_code == 422

    unknown_symbol = {"drawing_id": drawing["id"], "symbol_ids": ["0123456789ab"]}
    assert client.post("/api/search", json=unknown_symbol).status_code == 404


def test_symbols_survive_drawing_deletion(client, drawing):
    symbol = create_symbol(client, drawing)
    other = upload(client, "other.pdf", make_pdf(pages=1))
    client.delete(f"/api/drawings/{drawing['id']}")

    orphan = client.get(f"/api/symbols/{symbol['id']}").json()
    assert orphan["drawing_name"] is None

    response = client.post(
        "/api/search", json={"drawing_id": other["id"], "symbol_ids": [symbol["id"]]}
    )
    assert response.status_code == 409


def test_failing_detector_is_reported(client, drawing):
    class Boom:
        name = "boom"
        description = ""

        def detect(self, *args):
            raise RuntimeError("всё сломалось")

    client.app.state.detector = Boom()
    query = {"drawing_id": drawing["id"], "page": 1, "bbox": BBOX}
    run = client.post("/api/search", json={"drawing_id": drawing["id"], "query": query}).json()
    assert run["status"] == "failed"
    assert "всё сломалось" in run["message"]


def test_stats(client, drawing):
    create_symbol(client, drawing)
    assert client.get("/api/stats").json() == {"drawings": 1, "symbols": 1, "runs": 0}


def test_serves_built_frontend(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<div id=root></div>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    settings = Settings(data_dir=tmp_path / "data", frontend_dist=dist, max_upload_mb=1,
                        detector="stub")
    with TestClient(create_app(settings)) as client:
        assert client.get("/").text == "<div id=root></div>"
        assert client.get("/drawings/0123456789ab").text == "<div id=root></div>"
        assert client.get("/assets/app.js").text == "console.log(1)"
        assert client.get("/api/nope").status_code == 404
        assert client.get("/api/health").status_code == 200
        assert client.get("/docs").status_code == 200
