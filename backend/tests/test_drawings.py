from __future__ import annotations

import pytest
from helpers import make_dxf, make_pdf, upload

from app.converters import dwg_converter


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["detector"]["name"] == "stub"


def test_upload_pdf(client):
    pdf = make_pdf(pages=2)
    drawing = upload(client, "План этажа.pdf", pdf)

    assert drawing["status"] == "ready"
    assert drawing["format"] == "pdf"
    assert drawing["units"] == "pt"
    assert [p["index"] for p in drawing["pages"]] == [1, 2]
    assert drawing["pages"][0]["width"] == pytest.approx(842)
    assert drawing["pages"][0]["height"] == pytest.approx(595)

    page = drawing["pages"][0]
    image = client.get(page["image_url"])
    assert image.status_code == 200
    assert image.headers["content-type"].startswith("image/svg+xml")
    assert "default-src 'none'" in image.headers["content-security-policy"]
    assert b'viewBox="0 0 842 595"' in image.content

    thumb = client.get(page["thumb_url"])
    assert thumb.content.startswith(b"\x89PNG")

    original = client.get(drawing["original_url"])
    assert original.content == pdf
    assert "attachment" in original.headers["content-disposition"]

    listed = client.get("/api/drawings").json()
    assert [d["id"] for d in listed] == [drawing["id"]]


@pytest.mark.parametrize(
    ("root", "size", "expected_attrs"),
    [
        # нет viewBox — дописываем его по width/height
        ('<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">', (200, 100),
         [b'viewBox="0 0 200 100"']),
        # нет width/height — дописываем по viewBox
        ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="10 20 400 300">', (400, 300),
         [b'width="400"', b'height="300"']),
        # всё есть — размер страницы берём из viewBox, файл не трогаем
        ('<svg xmlns="http://www.w3.org/2000/svg" width="420mm" height="297mm" viewBox="0 0 420 297">',
         (420, 297), [b'width="420mm"']),
        # проценты в width/height заменяем на числа
        ('<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0,0,50,25">',
         (50, 25), [b'width="50"', b'height="25"']),
    ],
)
def test_upload_svg_normalizes_root(client, root, size, expected_attrs):
    svg = f'<?xml version="1.0"?>\n{root}<rect x="1" y="1" width="5" height="5"/></svg>'
    drawing = upload(client, "scheme.svg", svg.encode())

    assert drawing["status"] == "ready"
    assert drawing["units"] == "svg"
    page = drawing["pages"][0]
    assert (page["width"], page["height"]) == pytest.approx(size)
    # для SVG миниатюрой служит сама страница
    assert page["thumb_url"] == page["image_url"]

    served = client.get(page["image_url"]).content
    for attr in expected_attrs:
        assert attr in served
    assert served.count(b"<svg") == 1
    assert b'<rect x="1" y="1" width="5" height="5"/>' in served


def test_upload_svg_without_size_is_error(client):
    drawing = upload(client, "bad.svg", b'<svg xmlns="http://www.w3.org/2000/svg"><g/></svg>')
    assert drawing["status"] == "error"
    assert "размер" in drawing["error"]


def test_upload_dxf(client):
    drawing = upload(client, "plan.dxf", make_dxf())
    assert drawing["status"] == "ready", drawing["error"]
    assert drawing["units"] == "pt"
    assert len(drawing["pages"]) == 1
    page = drawing["pages"][0]
    # 200 × 100 мм + поля 5 мм в масштабе 1:1 → 210 × 110 мм в пунктах
    assert page["width"] == pytest.approx(210 / 25.4 * 72, rel=0.02)
    assert page["height"] == pytest.approx(110 / 25.4 * 72, rel=0.02)
    assert client.get(page["thumb_url"]).content.startswith(b"\x89PNG")


@pytest.mark.skipif(dwg_converter() is not None, reason="установлен конвертер DWG")
def test_upload_dwg_without_converter_keeps_drawing_with_error(client):
    drawing = upload(client, "plan.dwg", b"AC1032 not really a dwg")
    assert drawing["status"] == "error"
    assert "конвертер" in drawing["error"]
    assert drawing["pages"] == []

    again = client.post(f"/api/drawings/{drawing['id']}/reprocess").json()
    assert again["status"] == "error"


def test_upload_corrupt_pdf_is_error(client):
    drawing = upload(client, "broken.pdf", b"%PDF-1.7 definitely broken")
    assert drawing["status"] == "error"
    assert drawing["error"]


def test_upload_rejects_bad_files(client):
    assert client.post("/api/drawings", files={"file": ("a.png", b"x")}).status_code == 415
    assert client.post("/api/drawings", files={"file": ("a.pdf", b"")}).status_code == 400
    too_big = b"0" * (1024 * 1024 + 1)  # в тестах лимит 1 МБ
    assert client.post("/api/drawings", files={"file": ("a.pdf", too_big)}).status_code == 413
    assert client.get("/api/drawings").json() == []


def test_delete_drawing(client):
    drawing = upload(client, "a.pdf", make_pdf(pages=1))
    assert client.delete(f"/api/drawings/{drawing['id']}").status_code == 204
    assert client.get(f"/api/drawings/{drawing['id']}").status_code == 404
    assert client.delete(f"/api/drawings/{drawing['id']}").status_code == 404


@pytest.mark.parametrize("bad_id", ["..", "..%2F..%2Fetc", "abc", "0123456789AB"])
def test_ids_are_validated(client, bad_id):
    assert client.get(f"/api/drawings/{bad_id}").status_code == 404
    assert client.get(f"/api/drawings/{bad_id}/pages/1/image").status_code == 404


def test_missing_page_is_404(client):
    drawing = upload(client, "a.pdf", make_pdf(pages=1))
    assert client.get(f"/api/drawings/{drawing['id']}/pages/2/image").status_code == 404
