"""Маленькие тестовые чертежи, которые создаются прямо в коде."""

from __future__ import annotations

import base64
import io

import ezdxf
import pymupdf
from fastapi.testclient import TestClient


def make_pdf(pages: int = 2, width: float = 842, height: float = 595) -> bytes:
    doc = pymupdf.open()
    for n in range(pages):
        page = doc.new_page(width=width, height=height)
        page.draw_rect(pymupdf.Rect(100, 100, 160, 140), color=(0, 0, 0))
        page.insert_text((100, 90), f"Page {n + 1}", fontsize=12)
    return doc.tobytes()


def make_dxf() -> bytes:
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (200, 0), (200, 100), (0, 100)], close=True)
    msp.add_circle((50, 50), 5)
    stream = io.StringIO()
    doc.write(stream)
    return stream.getvalue().encode()


def png_data_url() -> str:
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 4, 4), False)
    pixmap.clear_with(255)
    return "data:image/png;base64," + base64.b64encode(pixmap.tobytes("png")).decode()


def upload(client: TestClient, name: str, data: bytes) -> dict:
    response = client.post("/api/drawings", files={"file": (name, data)})
    assert response.status_code == 201, response.text
    return response.json()
