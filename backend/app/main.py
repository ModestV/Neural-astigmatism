"""Точка входа бэкенда: uvicorn app.main:app --reload"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from . import __version__
from .config import Settings
from .detection import get_detector
from .routers import drawings, search, symbols, system
from .storage import Storage

DESCRIPTION = """
API сервиса поиска условных обозначений на чертежах.

* **drawings** — загрузка чертежей (PDF, SVG, DXF, DWG) и страницы для просмотра;
* **symbols** — библиотека обозначений, то есть образцы для поиска;
* **search** — запуск поиска и история запусков (пока работает заглушка).

Координаты везде в единицах страницы: (0, 0) — левый верхний угол, ось Y вниз.
"""

FRONTEND_MISSING_HTML = """<!doctype html>
<html lang="ru"><meta charset="utf-8"><title>Symbol Search API</title>
<body style="font-family: sans-serif; max-width: 40rem; margin: 3rem auto">
<h1>Symbol Search API</h1>
<p>Фронтенд ещё не собран. Для разработки запустите <code>npm run dev</code> в папке
<code>frontend/</code> и откройте <a href="http://localhost:5173">http://localhost:5173</a>,
а чтобы бэкенд раздавал его сам, выполните <code>npm run build</code>.</p>
<p>Документация API: <a href="/docs">/docs</a></p>
</body></html>"""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="Symbol Search", version=__version__, description=DESCRIPTION)
    app.state.settings = settings
    app.state.storage = Storage(settings.data_dir)
    app.state.detector = get_detector(settings.detector)

    for module in (system, drawings, symbols, search):
        app.include_router(module.router)
    _serve_frontend(app, settings.frontend_dist)
    return app


def _serve_frontend(app: FastAPI, dist: Path) -> None:
    """Собранный фронтенд раздаём этим же сервером. Во время разработки его раздаёт Vite."""
    index = dist / "index.html"
    if not index.is_file():

        @app.get("/", include_in_schema=False)
        def frontend_missing() -> HTMLResponse:
            return HTMLResponse(FRONTEND_MISSING_HTML)

        return

    root = dist.resolve()

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(404, "Нет такого метода API")
        file = (root / path).resolve()
        if path and file.is_file() and file.is_relative_to(root):
            return FileResponse(file)
        # Остальные адреса (/drawings/…, /symbols, …) — маршруты React, отдаём index.html.
        return FileResponse(index)


app = create_app()
