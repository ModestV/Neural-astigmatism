"""Локальный редактор рамок ИУО с импортом COCO и экспортом в benchmark."""

from __future__ import annotations

import argparse
import hmac
import json
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from annotation_core import (
    AnnotationStore,
    ConflictError,
    Workspace,
    discover_workspace,
    import_coco,
)
from pdf_io import pdf_path, render_page

BASE = Path(__file__).resolve().parent


def initialize(drawings: Path, path: Path, coco_files: list[Path] | None = None) -> Workspace:
    """Первый запуск импортирует набор; последующие открывают сохранённую работу."""
    store = AnnotationStore(path, drawings)
    if path.exists():
        workspace = store.load()
        store.validate_sources(workspace)
        return workspace
    workspace = discover_workspace(drawings)
    sources = coco_files if coco_files is not None else sorted(drawings.rglob("*.json"))
    for source in sources:
        data = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(data, dict) and {"images", "annotations", "categories"} <= data.keys():
            workspace = import_coco(workspace, source, drawings)
    return store.save(workspace)


def make_server(
    store: AnnotationStore,
    cache: Path,
    export_dir: Path,
    port: int = 0,
) -> ThreadingHTTPServer:
    """Сервер доступен только с этого компьютера; запись защищена токеном и revision."""
    token = secrets.token_urlsafe(24)
    render_lock = threading.Lock()
    static = {
        "/": ("index.html", "text/html; charset=utf-8"),
        "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        "/style.css": ("style.css", "text/css; charset=utf-8"),
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            # Токен и содержимое разметки не выводим в журнал запросов.
            pass

        def send(self, status: int, data: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' blob:; "
                "script-src 'self'; style-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(data)

        def json(self, status: int, data: dict) -> None:
            self.send(
                status,
                json.dumps(data, ensure_ascii=False, allow_nan=False).encode(),
                "application/json; charset=utf-8",
            )

        def authorized(self) -> bool:
            value = self.headers.get("Authorization", "")
            if not hmac.compare_digest(value, "Bearer " + token):
                self.json(403, {"error": "Нет доступа: откройте ссылку, выданную при запуске"})
                return False
            expected = f"http://127.0.0.1:{self.server.server_port}"
            origin = self.headers.get("Origin")
            if origin and origin != expected:
                self.json(403, {"error": "Недопустимый источник запроса"})
                return False
            return True

        def do_GET(self) -> None:
            route = urlsplit(self.path)
            if route.path in static:
                name, mime = static[route.path]
                self.send(200, (BASE / "annotation_ui" / name).read_bytes(), mime)
                return
            if not self.authorized():
                return
            try:
                if route.path == "/api/state":
                    self.json(
                        200,
                        {
                            "workspace": store.load().model_dump(),
                            "workspace_path": str(store.path),
                            "drawings_path": str(store.drawings),
                        },
                    )
                elif route.path == "/api/image":
                    query = parse_qs(route.query)
                    page_id = query.get("page", [""])[0]
                    dpi = int(query.get("dpi", ["150"])[0])
                    if dpi not in (100, 150, 200):
                        raise ValueError("Допустимые DPI: 100, 150, 200")
                    workspace = store.load()
                    page = next((p for p in workspace.pages if p.id == page_id), None)
                    if page is None:
                        raise ValueError("Страница не найдена")
                    with render_lock:
                        import cv2

                        image, meta = render_page(
                            pdf_path(store.drawings, page.file),
                            page.page,
                            dpi,
                            cache,
                            40_000_000,
                        )
                        if meta["sha256"] != page.sha256:
                            raise ValueError("PDF изменился; требуется проверить привязку разметки")
                        ok, encoded = cv2.imencode(".png", image)
                    if not ok:
                        raise ValueError("Не удалось подготовить страницу")
                    self.send(200, encoded.tobytes(), "image/png")
                else:
                    self.json(404, {"error": "Не найдено"})
            except (ValueError, OSError, RuntimeError) as error:
                self.json(400, {"error": str(error)})

        def do_POST(self) -> None:
            if not self.authorized():
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 10_000_000:
                    raise ValueError("Недопустимый размер запроса")
                body = json.loads(self.rfile.read(length))
                if self.path == "/api/save":
                    updated = store.save(Workspace.model_validate(body))
                    self.json(200, {"revision": updated.revision})
                elif self.path == "/api/export":
                    output, warnings = store.export(export_dir, body["page_id"])
                    self.json(200, {"path": str(output), "warnings": warnings})
                else:
                    self.json(404, {"error": "Не найдено"})
            except ConflictError as error:
                self.json(409, {"error": str(error)})
            except (ValueError, OSError, RuntimeError, KeyError) as error:
                self.json(400, {"error": str(error)})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.editor_url = f"http://127.0.0.1:{server.server_port}/#{token}"
    return server


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drawings-dir", type=Path, default=BASE / "drawings")
    parser.add_argument("--workspace", type=Path, default=BASE / "annotations/workspace.json")
    parser.add_argument("--cache-dir", type=Path, default=BASE / "cache")
    parser.add_argument("--export-dir", type=Path, default=BASE / "annotations/exports")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    try:
        workspace = initialize(args.drawings_dir, args.workspace)
        server = make_server(
            AnnotationStore(args.workspace, args.drawings_dir),
            args.cache_dir,
            args.export_dir,
            args.port,
        )
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Не удалось открыть редактор: {error}", flush=True)
        return 1
    print(
        f"Страниц: {len(workspace.pages)}, рамок: "
        f"{sum(len(p.annotations) for p in workspace.pages)}",
        flush=True,
    )
    print(f"Разметка: {args.workspace}", flush=True)
    print(f"Редактор: {server.editor_url}", flush=True)
    if not args.no_browser:
        webbrowser.open(server.editor_url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
