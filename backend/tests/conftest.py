from __future__ import annotations

import os
import tempfile

# app.main при импорте создаёт приложение с настройками по умолчанию — уводим его данные
# во временный каталог (удалится по завершении), чтобы тесты не мусорили в data/ проекта.
_import_data_dir = tempfile.TemporaryDirectory(prefix="symsearch-test-")
os.environ.setdefault("SYMSEARCH_DATA_DIR", _import_data_dir.name)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        frontend_dist=tmp_path / "no-frontend",
        max_upload_mb=1,
        detector="stub",
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client
