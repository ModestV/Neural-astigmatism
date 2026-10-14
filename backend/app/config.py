"""Настройки приложения. Все значения можно переопределить переменными окружения."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    """Каталог с загруженными чертежами, библиотекой обозначений и историей запусков."""

    frontend_dist: Path
    """Собранный фронтенд (npm run build). Если он есть, бэкенд раздаёт его сам."""

    max_upload_mb: int
    detector: str
    """Имя алгоритма поиска из app.detection.DETECTORS."""

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            data_dir=Path(os.environ.get("SYMSEARCH_DATA_DIR", PROJECT_DIR / "data")),
            frontend_dist=Path(
                os.environ.get("SYMSEARCH_FRONTEND_DIST", PROJECT_DIR / "frontend" / "dist")
            ),
            max_upload_mb=int(os.environ.get("SYMSEARCH_MAX_UPLOAD_MB", "100")),
            detector=os.environ.get("SYMSEARCH_DETECTOR", "stub"),
        )
