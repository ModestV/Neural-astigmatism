"""Зависимости FastAPI: доступ к общим объектам приложения из обработчиков."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from .config import Settings
from .detection import Detector
from .storage import Storage


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_storage(request: Request) -> Storage:
    return request.app.state.storage


def get_detector(request: Request) -> Detector:
    return request.app.state.detector


SettingsDep = Annotated[Settings, Depends(get_settings)]
StorageDep = Annotated[Storage, Depends(get_storage)]
DetectorDep = Annotated[Detector, Depends(get_detector)]


def drawing_names(storage: Storage) -> dict[str, str]:
    return {d.id: d.name for d in storage.list_drawings()}
