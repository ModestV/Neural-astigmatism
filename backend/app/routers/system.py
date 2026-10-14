from __future__ import annotations

from fastapi import APIRouter

from .. import __version__
from ..converters import dwg_converter
from ..deps import DetectorDep, StorageDep
from ..schemas import DetectorInfo, Health, Stats

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health", response_model=Health)
def health(detector: DetectorDep) -> Health:
    return Health(
        version=__version__,
        detector=DetectorInfo(name=detector.name, description=detector.description),
        dwg_converter=dwg_converter(),
    )


@router.get("/stats", response_model=Stats)
def stats(storage: StorageDep) -> Stats:
    return Stats(
        drawings=len(storage.list_drawings()),
        symbols=len(storage.list_symbols()),
        runs=len(storage.list_runs()),
    )
