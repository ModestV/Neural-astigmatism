from .base import Detector, PageRef, Template
from .stub import StubDetector

DETECTORS: dict[str, type[Detector]] = {
    StubDetector.name: StubDetector,
}


def get_detector(name: str) -> Detector:
    try:
        return DETECTORS[name]()
    except KeyError:
        known = ", ".join(DETECTORS)
        raise ValueError(f"Неизвестный алгоритм {name!r}, доступны: {known}") from None


__all__ = ["DETECTORS", "Detector", "PageRef", "Template", "get_detector"]
