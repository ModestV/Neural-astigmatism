from __future__ import annotations

from collections.abc import Sequence

from ..models import Detection, SearchParams
from .base import Detector, PageRef, Template


class StubDetector(Detector):
    """Заглушка: «находит» только сам образец, если он лежит на просматриваемой странице.

    Нужна, чтобы весь путь «запрос → результаты → подсветка на чертеже» работал
    ещё до появления настоящего алгоритма.
    """

    name = "stub"
    description = "Заглушка: алгоритм не подключён, находится только сам образец"

    def detect(
        self, target: PageRef, templates: Sequence[Template], params: SearchParams
    ) -> list[Detection]:
        return [
            Detection(page=target.index, bbox=t.bbox, score=1.0, symbol_id=t.symbol_id)
            for t in templates
            if t.page.drawing.id == target.drawing.id and t.page.index == target.index
        ]
