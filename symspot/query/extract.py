"""3.1 Извлечение эталона из рамки. Ядро (Opus)."""

from __future__ import annotations

from symspot.core.config import Config
from symspot.core.types import Query, QuerySpec
from symspot.normalize.index import PrimIndex


def extract_query(spec: QuerySpec, index: PrimIndex, cfg: Config) -> Query:
    """Эталон из нормализованного чертежа (индекс по его кускам).

    - Кандидаты: куски, рамка которых пересекает рамку эталона (расширенную на
      ``query_bbox_pad`` диагонали).
    - Кусок берется, если его штрих (все куски с тем же ``stroke_id``) лежит в рамке по длине
      >= ``query_parent_inside``, и сам кусок лежит в рамке хотя бы наполовину по длине.
      Так стена или размерная линия, проходящая через рамку, отбрасывается.
    - TextPrim берется, если центр его рамки в рамке эталона.
    - Нормализация: центр рамки -> 0, диагональ рамки -> 1; ``frame`` =
      Similarity(s=diag, theta=0, tx=cx, ty=cy) переводит нормализованное в чертеж.
    - Сэмплы и касательные с шагом ``sample_step`` (в нормализованных единицах),
      веса = 1 (до ``discriminative``), ``piece_ids`` = id взятых кусков.
    - ValueError, если в рамке не нашлось ни одного куска и текста.
    """
    raise NotImplementedError
