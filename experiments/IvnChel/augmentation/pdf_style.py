"""Внешний вид в векторе: толщина линий и инверсия цветов в готовой PDF.

Рамки здесь не меняются: страница остаётся той же, меняется только то, как она нарисована.

Инверсия: поверх страницы кладётся белый прямоугольник с режимом наложения Difference, под
страницу — белая подложка. Результат точно равен `1 - цвет` для каждого пикселя (проверено сравнением
с растровой инверсией), включая картинки внутри PDF.

Толщина линий: у PDF нет общего множителя толщины, поэтому правится содержимое страницы: у каждой
команды `w` (толщина линии) прибавляется поправка. Поправка делится на масштаб в этом месте (его
накапливают команды `cm` и матрицы вложенных форм), поэтому линия меняется на одно и то же число
пикселей готового листа, как в растре (`photometric.line_thickness`): на `2 * delta` px по ширине.
Утолщаются только штрихи (линии), закрашенные фигуры остаются прежними — в этом отличие от растра.
"""

from __future__ import annotations

import re

DPI = 200
# Самая тонкая линия после утончения, в пикселях листа: иначе линия исчезла бы совсем.
MIN_WIDTH_PX = 0.1

_NUMBER = re.compile(rb"[+-]?(?:\d+\.?\d*|\.\d+)\Z")
_DELIMS = rb"\s()<>\[\]{}/%"
_TOKEN = re.compile(
    rb"(?P<ws>\s+)"
    rb"|(?P<comment>%[^\r\n]*)"
    rb"|(?P<dict><<|>>)"
    rb"|(?P<hex><[0-9A-Fa-f\s]*>)"
    rb"|(?P<bracket>[\[\]{}])"
    rb"|(?P<name>/[^" + _DELIMS + rb"]*)"
    rb"|(?P<word>[^" + _DELIMS + rb"]+)"
)
_INLINE_IMAGE_END = re.compile(rb"\sEI(?=\s|$)")


def _skip_string(data: bytes, start: int) -> int:
    """Конец строки `( ... )` с вложенными скобками и экранированием; `start` указывает на `(`."""
    depth = 0
    i = start
    while i < len(data):
        c = data[i]
        if c == 0x5C:  # обратная косая черта: следующий символ не считается
            i += 2
            continue
        if c == 0x28:
            depth += 1
        elif c == 0x29:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(data)


def _tokens(data: bytes):
    """Выдаёт `(вид, текст, начало, конец)`; вид: word, name, string, hex, dict, bracket, inline."""
    pos = 0
    while pos < len(data):
        if data[pos] == 0x28:
            end = _skip_string(data, pos)
            yield "string", data[pos:end], pos, end
            pos = end
            continue
        match = _TOKEN.match(data, pos)
        if match is None:  # одиночный «закрывающий» символ вроде `)`: пропускаем
            pos += 1
            continue
        kind, end = match.lastgroup, match.end()
        pos = end
        if kind in ("ws", "comment"):
            continue
        text = match.group()
        if kind == "word" and text == b"BI":
            image_end = _INLINE_IMAGE_END.search(data, end)
            end = image_end.end() if image_end else len(data)
            yield "inline", data[match.start():end], match.start(), end
            pos = end
            continue
        yield kind, text, match.start(), end


def _multiply(m: tuple, ctm: tuple) -> tuple:
    """Новая матрица `m x ctm` (строки-векторы, как в PDF); матрицы — шестёрки `a b c d e f`."""
    a, b, c, d, e, f = m
    A, B, C, D, E, F = ctm
    return (a * A + b * C, a * B + b * D, c * A + d * C, c * B + d * D,
            e * A + f * C + E, e * B + f * D + F)


def _scale(ctm: tuple) -> float:
    return abs(ctm[0] * ctm[3] - ctm[1] * ctm[2]) ** 0.5


def _fmt(value: float) -> bytes:
    return f"{value:.5f}".rstrip("0").rstrip(".").encode() or b"0"


class _Thickener:
    """Правит толщину линий во всех потоках страницы, включая вложенные формы."""

    def __init__(self, doc, delta_px: float) -> None:
        self.doc = doc
        self.shift_pt = 2 * delta_px * 72 / DPI  # на сколько пунктов меняется толщина линии
        self.min_pt = MIN_WIDTH_PX * 72 / DPI
        self.seen: set[int] = set()
        self.changed = 0

    def new_width(self, width: float, scale: float) -> float:
        if scale <= 0:
            return width
        return max(width + self.shift_pt / scale, self.min_pt / scale)

    def xobject(self, owner: int | None, name: str, parent_lookup) -> int | None:
        """xref формы по имени из `/Resources/XObject` владельца (или из ресурсов родителя)."""
        if owner is not None:
            kind, value = self.doc.xref_get_key(owner, f"Resources/XObject/{name}")
            if kind == "xref":
                return int(value.split()[0])
        return parent_lookup(name) if parent_lookup else None

    def run(self, xref: int, ctm: tuple, inherited_width: float, owner: int | None, parent_lookup):
        """Переписывает поток `xref` (страницы или формы); возвращает ничего, правит документ."""
        data = self.doc.xref_stream(xref)
        if not data:
            return
        start_ctm = ctm
        out: list[bytes] = []
        last = 0
        stack: list[tuple[tuple, float]] = []
        width = inherited_width  # исходная (до правки) толщина в единицах текущего пространства
        operands: list[tuple[str, bytes, int, int]] = []
        lookup = lambda name, o=owner, p=parent_lookup: self.xobject(o, name, p)  # noqa: E731

        for kind, text, start, end in _tokens(data):
            if kind != "word" or _NUMBER.match(text):
                operands.append((kind, text, start, end))
                continue
            if kind == "word" and text in (b"true", b"false", b"null"):
                operands.append((kind, text, start, end))
                continue
            op = text
            if op == b"q":
                stack.append((ctm, width))
            elif op == b"Q":
                if stack:
                    ctm, width = stack.pop()
            elif op == b"cm" and len(operands) >= 6:
                try:
                    m = tuple(float(o[1]) for o in operands[-6:])
                except ValueError:
                    m = None
                if m:
                    ctm = _multiply(m, ctm)
            elif op == b"w" and operands and _NUMBER.match(operands[-1][1]):
                width = float(operands[-1][1])
                _, _, s0, e0 = operands[-1]
                out.append(data[last:s0])
                out.append(_fmt(self.new_width(width, _scale(ctm))))
                last = e0
                self.changed += 1
            elif op == b"Do" and operands and operands[-1][0] == "name":
                name = operands[-1][1][1:].decode("latin-1")
                form = lookup(name)
                if form is not None and self.doc.xref_get_key(form, "Subtype")[1] == "/Form":
                    self.form(form, ctm, width, lookup)
            operands = []
        out.append(data[last:])
        body = b"".join(out)
        prefix = _fmt(self.new_width(inherited_width, _scale(start_ctm))) + b" w\n"
        self.doc.update_stream(xref, prefix + body, compress=True)
        self.changed += 1

    def form(self, xref: int, ctm: tuple, width: float, parent_lookup) -> None:
        if xref in self.seen:
            return
        self.seen.add(xref)
        kind, value = self.doc.xref_get_key(xref, "Matrix")
        matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        if kind == "array":
            try:
                matrix = tuple(float(v) for v in value.strip("[]").split())
            except ValueError:
                pass
        self.run(xref, _multiply(matrix, ctm), width, xref, parent_lookup)


def thicken_page(doc, page, delta_px: float) -> int:
    """Меняет толщину всех линий на странице на `2 * delta_px` px листа (при 200 dpi).

    delta_px > 0 — толще, < 0 — тоньше. Возвращает число переписанных потоков.
    """
    page.clean_contents()  # сливает потоки страницы в один с парными q/Q
    contents = page.get_contents()
    thickener = _Thickener(doc, delta_px)
    for xref in contents:
        thickener.run(xref, (1, 0, 0, 1, 0, 0), 1.0, page.xref, None)
    return thickener.changed


def invert_page(doc, page) -> None:
    """Инвертирует цвета страницы (чёрные линии на белом становятся белыми на чёрном)."""
    res = page.xref
    kind, value = doc.xref_get_key(page.xref, "Resources")
    if kind == "xref":
        res = int(value.split()[0])
        target = ""
    else:
        target = "Resources/"
    gs = doc.get_new_xref()
    doc.update_object(gs, "<</Type/ExtGState/BM/Difference>>")
    kind, _ = doc.xref_get_key(res, f"{target}ExtGState")
    if kind == "null":
        doc.xref_set_key(res, f"{target}ExtGState", f"<</GSinv {gs} 0 R>>")
    else:
        doc.xref_set_key(res, f"{target}ExtGState/GSinv", f"{gs} 0 R")
    box = page.mediabox
    w, h = box.width, box.height
    rect = f"-100 -100 {w + 200:g} {h + 200:g} re f Q"
    under = doc.get_new_xref()
    doc.update_object(under, "<<>>")
    doc.update_stream(under, f"q 1 1 1 rg {rect}".encode())
    over = doc.get_new_xref()
    doc.update_object(over, "<<>>")
    doc.update_stream(over, f"q /GSinv gs 1 1 1 rg {rect}".encode())
    chain = " ".join(f"{x} 0 R" for x in [under, *page.get_contents(), over])
    doc.xref_set_key(page.xref, "Contents", f"[{chain}]")
