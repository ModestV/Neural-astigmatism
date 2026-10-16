"""Ленивая загрузка cairosvg (на macOS libcairo из Homebrew не всегда видна загрузчику)."""

from __future__ import annotations

import ctypes.util
from pathlib import Path
from types import ModuleType

_EXTRA_LIB_DIRS = ("/opt/homebrew/lib", "/usr/local/lib", "/opt/local/lib")


def load_cairosvg() -> ModuleType:
    """Импортировать cairosvg; при OSError добавить типовые каталоги Homebrew/MacPorts."""
    try:
        import cairosvg

        return cairosvg
    except OSError:
        pass

    original = ctypes.util.find_library

    def patched(name: str) -> str | None:
        found = original(name)
        if found:
            return found
        for d in _EXTRA_LIB_DIRS:
            for fn in (f"lib{name}.dylib", f"lib{name}.2.dylib", f"lib{name}.so"):
                p = Path(d) / fn
                if p.exists():
                    return str(p)
        return None

    ctypes.util.find_library = patched
    try:
        import cairosvg

        return cairosvg
    finally:
        ctypes.util.find_library = original
