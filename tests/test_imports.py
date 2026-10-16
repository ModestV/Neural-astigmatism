import importlib
import pkgutil

import symspot


def test_all_modules_import():
    names = [m.name for m in pkgutil.walk_packages(symspot.__path__, "symspot.")]
    assert len(names) > 30
    for name in names:
        importlib.import_module(name)
