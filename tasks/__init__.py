"""Task registry with lazy per-benchmark imports."""

from .base import Task

_LAZY = {
    "AlfWorldTask": ".alfworld",
    "SciWorldTask": ".sciworld",
    "BabyAITask": ".babyai",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(_LAZY[name], __name__)
        cls = getattr(mod, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
