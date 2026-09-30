"""Env registry with lazy per-benchmark imports."""

from .base import BaseEnv

_LAZY = {
    "AlfWorldEnv": ".alfworld_env",
    "WMAlfWorldEnv": ".wm_alfworld_env",
    "DeterministicWMAlfWorldEnv": ".wm_alfworld_env",
    "NoProbWMAlfWorldEnv": ".wm_alfworld_env",
    "WalleAlfWorldEnv": ".walle_alfworld_env",
    "WalleWMAlfWorldEnv": ".walle_alfworld_env",
    "SciWorldEnv": ".sciworld_env",
    "WMSciWorldEnv": ".wm_sciworld_env",
    "DeterministicWMSciWorldEnv": ".wm_sciworld_env",
    "WalleOracleSciWorldEnv": ".walle_oracle_sciworld_env",
    "WalleWMOracleSciWorldEnv": ".walle_wm_oracle_sciworld_env",
    "BabyAIEnv": ".babyai_env",
    "WMBabyAIEnv": ".wm_babyai_env",
    "WalleOracleBabyAIEnv": ".walle_oracle_babyai_env",
    "WalleWMOracleBabyAIEnv": ".walle_wm_oracle_babyai_env",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(_LAZY[name], __name__)
        cls = getattr(mod, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
