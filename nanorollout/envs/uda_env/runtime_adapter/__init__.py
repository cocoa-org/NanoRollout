"""Runtime-adapter registry.

A ``RuntimeAdapter`` lets the UDA agent loop drive a sandbox that is not
``uda-desktop``. Picked at TaskExecutor construction time via
``sandbox.client_type`` (e.g. ``"osworld-v1"``).

Sister abstraction to :mod:`nanorollout.envs.uda_env.driver` — drivers
encapsulate per-benchmark task semantics (load_task / score / GT
injection), adapters encapsulate per-runtime action semantics (how
``computer_use_left_click`` reaches the screen).

Add a new runtime:

::

    # nanorollout/envs/uda_env/runtime_adapter/osworld_v2.py
    from .base import RuntimeAdapter

    class OSWorldV2Adapter:
        runtime_type = "osworld-v2"
        ...   # implement the RuntimeAdapter protocol

    # this file's REGISTRY (lazy because adapters pull in heavy deps
    # like boto3 / DesktopEnv that we don't want imported at process start):
    _FACTORY["osworld-v2"] = lambda cfg: OSWorldV2Adapter(cfg)
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from .base import (
    XDOTOOL_TO_PYAUTOGUI,
    CoordScaler,
    RuntimeAdapter,
    map_key,
    map_key_combo,
)

__all__ = [
    "RuntimeAdapter",
    "CoordScaler",
    "XDOTOOL_TO_PYAUTOGUI",
    "map_key",
    "map_key_combo",
    "load_adapter",
    "register_adapter",
    "REGISTRY",
]


# Lazy factory map: name -> (sandbox_config_dict) -> RuntimeAdapter. Lazy so
# that importing this package doesn't pull boto3 / DesktopEnv on every run.
_FACTORY: Dict[str, Callable[[Dict[str, Any]], RuntimeAdapter]] = {}


def _osworld_v1_factory(sandbox_config: Dict[str, Any]) -> RuntimeAdapter:
    from .osworld_v1 import OSWorldV1Adapter

    return OSWorldV1Adapter(sandbox_config=sandbox_config)


_FACTORY["osworld-v1"] = _osworld_v1_factory


# Backward-compat REGISTRY view (read-only mapping of names -> factories).
REGISTRY: Dict[str, Callable[[Dict[str, Any]], RuntimeAdapter]] = _FACTORY


def register_adapter(
    name: str,
    factory: Callable[[Dict[str, Any]], RuntimeAdapter],
) -> None:
    """Register a runtime-adapter factory under ``name``."""
    _FACTORY[name] = factory


def load_adapter(name: str, sandbox_config: Dict[str, Any]) -> RuntimeAdapter:
    """Return a fresh adapter instance for ``name``.

    Raises ``KeyError`` with a helpful message listing known adapters.
    """
    if name not in _FACTORY:
        raise KeyError(
            f"Unknown runtime adapter: {name!r}. "
            f"Known adapters: {sorted(_FACTORY)}. "
            f"To add one, see nanorollout/envs/uda_env/runtime_adapter/__init__.py."
        )
    return _FACTORY[name](sandbox_config)
