"""Bench-driver registry.

Public entry points:

- :class:`BenchDriver` — the protocol every driver implements.
- :func:`load_driver(name)` — return the driver instance for a given
  benchmark identifier (``"cocoa-v1"`` / ``"wildclaw-v1"``).
- :func:`load_driver_for_task_dir(task_dir)` — infer the driver from
  ``meta.json``'s ``driver`` field, falling back to ``adapter/<bench>/``
  path lookup.

Adding a new benchmark:

::

    # nanorollout/envs/uda_env/driver/osworld_v2.py
    from .base import BenchDriver

    class OSWorldV2Driver:
        name = "osworld-v2"
        ...   # implement the 7-method protocol

    # this file's REGISTRY:
    from .osworld_v2 import OSWorldV2Driver
    REGISTRY[OSWorldV2Driver.name] = OSWorldV2Driver()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .base import BenchDriver, discover_workspace_assets
from .cocoa_v1 import CocoaV1Driver
from .wildclaw_v1 import WildclawV1Driver

__all__ = [
    "BenchDriver",
    "discover_workspace_assets",
    "load_driver",
    "load_driver_for_task_dir",
    "register_driver",
    "REGISTRY",
]


REGISTRY: Dict[str, BenchDriver] = {}


def register_driver(driver: BenchDriver) -> None:
    """Register a driver instance under its ``.name``."""
    REGISTRY[driver.name] = driver


# Built-in drivers
register_driver(CocoaV1Driver())
register_driver(WildclawV1Driver())


def load_driver(name: str) -> BenchDriver:
    """Return the driver registered under ``name``.

    Raises ``KeyError`` with a helpful message listing known drivers.
    """
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown bench driver: {name!r}. "
            f"Known drivers: {sorted(REGISTRY)}. "
            f"To add one, see nanorollout/envs/uda_env/driver/__init__.py."
        )
    return REGISTRY[name]


def load_driver_for_task_dir(task_dir: Path) -> BenchDriver:
    """Infer the driver from a task adapter directory.

    Resolution order:

    1. ``<task_dir>/meta.json`` has a ``driver`` field.
    2. The task_dir's parent name matches a registered driver
       (``adapter/wildclaw-v1/<task_id>`` → ``wildclaw-v1``).
    3. Heuristic: presence of ``test.py.enc`` → cocoa-v1;
       presence of ``grade.py`` → wildclaw-v1.

    Raises ``ValueError`` if none of the above match.
    """
    meta_path = task_dir / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta, dict) and meta.get("driver") in REGISTRY:
                return REGISTRY[meta["driver"]]
        except (ValueError, OSError):
            pass

    parent_name = task_dir.parent.name
    if parent_name in REGISTRY:
        return REGISTRY[parent_name]

    assets = discover_workspace_assets(task_dir)
    if "test_py_enc" in assets:
        return REGISTRY["cocoa-v1"]
    if "grade_py" in assets:
        return REGISTRY["wildclaw-v1"]

    raise ValueError(
        f"Could not infer bench driver for {task_dir}. "
        "Set meta.json's 'driver' field, or place the task under "
        "adapter/<known-driver>/."
    )
