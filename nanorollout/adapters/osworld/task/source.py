"""OSWorld task source resolution."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

OSWORLD_DATA_ROOT = os.environ.get("OSWORLD_ROOT")


def resolve_test_all_path(extra_args: Dict[str, Any]) -> str:
    test_all_meta_path = extra_args.get("test_all_meta_path")
    if test_all_meta_path:
        return str(Path(test_all_meta_path).expanduser())

    data_root = extra_args.get("osworld_root") or OSWORLD_DATA_ROOT
    if data_root:
        return str(
            Path(data_root).expanduser() / "evaluation_examples" / "test_all.json"
        )

    raise ValueError(
        "OSWorld requires a task metadata path. Set OSWORLD_ROOT, pass "
        "--osworld-root, or pass --test-all-meta-path."
    )


def load_osworld_task(task_id: str, test_all_meta_path: str) -> Dict[str, Any]:
    meta_path = Path(test_all_meta_path).expanduser()
    if not meta_path.exists():
        raise FileNotFoundError(
            f"OSWorld metadata file not found: {test_all_meta_path}"
        )

    with open(meta_path, encoding="utf-8") as handle:
        test_all = json.load(handle)

    for domain, ids in test_all.items():
        if task_id not in ids:
            continue
        config_path = meta_path.parent / "examples" / domain / f"{task_id}.json"
        with open(config_path, encoding="utf-8") as handle:
            task = json.load(handle)
        task["_domain"] = domain
        return task

    raise ValueError(f"Task {task_id} not found in {test_all_meta_path}")
