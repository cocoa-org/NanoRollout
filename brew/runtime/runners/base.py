from abc import ABC, abstractmethod
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel


class RunnerSpec(BaseModel):
    runner_type: str
    runner_module: Optional[str] = None
    runner_entrypoint: Optional[str] = None


class BaseRunner(ABC):
    def __init__(self, spec: RunnerSpec) -> None:
        self.spec = spec

    @staticmethod
    def _load_runner(module_name: str, entrypoint: str):
        # If module name looks like a file path or ends with .py, load by path.
        if module_name.endswith(".py") or os.sep in module_name or "/" in module_name:
            module_path = Path(module_name)
            if not module_path.is_absolute():
                # Resolve relative paths against repo root so runtime CWD doesn't matter.
                repo_root = Path(__file__).resolve().parents[3]
                module_path = (repo_root / module_path).resolve()
            if not module_path.exists():
                raise FileNotFoundError(f"Runner module not found: {module_path}")
            module_dir = str(module_path.parent)
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)
            spec = importlib.util.spec_from_file_location(
                f"tinyflow_runner_{module_path.stem}", module_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load runner module from {module_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, entrypoint)
        module = importlib.import_module(module_name)
        return getattr(module, entrypoint)

    @abstractmethod
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
