import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


_R2E_SKIP_FILES = ["run_tests.sh", "r2e_tests"]


def setup_r2egym_env(env, workspace_dir: str = "/testbed", alt_path: str = "/root") -> None:
    """Mirror upstream ``DockerRuntime.setup_env()`` for R2E-Gym containers."""
    env.execute(f"ln -sf {workspace_dir}/.venv {alt_path}/.venv")
    env.execute(f"mkdir -p {alt_path}/.local/bin")
    env.execute(f"ln -sf {workspace_dir}/.venv/bin/python {alt_path}/.local/bin/python")
    env.execute(f"ln -sf {workspace_dir}/.venv/bin/python {alt_path}/.local/bin/python3")
    env.execute(
        f"find {workspace_dir}/.venv/bin -type f -executable -exec ln -sf {{}} {alt_path}/.local/bin/ \\;"
    )
    env.execute("find . -name '*.pyc' -delete")
    env.execute("find . -name '__pycache__' -exec rm -rf {} +")
    env.execute("find /r2e_tests -name '*.pyc' -delete 2>/dev/null")
    env.execute("find /r2e_tests -name '__pycache__' -exec rm -rf {} + 2>/dev/null")
    for skip_file in _R2E_SKIP_FILES:
        env.execute(f"mv {workspace_dir}/{skip_file} {alt_path}/{skip_file} 2>/dev/null")
    env.execute(f"mv /r2e_tests {alt_path}/r2e_tests 2>/dev/null")
    env.execute(f"ln -sf {alt_path}/r2e_tests {workspace_dir}/r2e_tests 2>/dev/null")
    logger.info("R2E-Gym environment setup complete")
