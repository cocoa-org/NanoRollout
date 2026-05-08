from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import logging

from .constants import FAIL_TO_PASS, MAP_REPO_TO_EXT, NON_TEST_EXTS, PASS_TO_PASS
from .constants.swesmith.profiles import registry as _swesmith_registry
from .repo_specs import get_repo_test_cmd
from .script_gen import make_eval_script

DEFAULT_REPO_DIR = "/testbed"
DEFAULT_PYTEST_CMD = "python -m pytest -rA --color=no"

logger = logging.getLogger(__name__)


def _resolve_base_commit(instance: dict[str, Any]) -> str | None:
    """Derive base_commit from the SWE-smith profile registry when the
    instance doesn't carry one (the HF dataset omits it)."""
    repo = instance.get("repo", "")
    try:
        profile = _swesmith_registry.get(repo)
        return profile.commit
    except KeyError:
        return None


@dataclass(frozen=True)
class TestSpec:
    instance_id: str
    repo: str
    base_commit: str
    env_name: str
    test_patch: str
    test_cmd: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    eval_script: str


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
                logger.error("Failed to parse list: %s", text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
    return [str(value)]


def get_test_directives(instance: dict[str, Any]) -> list[str]:
    repo = instance.get("repo", "")
    if repo == "swe-bench/humaneval":
        return ["test.py"]

    test_patch = instance.get("test_patch") or ""
    diff_pat = r"diff --git a/.* b/(.*)"
    directives = re.findall(diff_pat, test_patch)
    directives = [
        directive
        for directive in directives
        if not any(directive.endswith(ext) for ext in NON_TEST_EXTS)
    ]

    if repo == "django/django":
        directives_transformed = []
        for directive in directives:
            if directive.endswith(".py"):
                directive = directive[: -len(".py")]
            if directive.startswith("tests/"):
                directive = directive[len("tests/") :]
            directive = directive.replace("/", ".")
            directives_transformed.append(directive)
        directives = directives_transformed

    return directives


def _append_test_directives(base_cmd: str, instance: dict[str, Any]) -> str:
    directives = get_test_directives(instance)
    if not directives:
        return base_cmd
    return f"{base_cmd} {' '.join(directives)}"


def _default_pytest_cmd(instance: dict[str, Any]) -> str:
    return _append_test_directives(DEFAULT_PYTEST_CMD, instance)


def _get_mypy_test_cmd(instance: dict[str, Any], base_cmd: str) -> str:
    """Build test command for python/mypy.

    mypy uses ``[case testName]`` markers inside ``.test`` data files.
    The pytest ``-k`` flag must receive these case names, not file paths.
    """
    test_patch = instance.get("test_patch") or ""
    test_keys = re.findall(r"\[case ([^\]]+)\]", test_patch)
    if test_keys:
        expr = " or ".join(test_keys)
        return f'{base_cmd} "{expr}"'
    # Fallback: no case markers found, run without -k filter
    return base_cmd.rstrip().removesuffix("-k").rstrip()


def _get_test_cmd(instance: dict[str, Any]) -> str:
    repo = instance.get("repo", "")
    repo_key = repo.lower()

    repo_cmd = get_repo_test_cmd(repo_key, instance.get("version"), instance)
    if repo_cmd:
        # mypy needs special handling: -k expects case names, not file paths
        if repo_key == "python/mypy":
            return _get_mypy_test_cmd(instance, repo_cmd)
        if MAP_REPO_TO_EXT.get(repo_key, "py") == "py":
            return _append_test_directives(repo_cmd, instance)
        return repo_cmd

    return _default_pytest_cmd(instance)


def make_test_spec(
    instance: dict[str, Any],
    *,
    arch: str = "x86_64",
    repo_directory: str = DEFAULT_REPO_DIR,
) -> TestSpec:
    _ = arch  # Reserved for API parity with SWE-bench harness.

    base_commit = instance.get("base_commit") or _resolve_base_commit(instance)
    if not base_commit:
        raise KeyError("Instance missing base_commit")

    test_cmd = _get_test_cmd(instance)
    if not test_cmd:
        raise KeyError(
            "Instance missing test_cmd (expected test_cmd/test.test_cmd or "
            "derivable from test_patch for python repos)."
        )

    env_name = "testbed"
    test_patch = instance.get("test_patch") or ""

    eval_script = make_eval_script(
        instance=instance,
        repo_directory=repo_directory,
        base_commit=base_commit,
        env_name=str(env_name),
        test_patch=test_patch,
        test_cmd=test_cmd,
        build_cmds=None,
    )

    return TestSpec(
        instance_id=str(instance.get("instance_id", "")),
        repo=instance["repo"],
        base_commit=base_commit,
        env_name=str(env_name),
        test_patch=test_patch,
        test_cmd=test_cmd,
        FAIL_TO_PASS=_coerce_list(instance.get(FAIL_TO_PASS)),
        PASS_TO_PASS=_coerce_list(instance.get(PASS_TO_PASS)),
        eval_script=eval_script,
    )
