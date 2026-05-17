"""SWE-rebench evaluation helpers."""

from __future__ import annotations

import logging
from pathlib import Path
import re
import shlex
import tempfile
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _missing_package_error() -> RuntimeError:
    return RuntimeError(
        "swebench is required for SWE-rebench grading. Install NanoRollout with "
        "the SWE eval dependencies from pyproject.toml."
    )


def _write_temp_log(output: str) -> str:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".log",
        delete=False,
    )
    with handle:
        handle.write(output)
    return handle.name


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _modified_files(patch: str) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()
    for path in re.findall(
        r"^diff --git a/(?:.*?) b/(.*?)$", patch, flags=re.MULTILINE
    ):
        if path not in seen:
            files.append(path)
            seen.add(path)
    return files


def _write_env_file(env_obj: Any, path: str, content: str) -> None:
    parent = str(Path(path).parent)
    if parent and parent != ".":
        mkdir = env_obj.execute(f"mkdir -p {shlex.quote(parent)}")
        if mkdir.exit_code != 0:
            raise RuntimeError(f"Failed to create {parent}: {mkdir.output}")

    upload_file = getattr(env_obj, "upload_file", None)
    if callable(upload_file):
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".patch",
            delete=False,
        )
        source_path = Path(handle.name)
        try:
            with handle:
                handle.write(content)
            upload_file(source_path, path)
        finally:
            source_path.unlink(missing_ok=True)
        return

    result = env_obj.write_file(path, content)
    if result.exit_code != 0:
        raise RuntimeError(f"Failed to write {path}: {result.output}")


def _patch_file_path(label: str) -> str:
    return f"/tmp/nanorollout_swerebench/{label}.patch"


def _write_patch_file(env_obj: Any, label: str, patch: str) -> Optional[str]:
    if not patch.strip():
        return None
    path = _patch_file_path(label)
    _write_env_file(env_obj, path, patch)
    return path


def _apply_patch_command(test_patch: str, patch_path: Optional[str] = None) -> str:
    if not test_patch.strip():
        return "echo 'No test patch'"
    if patch_path:
        return (
            "git apply -v --whitespace=nowarn "
            f"{shlex.quote(patch_path)} || exit 2"
        )
    delimiter = "EOF_417120113996"
    return f"git apply -v - <<'{delimiter}'\n{test_patch}\n{delimiter}"


def _install_command(install_config: dict[str, Any]) -> Optional[str]:
    install = install_config.get("install")
    if not install:
        return None
    return str(install)


def _test_command(instance: dict[str, Any], test_cmd: Any) -> str:
    try:
        from swebench.harness.test_spec.python import get_test_directives
    except ImportError as exc:
        raise _missing_package_error() from exc

    command = str(test_cmd[-1] if isinstance(test_cmd, list) else test_cmd)
    directives = [shlex.quote(str(item)) for item in get_test_directives(instance)]
    return " ".join([command, *directives])


def _resolve_parser(parser_name: str):
    try:
        from swebench.harness.log_parsers import python as python_parsers
    except ImportError as exc:
        raise _missing_package_error() from exc

    parser = getattr(python_parsers, parser_name, None)
    if not callable(parser):
        raise ValueError(f"Unsupported SWE-rebench log parser: {parser_name}")

    def parse(log: str) -> dict[str, str]:
        try:
            status_map = parser(log, None)
        except TypeError:
            status_map = parser(log)
        return status_map if isinstance(status_map, dict) else {}

    return parse


def _build_eval_script(
    instance: dict[str, Any],
    workspace_dir: str,
    test_patch_path: Optional[str] = None,
) -> str:
    install_config = instance.get("install_config") or {}
    test_cmd = install_config.get("test_cmd")
    if not test_cmd:
        raise ValueError("SWE-rebench instance is missing install_config.test_cmd")

    test_patch = instance.get("test_patch") or ""
    test_files = _modified_files(test_patch)
    reset_tests = (
        f"git checkout {instance['base_commit']} {' '.join(test_files)}"
        if test_files
        else "echo 'No test files to reset'"
    )

    try:
        from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT
    except ImportError as exc:
        raise _missing_package_error() from exc

    eval_commands = [
        "#!/bin/bash",
        "set -uxo pipefail",
        "source /opt/miniconda3/bin/activate",
        "conda activate testbed",
        f"cd {workspace_dir}",
        f"git config --global --add safe.directory {workspace_dir}",
    ]
    install_command = _install_command(install_config)
    if install_command:
        eval_commands.append(install_command)
    eval_commands.extend(
        [
            reset_tests,
            _apply_patch_command(test_patch, test_patch_path),
            f"echo '{START_TEST_OUTPUT}'",
            _test_command(instance, test_cmd),
            f"echo '{END_TEST_OUTPUT}'",
            reset_tests,
        ]
    )

    return "\n".join(eval_commands)


def _extract_marked_output(output: str) -> tuple[str, bool]:
    try:
        from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT
    except ImportError as exc:
        raise _missing_package_error() from exc

    if START_TEST_OUTPUT not in output or END_TEST_OUTPUT not in output:
        return output, False
    return output.split(START_TEST_OUTPUT, 1)[1].split(END_TEST_OUTPUT, 1)[0], True


def _build_report(
    instance: dict[str, Any],
    status_map: dict[str, str],
) -> tuple[dict[str, Any], str]:
    try:
        from swebench.harness.constants import (
            FAIL_TO_FAIL,
            FAIL_TO_PASS,
            PASS_TO_FAIL,
            PASS_TO_PASS,
            EvalType,
        )
        from swebench.harness.grading import (
            get_eval_tests_report,
            get_resolution_status,
        )
    except ImportError as exc:
        raise _missing_package_error() from exc

    gold_results = {
        FAIL_TO_PASS: _coerce_list(instance.get(FAIL_TO_PASS)),
        PASS_TO_PASS: _coerce_list(instance.get(PASS_TO_PASS)),
        FAIL_TO_FAIL: _coerce_list(instance.get(FAIL_TO_FAIL)),
        PASS_TO_FAIL: _coerce_list(instance.get(PASS_TO_FAIL)),
    }
    report = get_eval_tests_report(
        status_map,
        gold_results,
        calculate_to_fail=bool(
            gold_results[FAIL_TO_FAIL] or gold_results[PASS_TO_FAIL]
        ),
        eval_type=EvalType.PASS_AND_FAIL,
    )
    return report, get_resolution_status(report)


def run_rebench_eval(
    env_obj: Any,
    instance: dict[str, Any],
    eval_timeout: Optional[int],
    workspace_dir: str,
) -> tuple[dict[str, Any], Optional[str]]:
    try:
        from swebench.harness.constants import ResolvedStatus
    except ImportError as exc:
        raise _missing_package_error() from exc

    instance_id = instance.get("instance_id", "unknown")
    eval_output = None
    try:
        test_patch_path = _write_patch_file(
            env_obj,
            "test",
            str(instance.get("test_patch") or ""),
        )
        eval_script = _build_eval_script(instance, workspace_dir, test_patch_path)
        result = env_obj.execute(eval_script, timeout=eval_timeout or 1800)
        eval_output = result.output or ""
        _write_temp_log(eval_output)
        test_output, found = _extract_marked_output(eval_output)
        parser_name = (instance.get("install_config") or {}).get(
            "log_parser",
            "parse_log_pytest",
        )
        status_map = _resolve_parser(str(parser_name))(test_output)

        if not found:
            return (
                {
                    "resolved": False,
                    "resolved_status": ResolvedStatus.NO.value,
                    "reward": 0,
                    "error": "missing test output markers",
                    "eval_exit_code": result.exit_code,
                    "status_map": status_map,
                    "report": {},
                },
                eval_output,
            )

        report, resolved_status = _build_report(instance, status_map)
        resolved = resolved_status == ResolvedStatus.FULL.value
        return (
            {
                "resolved": resolved,
                "resolved_status": resolved_status,
                "reward": 1 if resolved else 0,
                "eval_exit_code": result.exit_code,
                "status_map": status_map,
                "report": report,
            },
            eval_output,
        )
    except Exception as exc:
        logger.exception("[%s] SWE-rebench eval failed: %s", instance_id, exc)
        return (
            {
                "resolved": False,
                "resolved_status": ResolvedStatus.NO.value,
                "reward": 0,
                "error": str(exc),
                "status_map": {},
                "report": {},
            },
            eval_output,
        )
