"""SWE-Bench style evaluation backed by upstream harness packages."""

from __future__ import annotations

import logging
import tempfile
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _missing_package_error(package: str) -> RuntimeError:
    return RuntimeError(
        f"{package} is required for SWE evaluation. Install NanoRollout with the "
        "SWE eval dependencies from pyproject.toml."
    )


def _is_swesmith_instance(instance: dict[str, Any]) -> bool:
    instance_id = str(instance.get("instance_id", ""))
    source = str(instance.get("source", ""))
    return (
        "smith" in source.lower()
        or "SWE-smith" in str(instance.get("dataset", ""))
        or (not instance.get("test_patch") and "." in instance_id)
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


def _extract_marked_output(output: str, start: str, end: str) -> str:
    if start not in output or end not in output:
        return output
    return output.split(start, 1)[1].split(end, 1)[0]


def _restore_swesmith_hidden_tests_command() -> str:
    return "\n".join(
        [
            (
                "mapfile -d '' NANOROLLOUT_SWESMITH_TEST_FILES "
                "< <(git diff --name-only -z HEAD~1..HEAD)"
            ),
            'if [ "${#NANOROLLOUT_SWESMITH_TEST_FILES[@]}" -gt 0 ]; then',
            '  git checkout HEAD~1 -- "${NANOROLLOUT_SWESMITH_TEST_FILES[@]}"',
            "else",
            "  echo 'No SWE-smith hidden tests to restore'",
            "fi",
        ]
    )


def _build_swesmith_eval_script(workspace_dir: str, test_cmd: str) -> str:
    try:
        from swesmith.constants import TEST_OUTPUT_END, TEST_OUTPUT_START
    except ImportError as exc:
        raise _missing_package_error("swesmith") from exc

    return "\n".join(
        [
            "#!/bin/bash",
            "set -uxo pipefail",
            "export PATH=/go/bin:/usr/local/go/bin:/usr/local/cargo/bin:"
            "/root/.cargo/bin:/root/.local/bin:$PATH",
            f"cd {workspace_dir}",
            f"git config --global --add safe.directory {workspace_dir}",
            _restore_swesmith_hidden_tests_command(),
            f"echo '{TEST_OUTPUT_START}'",
            test_cmd,
            f"echo '{TEST_OUTPUT_END}'",
        ]
    )


def _run_swesmith_eval(
    env_obj: Any,
    instance: dict[str, Any],
    eval_timeout: Optional[int],
    workspace_dir: str,
) -> tuple[dict[str, Any], Optional[str]]:
    try:
        from swebench.harness.constants import (
            KEY_INSTANCE_ID,
            KEY_MODEL,
            KEY_PREDICTION,
        )
        from swebench.harness.grading import get_resolution_status
        from swesmith.constants import TEST_OUTPUT_END, TEST_OUTPUT_START
        from swesmith.harness.grading import get_eval_report
        from swesmith.profiles import registry
    except ImportError as exc:
        raise _missing_package_error("swesmith") from exc

    instance_id = instance.get("instance_id", "unknown")
    eval_output = None
    try:
        profile = registry.get_from_inst(instance)
        test_cmd, _ = profile.get_test_cmd(instance)
        eval_script = _build_swesmith_eval_script(workspace_dir, test_cmd)
        eval_result = env_obj.execute(
            eval_script, timeout=eval_timeout or profile.timeout
        )
        eval_output = eval_result.output or ""
        log_path = _write_temp_log(eval_output)
        prediction = {
            KEY_INSTANCE_ID: instance_id,
            KEY_MODEL: "nanorollout",
            KEY_PREDICTION: "",
        }
        report = get_eval_report(prediction, instance, log_path)
        tests_status = report.get("tests_status", {})
        resolved_status = (
            get_resolution_status(tests_status)
            if tests_status
            else ("RESOLVED_FULL" if report.get("resolved") else "RESOLVED_NO")
        )
        test_output = _extract_marked_output(
            eval_output,
            TEST_OUTPUT_START,
            TEST_OUTPUT_END,
        )
        status_map = profile.log_parser(test_output)
        resolved = bool(report.get("resolved"))
        return (
            {
                "resolved": resolved,
                "resolved_status": resolved_status,
                "reward": 1 if resolved else 0,
                "eval_exit_code": eval_result.exit_code,
                "status_map": status_map,
                "report": tests_status,
            },
            eval_output,
        )
    except Exception as exc:
        logger.exception("[%s] SWE-smith eval failed: %s", instance_id, exc)
        return (
            {
                "resolved": False,
                "resolved_status": "RESOLVED_NO",
                "reward": 0,
                "error": str(exc),
                "status_map": {},
                "report": {},
            },
            eval_output,
        )


def _run_swebench_eval(
    env_obj: Any,
    instance: dict[str, Any],
    eval_timeout: Optional[int],
    workspace_dir: str,
) -> tuple[dict[str, Any], Optional[str]]:
    try:
        from swebench.harness.constants import (
            KEY_INSTANCE_ID,
            KEY_MODEL,
            KEY_PREDICTION,
            ResolvedStatus,
        )
        from swebench.harness.grading import (
            get_eval_report,
            get_logs_eval,
            get_resolution_status,
        )
        from swebench.harness.test_spec.test_spec import make_test_spec
    except ImportError as exc:
        raise _missing_package_error("swebench") from exc

    instance_id = instance.get("instance_id", "unknown")
    eval_output = None
    try:
        test_spec = make_test_spec(instance, arch="x86_64")
        eval_script = test_spec.eval_script
        if workspace_dir != "/testbed":
            eval_script = eval_script.replace("/testbed", workspace_dir)
        eval_result = env_obj.execute(eval_script, timeout=eval_timeout or 1800)
        eval_output = eval_result.output or ""
        log_path = _write_temp_log(eval_output)
        status_map, found = get_logs_eval(test_spec, log_path)
        if not found:
            return (
                {
                    "resolved": False,
                    "resolved_status": ResolvedStatus.NO.value,
                    "reward": 0,
                    "error": "missing test output markers",
                    "eval_exit_code": eval_result.exit_code,
                    "status_map": status_map,
                    "report": {},
                },
                eval_output,
            )

        prediction = {
            KEY_INSTANCE_ID: instance_id,
            KEY_MODEL: "nanorollout",
            KEY_PREDICTION: "",
        }
        report_map = get_eval_report(
            test_spec,
            prediction,
            log_path,
            include_tests_status=True,
        )
        report = report_map.get(instance_id, {})
        tests_status = report.get("tests_status", {})
        resolved_status = get_resolution_status(tests_status)
        resolved = resolved_status == ResolvedStatus.FULL.value
        logger.info(
            "[%s] Eval result: resolved=%s, resolved_status=%s, reward=%s",
            instance_id,
            resolved,
            resolved_status,
            1 if resolved else 0,
        )
        return (
            {
                "resolved": resolved,
                "resolved_status": resolved_status,
                "reward": 1 if resolved else 0,
                "eval_exit_code": eval_result.exit_code,
                "status_map": status_map,
                "report": tests_status,
            },
            eval_output,
        )
    except Exception as exc:
        logger.exception("[%s] SWE-Bench eval failed: %s", instance_id, exc)
        return (
            {
                "resolved": False,
                "resolved_status": "RESOLVED_NO",
                "reward": 0,
                "error": str(exc),
                "status_map": {},
                "report": {},
            },
            eval_output,
        )


def run_swebench_eval(
    env_obj: Any,
    instance: dict[str, Any],
    eval_timeout: Optional[int],
    workspace_dir: str,
) -> tuple[dict[str, Any], Optional[str]]:
    if _is_swesmith_instance(instance):
        return _run_swesmith_eval(env_obj, instance, eval_timeout, workspace_dir)
    return _run_swebench_eval(env_obj, instance, eval_timeout, workspace_dir)
