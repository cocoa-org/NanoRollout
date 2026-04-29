"""Evaluation, grading, and environment setup for R2E-Gym tasks.

Ported from R2E-Gym's upstream codebase:
  - ``repo_analysis/execution_log_parser.py``  (parse_log_pytest, decolor_dict_keys)
  - ``agenthub/runtime/docker.py``             (_calculate_reward_r2e)
"""

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_TEST_SCRIPT_PATH = "/root/run_tests.sh"

# Regex to strip ANSI escape sequences (colour codes, cursor movement, \r).
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m|\r")

# Narrower regex matching the upstream ``decolor_dict_keys`` pattern.
_ANSI_KEY_RE = re.compile(r"\x1b\[\d+m")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes and carriage returns from *text*.

    Mirrors the regex used in ``DockerRuntime.run_tests``.
    """
    return _ANSI_RE.sub("", text)


def decolor_dict_keys(d: Dict[str, str]) -> Dict[str, str]:
    """Strip ANSI colour codes from dictionary keys.

    Faithfully reproduces
    ``r2egym.repo_analysis.execution_log_parser.decolor_dict_keys``.
    """
    return {_ANSI_KEY_RE.sub("", k): v for k, v in d.items()}


def parse_log_pytest(log: Optional[str]) -> Dict[str, str]:
    """Parse pytest *short test summary info* section into a status map.

    Faithfully reproduces
    ``r2egym.repo_analysis.execution_log_parser.parse_log_pytest``.

    Returns a dict mapping test names to status strings
    (``"PASSED"``, ``"FAILED"``, or ``"ERROR"``).
    """
    if log is None:
        return {}

    test_status_map: Dict[str, str] = {}
    if "short test summary info" not in log:
        return test_status_map

    log_tail = log.split("short test summary info")[1].strip()
    for line in log_tail.split("\n"):
        if "PASSED" in line:
            test_name = ".".join(line.split("::")[1:])
            test_status_map[test_name] = "PASSED"
        elif "FAILED" in line:
            test_name = ".".join(line.split("::")[1:]).split(" - ")[0]
            test_status_map[test_name] = "FAILED"
        elif "ERROR" in line:
            try:
                test_name = ".".join(line.split("::")[1:])
            except IndexError:
                test_name = line
            test_name = test_name.split(" - ")[0]
            test_status_map[test_name] = "ERROR"

    return test_status_map


def _normalize_keys(d: Dict[str, str]) -> Dict[str, str]:
    """Normalize dictionary keys by stripping ``" - ..."`` suffixes.

    Mirrors the key normalization in ``_calculate_reward_r2e``::

        {k.split(" - ")[0]: d[k] for k in sorted(d.keys())}
    """
    return {k.split(" - ")[0]: d[k] for k in sorted(d.keys())}


def compare_results(
    parsed: Dict[str, str],
    expected: Dict[str, str],
) -> Tuple[float, Dict[str, Any]]:
    """Compare *parsed* test results against *expected* results.

    Returns ``(reward, details)`` where reward is ``1.0`` (exact match) or
    ``0.0`` (any mismatch).

    The comparison logic mirrors ``DockerRuntime._calculate_reward_r2e``:

    1. Decolour both dicts' keys.
    2. Normalize keys (strip ``" - ..."`` suffixes).
    3. If key counts differ -> reward = 0.
    4. Otherwise, every non-empty key in *parsed* must exist in *expected*
       with an identical value.
    """
    parsed = decolor_dict_keys(parsed)
    expected = decolor_dict_keys(expected)

    parsed = _normalize_keys(parsed)
    expected = _normalize_keys(expected)

    if len(parsed) != len(expected):
        return 0.0, {
            "reason": "key_count_mismatch",
            "parsed_count": len(parsed),
            "expected_count": len(expected),
            "parsed_keys": sorted(parsed.keys()),
            "expected_keys": sorted(expected.keys()),
        }

    mismatches: list[Dict[str, Any]] = []
    for k in parsed:
        if not k:
            continue
        if k not in expected:
            mismatches.append({"key": k, "parsed": parsed[k], "expected": None})
        elif parsed[k] != expected[k]:
            mismatches.append(
                {"key": k, "parsed": parsed[k], "expected": expected[k]}
            )

    if mismatches:
        return 0.0, {"reason": "value_mismatch", "mismatches": mismatches}

    return 1.0, {"reason": "exact_match", "num_tests": len(parsed)}


def get_r2egym_eval_report(
    test_output: str,
    expected_output_json: str,
    *,
    instance_id: str = "unknown",
) -> Dict[str, Any]:
    """Grade a single R2E-Gym instance.

    Parameters
    ----------
    test_output
        Raw stdout from executing ``run_tests.sh`` inside the container
        (ANSI codes should already be stripped by the caller).
    expected_output_json
        JSON string from the dataset's ``expected_output_json`` field.
    instance_id
        For logging / payload tagging.

    Returns
    -------
    dict
        Payload compatible with TinyFlow's eval pipeline::

            {
                "instance_id": str,
                "resolved": bool,
                "resolved_status": "FULL" | "NO",
                "reward": 1.0 | 0.0,
                "status_map": {test_name: status, ...},
                "report": {...},   # comparison details
            }
    """
    parsed = parse_log_pytest(test_output)

    try:
        expected: Dict[str, str] = json.loads(expected_output_json)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.error(
            "[%s] Failed to parse expected_output_json: %s", instance_id, exc
        )
        return {
            "instance_id": instance_id,
            "resolved": False,
            "resolved_status": "NO",
            "reward": 0,
            "error": f"bad expected_output_json: {exc}",
            "status_map": parsed,
            "report": {},
        }

    reward, details = compare_results(parsed, expected)
    resolved = reward == 1.0

    return {
        "instance_id": instance_id,
        "resolved": resolved,
        "resolved_status": "FULL" if resolved else "NO",
        "reward": reward,
        "status_map": parsed,
        "report": details,
    }


def run_r2egym_eval(
    env_obj: Any,
    instance: Dict[str, Any],
    eval_timeout: Optional[int] = None,
    *,
    test_script_path: str = DEFAULT_TEST_SCRIPT_PATH,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Execute R2E-Gym evaluation inside the container and return the payload.

    This replaces ``_run_eval`` from ``harness/tinyflow_runner/common.py``
    for R2E-Gym tasks.

    Returns ``(eval_payload, raw_test_output)``.
    """
    instance_id = instance.get("instance_id", "unknown")
    timeout = eval_timeout or 300

    try:
        logger.info(
            "[%s] Running R2E-Gym tests: bash %s", instance_id, test_script_path
        )
        result = env_obj.execute(f"bash {test_script_path}", timeout=timeout)
        raw_output = result.output or ""
        test_output = strip_ansi(raw_output)
        logger.info(
            "[%s] Tests finished exit_code=%s output_length=%d",
            instance_id,
            result.exit_code,
            len(test_output),
        )

        expected_json = instance.get("expected_output_json", "{}")
        report = get_r2egym_eval_report(
            test_output,
            expected_json,
            instance_id=instance_id,
        )
        report["eval_exit_code"] = result.exit_code
        return report, test_output

    except Exception as exc:
        logger.exception("[%s] R2E-Gym eval failed: %s", instance_id, exc)
        return (
            {
                "resolved": False,
                "resolved_status": "NO",
                "reward": 0,
                "error": str(exc),
                "status_map": {},
                "report": {},
            },
            None,
        )
