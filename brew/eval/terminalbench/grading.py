"""
Terminal Bench evaluation.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


_PYTEST_STATUS_RE = re.compile(
    r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(\S+)", re.MULTILINE
)


def _parse_pytest_output(output: str) -> Dict[str, str]:
    """Extract {node_id: status} from pytest ``-rA`` short summary lines."""
    if not output:
        return {}
    status_map: Dict[str, str] = {}
    for match in _PYTEST_STATUS_RE.finditer(output):
        status, node_id = match.group(1), match.group(2)
        status_map[node_id] = status
    return status_map


def run_tb_1_eval(
    env_obj: Any,
    instance: Dict[str, Any],
    eval_timeout: Optional[int] = None,
    tests_dir: Optional[str] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Evaluate a Terminal-Bench v1 (task.yaml + run-tests.sh) styled instance.

    Uploads ``tests/`` to ``/tests`` and ``run-tests.sh`` to ``/run-tests.sh``,
    runs it with ``TEST_DIR=/tests``, then parses pytest output for per-test
    statuses. A rollout is resolved only when the script exits 0 and at least
    one test passed.
    """
    instance_id = instance.get("instance_id", "unknown")
    timeout = eval_timeout or 1800
    eval_output: Optional[str] = None

    try:
        run_tests_script = instance.get("run_tests_script")
        if not run_tests_script or not Path(run_tests_script).is_file():
            raise FileNotFoundError(
                f"Terminal-Bench v1 task {instance_id} missing run-tests.sh "
                f"(run_tests_script={run_tests_script})"
            )
        if tests_dir:
            tests_path = Path(tests_dir)
            if not tests_path.exists():
                raise FileNotFoundError(f"Tests directory not found: {tests_dir}")
            logger.info(
                "[%s] Uploading tests from %s to /tests/", instance_id, tests_dir
            )
            env_obj.upload_dir(tests_path, "/tests")

        logger.info(
            "[%s] Uploading run-tests.sh from %s", instance_id, run_tests_script
        )
        env_obj.upload_file(run_tests_script, "/run-tests.sh")
        env_obj.execute("mkdir -p /logs/verifier")

        logger.info(
            "[%s] Executing /run-tests.sh (timeout=%ss)", instance_id, timeout
        )
        eval_result = env_obj.execute(
            "chmod +x /run-tests.sh && TEST_DIR=/tests bash /run-tests.sh",
            timeout=timeout,
        )
        eval_output = eval_result.output
        eval_exit_code = eval_result.exit_code

        status_map = _parse_pytest_output(eval_output or "")
        passed = sum(1 for s in status_map.values() if s in ("PASSED", "XFAIL"))
        failed = sum(1 for s in status_map.values() if s in ("FAILED", "ERROR"))
        resolved = eval_exit_code == 0 and passed > 0 and failed == 0
        reward = 1 if resolved else 0
        resolved_status = "resolved" if resolved else "unresolved"

        logger.info(
            "[%s] Terminal-Bench v1 eval: exit=%s passed=%d failed=%d resolved=%s",
            instance_id,
            eval_exit_code,
            passed,
            failed,
            resolved,
        )

        return (
            {
                "resolved": resolved,
                "resolved_status": resolved_status,
                "reward": reward,
                "eval_exit_code": eval_exit_code,
                "status_map": status_map,
                "report": {"passed": passed, "failed": failed},
            },
            eval_output,
        )
    except Exception as exc:
        logger.exception("[%s] Terminal-Bench v1 eval failed: %s", instance_id, exc)
        return (
            {
                "resolved": False,
                "resolved_status": "error",
                "reward": 0,
                "error": str(exc),
                "status_map": {},
                "report": {},
            },
            eval_output,
        )


def run_tb_eval(
    env_obj: Any,
    instance: Dict[str, Any],
    eval_timeout: Optional[int] = None,
    tests_dir: Optional[str] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Run Terminal Bench evaluation inside the container.

    Args:
        env_obj: Environment object with execute() and upload_dir().
        instance: Instance dict from the manifest.
        eval_timeout: Timeout in seconds for running test.sh.
        tests_dir: Local path to the tests directory to upload.
            If provided, uploads to /tests/ before running.
            If None, assumes tests are already in the image.

    Returns (eval_payload, eval_output) matching build_eval_payload() format.
    """
    if instance.get("task_format") in ("tb_1", "tb_legacy"):
        return run_tb_1_eval(
            env_obj=env_obj,
            instance=instance,
            eval_timeout=eval_timeout,
            tests_dir=tests_dir,
        )

    instance_id = instance.get("instance_id", "unknown")
    eval_output: Optional[str] = None
    timeout = eval_timeout or 1800

    try:
        # Upload test files if a tests_dir is provided
        if tests_dir:
            tests_path = Path(tests_dir)
            if not tests_path.exists():
                raise FileNotFoundError(f"Tests directory not found: {tests_dir}")
            logger.info("[%s] Uploading tests from %s to /tests/", instance_id, tests_dir)
            env_obj.upload_dir(tests_path, "/tests")

        # Ensure the reward directory exists
        env_obj.execute("mkdir -p /logs/verifier")

        # Run the test script
        logger.info(
            "[%s] Executing /tests/test.sh (timeout=%ss)", instance_id, timeout
        )
        eval_result = env_obj.execute(
            "chmod +x /tests/test.sh && bash /tests/test.sh",
            timeout=timeout,
        )
        eval_output = eval_result.output
        eval_exit_code = eval_result.exit_code
        logger.info(
            "[%s] test.sh finished with exit_code=%s, output_length=%d",
            instance_id,
            eval_exit_code,
            len(eval_output) if eval_output else 0,
        )

        # Read reward from the file written by test.sh
        reward_result = env_obj.execute("cat /logs/verifier/reward.txt")
        reward_text = (reward_result.output or "").strip()

        try:
            reward = int(reward_text)
        except (ValueError, TypeError):
            try:
                reward = int(float(reward_text))
            except (ValueError, TypeError):
                logger.warning(
                    "[%s] Could not parse reward from '%s', defaulting to 0",
                    instance_id,
                    reward_text,
                )
                reward = 0

        resolved = reward >= 1
        resolved_status = "resolved" if resolved else "unresolved"

        eval_payload = {
            "resolved": resolved,
            "resolved_status": resolved_status,
            "reward": reward,
            "eval_exit_code": eval_exit_code,
            "status_map": {},
            "report": {},
        }
        logger.info(
            "[%s] Eval result: resolved=%s, resolved_status=%s, reward=%s",
            instance_id,
            resolved,
            resolved_status,
            reward,
        )
        return eval_payload, eval_output

    except Exception as exc:
        logger.exception("[%s] Eval failed with error: %s", instance_id, exc)
        return (
            {
                "resolved": False,
                "resolved_status": "error",
                "reward": 0,
                "error": str(exc),
                "status_map": {},
                "report": {},
            },
            eval_output,
        )