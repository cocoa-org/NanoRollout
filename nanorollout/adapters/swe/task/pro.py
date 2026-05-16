"""SWE-Bench Pro task helpers."""

from __future__ import annotations

import ast
import json
import logging
import os
import shlex
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_JSON_START = "NANOROLLOUT_SWEBENCH_PRO_JSON_START"
_JSON_END = "NANOROLLOUT_SWEBENCH_PRO_JSON_END"


def decode_literal_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in {"'", '"'}:
        return value
    try:
        decoded = ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return value
    return decoded if isinstance(decoded, str) else value


def parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str):
        return [str(value)]
    value = value.strip()
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return [value]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def resolve_scripts_dir(extra_args: dict[str, Any]) -> Optional[Path]:
    scripts_dir = (
        extra_args.get("swebench_pro_scripts_dir")
        or extra_args.get("pro_scripts_dir")
        or os.environ.get("SWEBENCH_PRO_SCRIPTS_DIR")
    )
    if scripts_dir:
        return Path(str(scripts_dir)).expanduser()

    repo_dir = (
        extra_args.get("swebench_pro_repo")
        or extra_args.get("pro_repo")
        or os.environ.get("SWEBENCH_PRO_REPO")
    )
    if repo_dir:
        return Path(str(repo_dir)).expanduser() / "run_scripts"
    return None


def _read_script(scripts_dir: Path, instance_id: str, filename: str) -> str:
    path = scripts_dir / instance_id / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"SWE-Bench Pro script not found: {path}. "
            "Point swebench_pro_scripts_dir at the official run_scripts directory."
        )
    return path.read_text(encoding="utf-8")


def _heredoc(path: str, content: str) -> str:
    delimiter = "NANOROLLOUT_EOF"
    while delimiter in content:
        delimiter += "_X"
    return f"cat > {shlex.quote(path)} <<'{delimiter}'\n{content}\n{delimiter}"


def _apply_patch_command(patch: str) -> str:
    patch = patch or ""
    if not patch.strip():
        return "echo 'No SWE-Bench Pro test patch'"
    return (
        _heredoc("/tmp/nanorollout_swebench_pro/test.patch", patch)
        + "\n"
        + (
            "git apply -v --whitespace=nowarn "
            "/tmp/nanorollout_swebench_pro/test.patch || exit 2"
        )
    )


def _extract_json(output: str) -> dict[str, Any]:
    if _JSON_START not in output or _JSON_END not in output:
        raise ValueError("SWE-Bench Pro parser output markers were not found")
    raw_json = output.split(_JSON_START, 1)[1].split(_JSON_END, 1)[0].strip()
    return json.loads(raw_json)


def _tests_by_status(output_json: dict[str, Any], status: str) -> set[str]:
    return {
        str(test.get("name"))
        for test in output_json.get("tests", [])
        if str(test.get("status", "")).upper() == status
    }


def _report(instance: dict[str, Any], output_json: dict[str, Any]) -> dict[str, Any]:
    passed = _tests_by_status(output_json, "PASSED")
    fail_to_pass = set(
        parse_list(instance.get("fail_to_pass") or instance.get("FAIL_TO_PASS"))
    )
    pass_to_pass = set(
        parse_list(instance.get("pass_to_pass") or instance.get("PASS_TO_PASS"))
    )
    return {
        "FAIL_TO_PASS": {
            "success": sorted(fail_to_pass & passed),
            "failure": sorted(fail_to_pass - passed),
        },
        "PASS_TO_PASS": {
            "success": sorted(pass_to_pass & passed),
            "failure": sorted(pass_to_pass - passed),
        },
    }


def run_swebench_pro_eval(
    env_obj: Any,
    instance: dict[str, Any],
    eval_timeout: Optional[int],
    workspace_dir: str,
    scripts_dir: Path,
) -> tuple[dict[str, Any], Optional[str]]:
    instance_id = str(instance.get("instance_id", "unknown"))
    eval_output = None
    try:
        run_script = _read_script(scripts_dir, instance_id, "run_script.sh")
        parser = _read_script(scripts_dir, instance_id, "parser.py")
        selected_tests = ",".join(
            parse_list(instance.get("selected_test_files_to_run"))
        )
        work_dir = "/tmp/nanorollout_swebench_pro"
        eval_script = "\n".join(
            [
                "#!/bin/bash",
                "set -uxo pipefail",
                f"mkdir -p {work_dir}",
                _heredoc(f"{work_dir}/run_script.sh", run_script),
                _heredoc(f"{work_dir}/parser.py", parser),
                f"chmod +x {work_dir}/run_script.sh",
                f"cd {shlex.quote(workspace_dir)}",
                "git config --global --add safe.directory "
                f"{shlex.quote(workspace_dir)}",
                _apply_patch_command(str(instance.get("test_patch") or "")),
                (
                    f"bash {work_dir}/run_script.sh {shlex.quote(selected_tests)} "
                    f"> {work_dir}/stdout.log 2> {work_dir}/stderr.log"
                ),
                (
                    f"python {work_dir}/parser.py {work_dir}/stdout.log "
                    f"{work_dir}/stderr.log {work_dir}/output.json"
                ),
                f"echo {_JSON_START}",
                f"cat {work_dir}/output.json",
                f"echo {_JSON_END}",
            ]
        )
        result = env_obj.execute(eval_script, timeout=eval_timeout or 3600)
        eval_output = result.output or ""
        output_json = _extract_json(eval_output)
        report = _report(instance, output_json)
        resolved = (
            not report["FAIL_TO_PASS"]["failure"]
            and not report["PASS_TO_PASS"]["failure"]
        )
        return (
            {
                "resolved": resolved,
                "resolved_status": "RESOLVED_FULL" if resolved else "RESOLVED_NO",
                "reward": 1 if resolved else 0,
                "eval_exit_code": result.exit_code,
                "status_map": {
                    str(test.get("name")): str(test.get("status"))
                    for test in output_json.get("tests", [])
                },
                "report": report,
                "output": output_json,
            },
            eval_output,
        )
    except Exception as exc:
        logger.exception("[%s] SWE-Bench Pro eval failed: %s", instance_id, exc)
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
