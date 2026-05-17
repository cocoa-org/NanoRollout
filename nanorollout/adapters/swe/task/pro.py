"""SWE-Bench Pro task helpers."""

from __future__ import annotations

import ast
import json
import logging
import os
import shlex
import tempfile
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_WORK_DIR = "/tmp/nanorollout_swebench_pro"
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
            suffix=".txt",
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


def _runtime_path(filename: str) -> str:
    return f"{_WORK_DIR}/{filename}"


def _write_patch_file(env_obj: Any, label: str, patch: str) -> Optional[str]:
    patch = patch or ""
    if not patch.strip():
        return None
    path = _runtime_path(f"{label}.patch")
    _write_env_file(env_obj, path, patch)
    return path


def _write_runtime_scripts(
    env_obj: Any,
    scripts_dir: Path,
    instance_id: str,
) -> None:
    _write_env_file(
        env_obj,
        _runtime_path("run_script.sh"),
        _read_script(scripts_dir, instance_id, "run_script.sh"),
    )
    _write_env_file(
        env_obj,
        _runtime_path("parser.py"),
        _read_script(scripts_dir, instance_id, "parser.py"),
    )


def _apply_patch_command(label: str, patch_path: Optional[str]) -> str:
    if not patch_path:
        return f"echo 'No SWE-Bench Pro {label} patch'"
    return (
        "git apply -v --whitespace=nowarn "
        f"{shlex.quote(patch_path)} || exit 2"
    )


def _extract_json(output: str) -> dict[str, Any]:
    if _JSON_START not in output or _JSON_END not in output:
        raise ValueError("SWE-Bench Pro parser output markers were not found")
    raw_json = output.split(_JSON_START, 1)[1].split(_JSON_END, 1)[0].strip()
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "SWE-Bench Pro parser output was not JSON: "
            f"{raw_json[-1000:] or '<empty>'}"
        ) from exc


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


def _build_eval_script(
    instance: dict[str, Any],
    workspace_dir: str,
    test_patch_path: Optional[str],
) -> str:
    selected_tests = ",".join(parse_list(instance.get("selected_test_files_to_run")))
    run_script = _runtime_path("run_script.sh")
    parser = _runtime_path("parser.py")
    stdout_log = _runtime_path("stdout.log")
    stderr_log = _runtime_path("stderr.log")
    output_json = _runtime_path("output.json")
    run_command = (
        f"bash {run_script} {shlex.quote(selected_tests)} "
        f"> {stdout_log} 2> {stderr_log}"
        if selected_tests
        else f"bash {run_script} > {stdout_log} 2> {stderr_log}"
    )
    return "\n".join(
        [
            "#!/bin/bash",
            "set -uxo pipefail",
            "export PATH=/go/bin:/usr/local/go/bin:/root/.local/bin:$PATH",
            f"mkdir -p {_WORK_DIR}",
            f"chmod +x {run_script}",
            f"cd {shlex.quote(workspace_dir)}",
            "git config --global --add safe.directory "
            f"{shlex.quote(workspace_dir)}",
            _apply_patch_command("test", test_patch_path),
            run_command,
            f"python3 {parser} {stdout_log} {stderr_log} {output_json}",
            "set +x",
            f"echo {_JSON_START}",
            f"cat {output_json}",
            f"echo {_JSON_END}",
            "echo NANOROLLOUT_SWEBENCH_PRO_STDOUT_TAIL",
            f"tail -c 4000 {stdout_log} || true",
            "echo NANOROLLOUT_SWEBENCH_PRO_STDERR_TAIL",
            f"tail -c 4000 {stderr_log} || true",
        ]
    )


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
        _write_runtime_scripts(env_obj, scripts_dir, instance_id)
        test_patch_path = _write_patch_file(
            env_obj,
            "test",
            str(instance.get("test_patch") or ""),
        )
        eval_script = _build_eval_script(instance, workspace_dir, test_patch_path)
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
