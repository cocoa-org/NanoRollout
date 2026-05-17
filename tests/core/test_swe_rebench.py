"""SWE-rebench eval and oracle checks.

The oracle test is intentionally opt-in. Running the full filtered split creates
one container/sandbox per SWE-rebench task.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shlex
import sys
import threading
import traceback
from typing import Any

import pytest

from nanorollout.adapters.swe.common import create_environment
from nanorollout.adapters.swe.task.datasets import resolve_swe_dataset_adapter
from nanorollout.adapters.swe.task.rebench import (
    _build_eval_script,
    _build_report,
    _install_command,
    _modified_files,
    _resolve_parser,
    _test_command,
    _write_patch_file,
)


RUN_ORACLE_ENV = "NANOROLLOUT_RUN_SWEREBENCH_ORACLE"
_FAILURE_WRITE_LOCK = threading.Lock()


def _reset_repo_command(instance: dict[str, Any]) -> str:
    base_commit = shlex.quote(str(instance["base_commit"]))
    return f"git reset --hard {base_commit}\ngit clean -fd"


def test_rebench_eval_script_reinstalls_and_targets_test_patch_files() -> None:
    instance = {
        "base_commit": "abc123",
        "repo": "owner/repo",
        "test_patch": (
            "diff --git a/tests/test_widget.py b/tests/test_widget.py\n"
            "index 0000000..1111111 100644\n"
            "--- a/tests/test_widget.py\n"
            "+++ b/tests/test_widget.py\n"
            "@@ -1 +1 @@\n"
            "-def test_old(): pass\n"
            "+def test_new(): pass\n"
        ),
        "install_config": {
            "install": "pip install -e .[dev]",
            "test_cmd": "pytest -q",
        },
    }

    script = _build_eval_script(instance, "/testbed")

    assert script.index("pip install -e .[dev]") < script.index("git apply -v -")
    assert "pytest -q tests/test_widget.py" in script


def test_rebench_eval_script_can_apply_test_patch_from_file() -> None:
    instance = {
        "base_commit": "abc123",
        "repo": "owner/repo",
        "test_patch": (
            "diff --git a/tests/test_widget.py b/tests/test_widget.py\n"
            "--- a/tests/test_widget.py\n"
            "+++ b/tests/test_widget.py\n"
            "@@ -1 +1 @@\n"
            "-def test_old(): pass\n"
            "+def test_new(): pass\n"
        ),
        "install_config": {
            "install": "pip install -e .[dev]",
            "test_cmd": "pytest -q",
        },
    }

    script = _build_eval_script(
        instance,
        "/testbed",
        test_patch_path="/tmp/nanorollout_swerebench/test.patch",
    )

    assert (
        "git apply -v --whitespace=nowarn "
        "/tmp/nanorollout_swerebench/test.patch"
    ) in script
    assert "diff --git a/tests/test_widget.py" not in script


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _apply_patch_file_command(label: str, patch_path: str | None) -> str:
    if not patch_path:
        return f"echo 'No {label} patch'"
    return (
        "git apply -v --whitespace=nowarn "
        f"{shlex.quote(patch_path)} || exit 2"
    )


def _reset_tests_command(instance: dict[str, Any]) -> str:
    test_files = _modified_files(str(instance.get("test_patch") or ""))
    if not test_files:
        return "echo 'No test files to reset'"
    base_commit = shlex.quote(str(instance["base_commit"]))
    files = " ".join(shlex.quote(path) for path in test_files)
    return f"git checkout {base_commit} {files}"


def _shell_prefix(workspace_dir: str) -> str:
    quoted_workspace = shlex.quote(workspace_dir)
    return "\n".join(
        [
            "source /opt/miniconda3/bin/activate",
            "conda activate testbed",
            f"cd {quoted_workspace}",
            f"git config --global --add safe.directory {quoted_workspace}",
        ]
    )


def _execute(env_obj: Any, workspace_dir: str, command: str, timeout: int):
    return env_obj.execute(
        f"{_shell_prefix(workspace_dir)}\n{command}",
        timeout=timeout,
    )


def _parse_status_map(instance: dict[str, Any], output: str) -> dict[str, str]:
    parser_name = (instance.get("install_config") or {}).get(
        "log_parser",
        "parse_log_pytest",
    )
    return _resolve_parser(str(parser_name))(output)


def _assert_tests_present(
    *,
    instance_id: str,
    phase: str,
    status_map: dict[str, str],
    tests: list[str],
) -> None:
    missing = [test for test in tests if test not in status_map]
    assert not missing, (
        f"{instance_id} {phase}: parser did not report expected tests: "
        f"{missing[:10]}"
    )


def _write_failure(
    *,
    failures_path: str | None,
    instance_id: str,
    exc: BaseException,
) -> None:
    if not failures_path:
        return
    payload = {
        "instance_id": instance_id,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }
    with _FAILURE_WRITE_LOCK:
        path = Path(failures_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _run_swerebench_oracle_instance(
    *,
    instance: dict[str, Any],
    dataset: Any,
    env_type: str,
    timeout: int,
    workspace_dir: str,
) -> str:
    instance_id = str(instance["instance_id"])
    image = dataset.image_name(instance, env_type, None)
    install_config = instance.get("install_config") or {}
    test_cmd = _test_command(instance, install_config["test_cmd"])
    fail_to_pass = _coerce_list(instance.get("FAIL_TO_PASS"))
    pass_to_pass = _coerce_list(instance.get("PASS_TO_PASS"))

    env_obj = create_environment(
        env_type=env_type,
        instance=instance,
        image=image,
        workspace_dir=workspace_dir,
        env_timeout=timeout,
        create_timeout=timeout,
        step_timeout=timeout,
        eval_timeout=timeout,
    )
    try:
        env_obj.start()

        install = _install_command(install_config)
        if install:
            install_result = _execute(env_obj, workspace_dir, install, timeout)
            assert install_result.exit_code == 0, (
                f"{instance_id} install failed:\n{install_result.output[-4000:]}"
            )

        test_patch_path = _write_patch_file(
            env_obj,
            "test",
            str(instance.get("test_patch") or ""),
        )
        apply_tests = _execute(
            env_obj,
            workspace_dir,
            "\n".join(
                [
                    _reset_tests_command(instance),
                    _apply_patch_file_command("test", test_patch_path),
                ]
            ),
            timeout,
        )
        assert apply_tests.exit_code == 0, (
            f"{instance_id} test_patch apply failed:\n{apply_tests.output[-4000:]}"
        )

        before = _execute(env_obj, workspace_dir, test_cmd, timeout)
        before_status = _parse_status_map(instance, before.output)
        _assert_tests_present(
            instance_id=instance_id,
            phase="before gold patch",
            status_map=before_status,
            tests=fail_to_pass + pass_to_pass,
        )
        unexpectedly_passed = [
            test for test in fail_to_pass if before_status.get(test) == "PASSED"
        ]
        assert not unexpectedly_passed, (
            f"{instance_id} before gold patch: FAIL_TO_PASS unexpectedly passed: "
            f"{unexpectedly_passed[:10]}"
        )
        broken_maintenance = [
            test for test in pass_to_pass if before_status.get(test) != "PASSED"
        ]
        assert not broken_maintenance, (
            f"{instance_id} before gold patch: PASS_TO_PASS did not pass: "
            f"{broken_maintenance[:10]}"
        )

        gold_patch_path = _write_patch_file(
            env_obj,
            "gold",
            str(instance.get("patch") or ""),
        )
        prepare_gold = _execute(
            env_obj,
            workspace_dir,
            "\n".join(
                [
                    _reset_repo_command(instance),
                    _apply_patch_file_command("gold", gold_patch_path),
                    _reset_tests_command(instance),
                    _apply_patch_file_command("test", test_patch_path),
                ]
            ),
            timeout,
        )
        assert prepare_gold.exit_code == 0, (
            f"{instance_id} gold/test patch setup failed:\n"
            f"{prepare_gold.output[-4000:]}"
        )

        after = _execute(env_obj, workspace_dir, test_cmd, timeout)
        after_status = _parse_status_map(instance, after.output)
        _assert_tests_present(
            instance_id=instance_id,
            phase="after gold patch",
            status_map=after_status,
            tests=fail_to_pass + pass_to_pass,
        )
        report, resolved_status = _build_report(instance, after_status)
        assert resolved_status == "RESOLVED_FULL", (
            f"{instance_id} after gold patch was {resolved_status}; "
            f"report={report}"
        )
        return instance_id
    finally:
        env_obj.stop()


def _run_swerebench_oracle_instances(
    *,
    instances: list[dict[str, Any]],
    dataset: Any,
    env_type: str,
    timeout: int,
    workspace_dir: str,
    concurrency: int,
    failures_path: str | None,
    progress_every: int,
) -> None:
    failures: list[str] = []
    total = len(instances)

    def record_progress(completed: int, instance_id: str) -> None:
        if progress_every <= 0:
            return
        if completed == total or completed % progress_every == 0:
            print(
                "[swerebench oracle] "
                f"{completed}/{total} complete; failures={len(failures)}; "
                f"last={instance_id}",
                flush=True,
            )

    if concurrency <= 1:
        for completed, instance in enumerate(instances, start=1):
            instance_id = str(instance["instance_id"])
            try:
                _run_swerebench_oracle_instance(
                    instance=instance,
                    dataset=dataset,
                    env_type=env_type,
                    timeout=timeout,
                    workspace_dir=workspace_dir,
                )
            except Exception as exc:
                failures.append(f"{instance_id}: {exc}")
                _write_failure(
                    failures_path=failures_path,
                    instance_id=instance_id,
                    exc=exc,
                )
            record_progress(completed, instance_id)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    _run_swerebench_oracle_instance,
                    instance=instance,
                    dataset=dataset,
                    env_type=env_type,
                    timeout=timeout,
                    workspace_dir=workspace_dir,
                ): str(instance["instance_id"])
                for instance in instances
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                instance_id = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failures.append(f"{instance_id}: {exc}")
                    _write_failure(
                        failures_path=failures_path,
                        instance_id=instance_id,
                        exc=exc,
                    )
                record_progress(completed, instance_id)

    if failures:
        sample = "\n\n".join(failures[:20])
        pytest.fail(
            f"{len(failures)}/{total} SWE-rebench oracle checks failed. "
            f"First {min(20, len(failures))} failures:\n\n{sample}"
        )


def test_swerebench_oracle_records_failures_jsonl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_worker(**kwargs: Any) -> str:
        del kwargs
        raise AssertionError("boom")

    monkeypatch.setattr(
        sys.modules[__name__],
        "_run_swerebench_oracle_instance",
        fail_worker,
    )
    failures_path = tmp_path / "nested" / "failures.jsonl"

    with pytest.raises(pytest.fail.Exception):
        _run_swerebench_oracle_instances(
            instances=[{"instance_id": "fake-1"}],
            dataset=object(),
            env_type="docker",
            timeout=1,
            workspace_dir="/testbed",
            concurrency=2,
            failures_path=str(failures_path),
            progress_every=0,
        )

    rows = [
        json.loads(line)
        for line in failures_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [
        {
            "instance_id": "fake-1",
            "error_type": "AssertionError",
            "error": "boom",
            "traceback": rows[0]["traceback"],
        }
    ]
    assert "AssertionError: boom" in rows[0]["traceback"]


def _oracle_env_value(name: str, default: str | None = None) -> str | None:
    return os.environ.get(f"NANOROLLOUT_SWEREBENCH_ORACLE_{name}", default)


def _run_swerebench_filtered_ab_all_tasks() -> None:
    dataset = resolve_swe_dataset_adapter("rebench")
    split = _oracle_env_value("SPLIT", "filtered") or "filtered"
    env_type = _oracle_env_value("ENV", "docker") or "docker"
    timeout = int(_oracle_env_value("TIMEOUT", "1800") or "1800")
    concurrency = int(_oracle_env_value("CONCURRENCY", "1") or "1")
    failures_path = _oracle_env_value("FAILURES_PATH")
    progress_every = int(_oracle_env_value("PROGRESS_EVERY", "10") or "10")
    workspace_dir = dataset.workspace_dir()

    instances = dataset.load_instances(split)
    selected_ids = {
        item.strip()
        for item in (_oracle_env_value("INSTANCE_IDS", "") or "").split(",")
        if item.strip()
    }
    if selected_ids:
        instances = [
            instance
            for instance in instances
            if str(instance.get("instance_id")) in selected_ids
        ]

    offset = int(_oracle_env_value("OFFSET", "0") or "0")
    limit = _oracle_env_value("LIMIT")
    if offset:
        instances = instances[offset:]
    if limit:
        instances = instances[: int(limit)]

    assert instances, "No SWE-rebench instances selected for oracle test"

    _run_swerebench_oracle_instances(
        instances=instances,
        dataset=dataset,
        env_type=env_type,
        timeout=timeout,
        workspace_dir=workspace_dir,
        concurrency=max(1, concurrency),
        failures_path=failures_path,
        progress_every=progress_every,
    )


@pytest.mark.skipif(
    os.environ.get(RUN_ORACLE_ENV) != "1",
    reason=f"set {RUN_ORACLE_ENV}=1 to run all SWE-rebench filtered oracle checks",
)
def test_swerebench_filtered_ab_all_tasks() -> None:
    _run_swerebench_filtered_ab_all_tasks()


@pytest.mark.skipif(
    os.environ.get(RUN_ORACLE_ENV) != "1",
    reason=f"set {RUN_ORACLE_ENV}=1 to run all SWE-rebench filtered oracle checks",
)
def test_swerebench_filtered_oracle_all_tasks() -> None:
    _run_swerebench_filtered_ab_all_tasks()
