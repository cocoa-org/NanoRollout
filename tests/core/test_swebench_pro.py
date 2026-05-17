"""SWE-Bench Pro eval and oracle checks.

The oracle test is intentionally opt-in. Running the full test split creates
one container/sandbox per SWE-Bench Pro task and requires the official
SWE-Bench Pro run_scripts directory.
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
from nanorollout.adapters.swe.task.pro import (
    _apply_patch_command,
    _build_eval_script,
    _extract_json,
    _report,
    _write_patch_file,
    _write_runtime_scripts,
    parse_list,
    resolve_scripts_dir,
)


RUN_ORACLE_ENV = "NANOROLLOUT_RUN_SWEBENCH_PRO_ORACLE"
_FAILURE_WRITE_LOCK = threading.Lock()


def test_swebench_pro_eval_script_uses_runtime_patch_file() -> None:
    instance = {
        "selected_test_files_to_run": ["tests/test_widget.py"],
        "test_patch": (
            "diff --git a/tests/test_widget.py b/tests/test_widget.py\n"
            "--- a/tests/test_widget.py\n"
            "+++ b/tests/test_widget.py\n"
            "@@ -1 +1 @@\n"
            "-def test_old(): pass\n"
            "+def test_new(): pass\n"
        ),
    }

    script = _build_eval_script(
        instance,
        "/app",
        test_patch_path="/tmp/nanorollout_swebench_pro/test.patch",
    )

    assert (
        "git apply -v --whitespace=nowarn "
        "/tmp/nanorollout_swebench_pro/test.patch"
    ) in script
    assert "diff --git a/tests/test_widget.py" not in script


def _oracle_env_value(name: str, default: str | None = None) -> str | None:
    return os.environ.get(f"NANOROLLOUT_SWEBENCH_PRO_ORACLE_{name}", default)


def _reset_repo_command(instance: dict[str, Any]) -> str:
    base_commit = shlex.quote(str(instance["base_commit"]))
    return "\n".join(
        [
            f"git reset --hard {base_commit}",
            "git clean -fd",
            f"git checkout {base_commit}",
        ]
    )


def _shell_prefix(workspace_dir: str) -> str:
    quoted_workspace = shlex.quote(workspace_dir)
    return "\n".join(
        [
            f"cd {quoted_workspace}",
            f"git config --global --add safe.directory {quoted_workspace}",
        ]
    )


def _execute(env_obj: Any, workspace_dir: str, command: str, timeout: int):
    return env_obj.execute(
        f"{_shell_prefix(workspace_dir)}\n{command}",
        timeout=timeout,
    )


def _status_map(output_json: dict[str, Any]) -> dict[str, str]:
    return {
        str(test.get("name")): str(test.get("status"))
        for test in output_json.get("tests", [])
    }


def _is_passed(status: str | None) -> bool:
    return str(status or "").upper() == "PASSED"


def _assert_tests_present(
    *,
    instance_id: str,
    phase: str,
    status_map: dict[str, str],
    tests: list[str],
    output: str = "",
) -> None:
    missing = [test for test in tests if test not in status_map]
    observed = sorted(status_map)[:20]
    assert not missing, (
        f"{instance_id} {phase}: parser did not report expected tests: "
        f"{missing[:10]}; observed={observed}; output_tail={output[-4000:]}"
    )


def _run_swebench_pro_tests(
    *,
    env_obj: Any,
    instance: dict[str, Any],
    workspace_dir: str,
    timeout: int,
    test_patch_path: str | None,
) -> tuple[dict[str, Any], str]:
    result = env_obj.execute(
        _build_eval_script(instance, workspace_dir, test_patch_path),
        timeout=timeout,
    )
    output = result.output or ""
    return _extract_json(output), output


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


def _run_swebench_pro_oracle_instance(
    *,
    instance: dict[str, Any],
    dataset: Any,
    scripts_dir: Path,
    env_type: str,
    timeout: int,
    workspace_dir: str,
) -> str:
    instance_id = str(instance["instance_id"])
    image = dataset.image_name(instance, env_type, None)
    fail_to_pass = parse_list(
        instance.get("fail_to_pass") or instance.get("FAIL_TO_PASS")
    )
    pass_to_pass = parse_list(
        instance.get("pass_to_pass") or instance.get("PASS_TO_PASS")
    )

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
        _write_runtime_scripts(env_obj, scripts_dir, instance_id)
        test_patch_path = _write_patch_file(
            env_obj,
            "test",
            str(instance.get("test_patch") or ""),
        )
        gold_patch_path = _write_patch_file(
            env_obj,
            "gold",
            str(instance.get("patch") or ""),
        )

        before_prepare = _execute(
            env_obj,
            workspace_dir,
            _reset_repo_command(instance),
            timeout,
        )
        assert before_prepare.exit_code == 0, (
            f"{instance_id} reset before A failed:\n"
            f"{before_prepare.output[-4000:]}"
        )

        before_json, before_output = _run_swebench_pro_tests(
            env_obj=env_obj,
            instance=instance,
            workspace_dir=workspace_dir,
            timeout=timeout,
            test_patch_path=test_patch_path,
        )
        before_status = _status_map(before_json)
        _assert_tests_present(
            instance_id=instance_id,
            phase="before gold patch",
            status_map=before_status,
            tests=pass_to_pass,
            output=before_output,
        )
        unexpectedly_passed = [
            test for test in fail_to_pass if _is_passed(before_status.get(test))
        ]
        assert not unexpectedly_passed, (
            f"{instance_id} before gold patch: FAIL_TO_PASS unexpectedly passed: "
            f"{unexpectedly_passed[:10]}"
        )
        broken_maintenance = [
            test for test in pass_to_pass if not _is_passed(before_status.get(test))
        ]
        assert not broken_maintenance, (
            f"{instance_id} before gold patch: PASS_TO_PASS did not pass: "
            f"{broken_maintenance[:10]}"
        )

        after_prepare = _execute(
            env_obj,
            workspace_dir,
            "\n".join(
                [
                    _reset_repo_command(instance),
                    _apply_patch_command("gold", gold_patch_path),
                ]
            ),
            timeout,
        )
        assert after_prepare.exit_code == 0, (
            f"{instance_id} gold patch setup failed:\n"
            f"{after_prepare.output[-4000:]}"
        )

        after_json, after_output = _run_swebench_pro_tests(
            env_obj=env_obj,
            instance=instance,
            workspace_dir=workspace_dir,
            timeout=timeout,
            test_patch_path=test_patch_path,
        )
        after_status = _status_map(after_json)
        _assert_tests_present(
            instance_id=instance_id,
            phase="after gold patch",
            status_map=after_status,
            tests=fail_to_pass + pass_to_pass,
            output=after_output,
        )
        report = _report(instance, after_json)
        assert not report["FAIL_TO_PASS"]["failure"] and not report["PASS_TO_PASS"][
            "failure"
        ], f"{instance_id} after gold patch did not resolve; report={report}"
        return instance_id
    finally:
        env_obj.stop()


def _run_swebench_pro_oracle_instances(
    *,
    instances: list[dict[str, Any]],
    dataset: Any,
    scripts_dir: Path,
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
                "[swebench-pro oracle] "
                f"{completed}/{total} complete; failures={len(failures)}; "
                f"last={instance_id}",
                flush=True,
            )

    if concurrency <= 1:
        for completed, instance in enumerate(instances, start=1):
            instance_id = str(instance["instance_id"])
            try:
                _run_swebench_pro_oracle_instance(
                    instance=instance,
                    dataset=dataset,
                    scripts_dir=scripts_dir,
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
                    _run_swebench_pro_oracle_instance,
                    instance=instance,
                    dataset=dataset,
                    scripts_dir=scripts_dir,
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
            f"{len(failures)}/{total} SWE-Bench Pro oracle checks failed. "
            f"First {min(20, len(failures))} failures:\n\n{sample}"
        )


def test_swebench_pro_oracle_records_failures_jsonl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_worker(**kwargs: Any) -> str:
        del kwargs
        raise AssertionError("boom")

    monkeypatch.setattr(
        sys.modules[__name__],
        "_run_swebench_pro_oracle_instance",
        fail_worker,
    )
    failures_path = tmp_path / "nested" / "failures.jsonl"

    with pytest.raises(pytest.fail.Exception):
        _run_swebench_pro_oracle_instances(
            instances=[{"instance_id": "fake-1"}],
            dataset=object(),
            scripts_dir=tmp_path,
            env_type="docker",
            timeout=1,
            workspace_dir="/app",
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


def _resolve_oracle_scripts_dir() -> Path:
    extra_args = {}
    scripts_dir = _oracle_env_value("SCRIPTS_DIR")
    repo_dir = _oracle_env_value("REPO")
    if scripts_dir:
        extra_args["swebench_pro_scripts_dir"] = scripts_dir
    if repo_dir:
        extra_args["swebench_pro_repo"] = repo_dir
    resolved = resolve_scripts_dir(extra_args)
    assert resolved is not None, (
        "SWE-Bench Pro oracle test needs official run scripts. Set "
        "SWEBENCH_PRO_SCRIPTS_DIR, SWEBENCH_PRO_REPO, "
        "NANOROLLOUT_SWEBENCH_PRO_ORACLE_SCRIPTS_DIR, or "
        "NANOROLLOUT_SWEBENCH_PRO_ORACLE_REPO."
    )
    return resolved


def _run_swebench_pro_test_oracle_all_tasks() -> None:
    dataset = resolve_swe_dataset_adapter("swebench-pro")
    split = _oracle_env_value("SPLIT", "test") or "test"
    env_type = _oracle_env_value("ENV", "docker") or "docker"
    timeout = int(_oracle_env_value("TIMEOUT", "3600") or "3600")
    concurrency = int(_oracle_env_value("CONCURRENCY", "1") or "1")
    failures_path = _oracle_env_value("FAILURES_PATH")
    progress_every = int(_oracle_env_value("PROGRESS_EVERY", "10") or "10")
    scripts_dir = _resolve_oracle_scripts_dir()
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

    assert instances, "No SWE-Bench Pro instances selected for oracle test"

    _run_swebench_pro_oracle_instances(
        instances=instances,
        dataset=dataset,
        scripts_dir=scripts_dir,
        env_type=env_type,
        timeout=timeout,
        workspace_dir=workspace_dir,
        concurrency=max(1, concurrency),
        failures_path=failures_path,
        progress_every=progress_every,
    )


@pytest.mark.skipif(
    os.environ.get(RUN_ORACLE_ENV) != "1",
    reason=f"set {RUN_ORACLE_ENV}=1 to run all SWE-Bench Pro oracle checks",
)
def test_swebench_pro_test_ab_all_tasks() -> None:
    _run_swebench_pro_test_oracle_all_tasks()
