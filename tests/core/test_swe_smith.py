"""SWE-Smith eval and oracle checks.

The oracle test is intentionally opt-in. By default it covers the language
specific SWE-Smith datasets except Python.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import threading
import traceback
from typing import Any

import pytest

from nanorollout.adapters.swe.common import create_environment
from nanorollout.adapters.swe.task.datasets import resolve_swe_dataset_adapter
from nanorollout.adapters.swe.task.swebench import (
    _build_swesmith_eval_script,
    _extract_marked_output,
)


RUN_ORACLE_ENV = "NANOROLLOUT_RUN_SWESMITH_ORACLE"
NON_PY_SWESMITH_DATASETS = (
    "smith-go",
    "smith-rs",
    "smith-cpp",
    "smith-java",
    "smith-js",
    "smith-ts",
    "smith-php",
)
_FAILURE_WRITE_LOCK = threading.Lock()


def test_swesmith_eval_script_restores_hidden_test_files() -> None:
    script = _build_swesmith_eval_script("/testbed", "go test -v ./...")

    assert "git diff --name-only -z HEAD~1..HEAD" in script
    assert 'git checkout HEAD~1 -- "${NANOROLLOUT_SWESMITH_TEST_FILES[@]}"' in script
    assert "go test -v ./..." in script


def _oracle_env_value(name: str, default: str | None = None) -> str | None:
    return os.environ.get(f"NANOROLLOUT_SWESMITH_ORACLE_{name}", default)


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _write_env_file(env_obj: Any, path: str, content: str) -> None:
    parent = str(Path(path).parent)
    if parent and parent != ".":
        mkdir = env_obj.execute(f"mkdir -p {shlex.quote(parent)}")
        assert mkdir.exit_code == 0, f"failed to create {parent}: {mkdir.output}"

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
    assert result.exit_code == 0, f"failed to write {path}: {result.output}"


def _write_patch_file(env_obj: Any, patch: str) -> str | None:
    if not patch.strip():
        return None
    path = "/tmp/nanorollout_swesmith/gold.patch"
    _write_env_file(env_obj, path, patch)
    return path


def _reset_repo_command(instance_id: str) -> str:
    quoted_instance_id = shlex.quote(instance_id)
    return "\n".join(
        [
            "git reset --hard",
            "git clean -fd",
            (
                f"git checkout {quoted_instance_id} || "
                f"(git fetch --all --prune && git checkout {quoted_instance_id})"
            ),
            "git reset --hard",
            "git clean -fd",
        ]
    )


def _apply_reverse_patch_command(patch_path: str | None) -> str:
    if not patch_path:
        return "echo 'No SWE-Smith gold patch'"
    quoted_path = shlex.quote(patch_path)
    return (
        "git apply -v --whitespace=nowarn --reverse "
        f"{quoted_path} || "
        "git apply -v --reverse --reject "
        f"{quoted_path} || "
        f"patch --batch --fuzz=5 -p1 -R -i {quoted_path} || exit 2"
    )


def _shell_prefix(workspace_dir: str) -> str:
    quoted_workspace = shlex.quote(workspace_dir)
    return "\n".join(
        [
            (
                "export PATH=/go/bin:/usr/local/go/bin:/usr/local/cargo/bin:"
                "/root/.cargo/bin:/root/.local/bin:$PATH"
            ),
            f"cd {quoted_workspace}",
            f"git config --global --add safe.directory {quoted_workspace}",
        ]
    )


def _execute(env_obj: Any, workspace_dir: str, command: str, timeout: int):
    return env_obj.execute(
        f"{_shell_prefix(workspace_dir)}\n{command}",
        timeout=timeout,
    )


def _run_swesmith_tests(
    *,
    env_obj: Any,
    instance: dict[str, Any],
    workspace_dir: str,
    timeout: int,
) -> tuple[dict[str, str], str]:
    from swesmith.constants import TEST_OUTPUT_END, TEST_OUTPUT_START
    from swesmith.profiles import registry

    profile = registry.get_from_inst(instance)
    test_cmd, _ = profile.get_test_cmd(instance)
    result = env_obj.execute(
        _build_swesmith_eval_script(workspace_dir, test_cmd),
        timeout=timeout or profile.timeout,
    )
    output = result.output or ""
    test_output = _extract_marked_output(
        output,
        TEST_OUTPUT_START,
        TEST_OUTPUT_END,
    )
    return profile.log_parser(test_output), output


def _is_passed(status: str | None) -> bool:
    return str(status or "").upper() in {"PASSED", "XFAIL"}


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


def _build_report(
    instance: dict[str, Any],
    status_map: dict[str, str],
) -> tuple[dict[str, Any], str]:
    from swebench.harness.grading import get_resolution_status
    from swesmith.harness.grading import get_eval_tests_report

    report = get_eval_tests_report(status_map, instance)
    return report, get_resolution_status(report)


def _write_failure(
    *,
    failures_path: str | None,
    dataset_name: str,
    instance_id: str,
    exc: BaseException,
) -> None:
    if not failures_path:
        return
    payload = {
        "dataset": dataset_name,
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


def _run_swesmith_oracle_instance(
    *,
    dataset_name: str,
    instance: dict[str, Any],
    dataset: Any,
    env_type: str,
    timeout: int,
    workspace_dir: str,
) -> str:
    del dataset_name
    instance_id = str(instance["instance_id"])
    image = dataset.image_name(instance, env_type, None)
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
        gold_patch_path = _write_patch_file(
            env_obj,
            str(instance.get("patch") or ""),
        )

        before_prepare = _execute(
            env_obj,
            workspace_dir,
            _reset_repo_command(instance_id),
            timeout,
        )
        assert before_prepare.exit_code == 0, (
            f"{instance_id} reset before A failed:\n"
            f"{before_prepare.output[-4000:]}"
        )

        before_status, before_output = _run_swesmith_tests(
            env_obj=env_obj,
            instance=instance,
            workspace_dir=workspace_dir,
            timeout=timeout,
        )
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
                    _reset_repo_command(instance_id),
                    _apply_reverse_patch_command(gold_patch_path),
                ]
            ),
            timeout,
        )
        assert after_prepare.exit_code == 0, (
            f"{instance_id} gold patch setup failed:\n"
            f"{after_prepare.output[-4000:]}"
        )

        after_status, after_output = _run_swesmith_tests(
            env_obj=env_obj,
            instance=instance,
            workspace_dir=workspace_dir,
            timeout=timeout,
        )
        _assert_tests_present(
            instance_id=instance_id,
            phase="after gold patch",
            status_map=after_status,
            tests=fail_to_pass + pass_to_pass,
            output=after_output,
        )
        report, resolved_status = _build_report(instance, after_status)
        assert resolved_status == "RESOLVED_FULL", (
            f"{instance_id} after gold patch was {resolved_status}; "
            f"report={report}"
        )
        return instance_id
    finally:
        env_obj.stop()


def _run_swesmith_oracle_instances(
    *,
    tasks: list[tuple[str, Any, dict[str, Any]]],
    env_type: str,
    timeout: int,
    concurrency: int,
    failures_path: str | None,
    progress_every: int,
) -> None:
    failures: list[str] = []
    total = len(tasks)

    def record_progress(completed: int, dataset_name: str, instance_id: str) -> None:
        if progress_every <= 0:
            return
        if completed == total or completed % progress_every == 0:
            print(
                "[swesmith oracle] "
                f"{completed}/{total} complete; failures={len(failures)}; "
                f"last={dataset_name}:{instance_id}",
                flush=True,
            )

    if concurrency <= 1:
        for completed, (dataset_name, dataset, instance) in enumerate(tasks, start=1):
            instance_id = str(instance["instance_id"])
            try:
                _run_swesmith_oracle_instance(
                    dataset_name=dataset_name,
                    instance=instance,
                    dataset=dataset,
                    env_type=env_type,
                    timeout=timeout,
                    workspace_dir=dataset.workspace_dir(),
                )
            except Exception as exc:
                failures.append(f"{dataset_name}:{instance_id}: {exc}")
                _write_failure(
                    failures_path=failures_path,
                    dataset_name=dataset_name,
                    instance_id=instance_id,
                    exc=exc,
                )
            record_progress(completed, dataset_name, instance_id)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    _run_swesmith_oracle_instance,
                    dataset_name=dataset_name,
                    instance=instance,
                    dataset=dataset,
                    env_type=env_type,
                    timeout=timeout,
                    workspace_dir=dataset.workspace_dir(),
                ): (dataset_name, str(instance["instance_id"]))
                for dataset_name, dataset, instance in tasks
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                dataset_name, instance_id = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failures.append(f"{dataset_name}:{instance_id}: {exc}")
                    _write_failure(
                        failures_path=failures_path,
                        dataset_name=dataset_name,
                        instance_id=instance_id,
                        exc=exc,
                    )
                record_progress(completed, dataset_name, instance_id)

    if failures:
        sample = "\n\n".join(failures[:20])
        pytest.fail(
            f"{len(failures)}/{total} SWE-Smith oracle checks failed. "
            f"First {min(20, len(failures))} failures:\n\n{sample}"
        )


def test_swesmith_oracle_records_failures_jsonl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_worker(**kwargs: Any) -> str:
        del kwargs
        raise AssertionError("boom")

    monkeypatch.setattr(
        sys.modules[__name__],
        "_run_swesmith_oracle_instance",
        fail_worker,
    )
    failures_path = tmp_path / "nested" / "failures.jsonl"
    dataset = resolve_swe_dataset_adapter("smith-go")

    with pytest.raises(pytest.fail.Exception):
        _run_swesmith_oracle_instances(
            tasks=[("smith-go", dataset, {"instance_id": "fake-1"})],
            env_type="docker",
            timeout=1,
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
            "dataset": "smith-go",
            "instance_id": "fake-1",
            "error_type": "AssertionError",
            "error": "boom",
            "traceback": rows[0]["traceback"],
        }
    ]
    assert "AssertionError: boom" in rows[0]["traceback"]


def _oracle_dataset_names() -> list[str]:
    value = _oracle_env_value("DATASETS")
    if not value:
        return list(NON_PY_SWESMITH_DATASETS)
    return [item.strip() for item in value.split(",") if item.strip()]


def _run_swesmith_non_py_ab_all_tasks() -> None:
    split = _oracle_env_value("SPLIT", "train") or "train"
    env_type = _oracle_env_value("ENV", "docker") or "docker"
    timeout = int(_oracle_env_value("TIMEOUT", "900") or "900")
    concurrency = int(_oracle_env_value("CONCURRENCY", "1") or "1")
    failures_path = _oracle_env_value("FAILURES_PATH")
    progress_every = int(_oracle_env_value("PROGRESS_EVERY", "25") or "25")
    selected_ids = {
        item.strip()
        for item in (_oracle_env_value("INSTANCE_IDS", "") or "").split(",")
        if item.strip()
    }

    tasks: list[tuple[str, Any, dict[str, Any]]] = []
    for dataset_name in _oracle_dataset_names():
        dataset = resolve_swe_dataset_adapter(dataset_name)
        instances = dataset.load_instances(split)
        if selected_ids:
            instances = [
                instance
                for instance in instances
                if str(instance.get("instance_id")) in selected_ids
            ]
        tasks.extend((dataset_name, dataset, instance) for instance in instances)

    offset = int(_oracle_env_value("OFFSET", "0") or "0")
    limit = _oracle_env_value("LIMIT")
    if offset:
        tasks = tasks[offset:]
    if limit:
        tasks = tasks[: int(limit)]

    assert tasks, "No SWE-Smith instances selected for oracle test"

    _run_swesmith_oracle_instances(
        tasks=tasks,
        env_type=env_type,
        timeout=timeout,
        concurrency=max(1, concurrency),
        failures_path=failures_path,
        progress_every=progress_every,
    )


@pytest.mark.skipif(
    os.environ.get(RUN_ORACLE_ENV) != "1",
    reason=f"set {RUN_ORACLE_ENV}=1 to run all non-Python SWE-Smith oracle checks",
)
def test_swesmith_non_py_ab_all_tasks() -> None:
    _run_swesmith_non_py_ab_all_tasks()
