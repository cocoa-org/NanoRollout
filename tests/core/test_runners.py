from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanorollout.core.models import RunRequest
from nanorollout.core.runners import (
    RUNNER_SPECS,
    build_runner_params,
    load_runner_callable,
    resolve_legacy_runner,
    resolve_runner,
)
from nanorollout.harness.agents.shared.installed_agent_factory import AGENT_REGISTRY
from nanorollout.harness.agents.shared.llm_config import build_llm_config
from nanorollout.runner import TaskAdapter, TaskRunRequest, TaskSpec, run_task
from nanorollout.adapters.cocoa.adapter import CocoaTaskAdapter
from nanorollout.adapters.osworld.adapter import OSWorldTaskAdapter
from nanorollout.adapters.swe.task.datasets import (
    R2EGymDatasetAdapter,
    SWE_REBENCH_REVISION,
    SweBenchProDatasetAdapter,
    SweRebenchDatasetAdapter,
    SweSmithDatasetAdapter,
    resolve_swe_dataset_adapter,
)
from nanorollout.adapters.swe.adapter import SweTaskAdapter, SweTaskSpec
from nanorollout.adapters.terminal.adapter import (
    TerminalTaskAdapter,
    TerminalTaskSpec,
)
import nanorollout.runner as runner_module


def test_shared_reward_payload_is_task_agnostic() -> None:
    payload = runner_module.build_reward_payload(
        "task-1",
        {
            "resolved": True,
            "resolved_status": "FULL",
            "reward": 1,
            "status_map": {"test_a": "PASSED"},
            "report": {"passed": 1},
        },
        None,
        default_status="NO",
    )

    assert payload == {
        "instance_id": "task-1",
        "resolved": True,
        "resolved_status": "FULL",
        "reward": 1,
        "eval_exit_code": None,
        "error": None,
    }


def test_swe_reward_payload_keeps_test_details() -> None:
    from nanorollout.adapters.swe.common import build_reward_payload

    payload = build_reward_payload(
        "swe-1",
        {
            "resolved": False,
            "resolved_status": "RESOLVED_NO",
            "reward": 0,
            "status_map": {"test_a": "FAILED"},
            "report": {"FAIL_TO_PASS": {"success": [], "failure": ["test_a"]}},
        },
        None,
    )

    assert payload["status_map"] == {"test_a": "FAILED"}
    assert payload["report"] == {"FAIL_TO_PASS": {"success": [], "failure": ["test_a"]}}


def test_terminal_reward_payload_omits_test_details() -> None:
    from nanorollout.adapters.terminal.common import build_reward_payload

    payload = build_reward_payload(
        "terminal-1",
        {
            "resolved": False,
            "resolved_status": "unresolved",
            "reward": 0,
            "status_map": {"test_a": "FAILED"},
            "report": {"failed": 1},
        },
        None,
    )

    assert "status_map" not in payload
    assert "report" not in payload


def test_runner_specs_load_entrypoints() -> None:
    for spec in RUNNER_SPECS:
        assert callable(load_runner_callable(spec.module, spec.entrypoint))


def test_adapters_implement_runner_contract() -> None:
    assert issubclass(SweTaskAdapter, TaskAdapter)
    assert issubclass(TerminalTaskAdapter, TaskAdapter)
    assert issubclass(OSWorldTaskAdapter, TaskAdapter)
    assert issubclass(CocoaTaskAdapter, TaskAdapter)


def test_installed_agent_factory_lives_with_harness_agents() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    assert "claude-code" in AGENT_REGISTRY
    assert not (repo_root / "nanorollout/task_adapters").exists()
    assert not (repo_root / "nanorollout/adapters/installed_agents.py").exists()
    assert not (repo_root / "nanorollout/adapters/llm.py").exists()
    assert not (repo_root / "nanorollout/adapters/cocoa_bench.py").exists()
    assert not (repo_root / "nanorollout/adapters/osworld.py").exists()


def test_task_adapter_layout_is_uniform() -> None:
    root = Path(__file__).resolve().parents[2] / "nanorollout/adapters"

    for task_name in ("swe", "terminal", "osworld", "cocoa"):
        assert (root / task_name / "__init__.py").is_file()
        assert (root / task_name / "adapter.py").is_file()
        assert (root / task_name / "entrypoints.py").is_file()

    assert (root / "swe" / "task" / "__init__.py").is_file()
    assert (root / "swe" / "task" / "datasets.py").is_file()
    assert (root / "cocoa" / "task" / "__init__.py").is_file()
    assert (root / "cocoa" / "task" / "source.py").is_file()
    assert (root / "osworld" / "task" / "__init__.py").is_file()
    assert (root / "osworld" / "task" / "source.py").is_file()
    assert (root / "terminal" / "task" / "__init__.py").is_file()
    assert (root / "terminal" / "task" / "grading.py").is_file()

    assert not (root / "swe" / "datasets.py").exists()
    assert not (root / "swe" / "eval").exists()
    assert not (root / "terminal" / "eval").exists()
    assert not (root / "cocoa_bench.py").exists()
    assert not (root / "osworld.py").exists()
    assert not (root / "swe" / "config.py").exists()
    assert not (root / "swe" / "lifecycle.py").exists()
    assert not (root / "terminal" / "lifecycle.py").exists()
    assert not (root / "swe" / "agents.py").exists()
    assert not (root / "terminal" / "agents.py").exists()


def test_agent_binding_lives_in_entrypoints_not_adapters() -> None:
    root = Path(__file__).resolve().parents[2] / "nanorollout/adapters"
    forbidden = (
        "AGENT_REGISTRY",
        "_create_agent",
        "CocoaAgent",
        "Qwen3VLAgent",
        "MiniSweAgent",
        "TerminalMiniSweAgent",
        "Terminus2Agent",
        "CodeActAgent",
        "CodeActLiteAgent",
        "R2EGymAgent",
    )

    adapter_paths = (
        root / "swe" / "adapter.py",
        root / "terminal" / "adapter.py",
        root / "osworld" / "adapter.py",
        root / "cocoa" / "adapter.py",
    )
    for adapter_path in adapter_paths:
        source = adapter_path.read_text()
        for token in forbidden:
            assert token not in source


def test_runner_does_not_own_agent_or_llm_config() -> None:
    assert not hasattr(runner_module, "AgentConfig")
    assert not hasattr(runner_module, "build_agent_config")
    assert not hasattr(runner_module, "parse_json_object")


def test_llm_config_accepts_provider_without_scaffold_args() -> None:
    config = build_llm_config(
        model="model",
        base_url="http://localhost:8000",
        api_key="key",
        sampling_params={"llm_provider": "custom", "temperature": 0.2},
    )

    assert config.llm_provider == "custom"
    assert config.temperature == 0.2
    assert not hasattr(config, "max_iterations")


def test_task_spec_has_normalized_outer_shape() -> None:
    task = TaskSpec(
        id="typed-task",
        kind="fake",
        instruction="do it",
        payload=SimpleNamespace(task_id="typed-task"),
        environment={"image": "adapter-owned"},
        evaluation={"timeout": 30},
        metadata={"source": "unit"},
    )

    assert task.id == "typed-task"
    assert task.kind == "fake"
    assert task.payload.task_id == "typed-task"
    assert task.environment == {"image": "adapter-owned"}
    assert task.evaluation == {"timeout": 30}
    assert task.metadata == {"source": "unit"}
    assert not hasattr(task, "data")
    assert not hasattr(task, "state")
    assert not hasattr(runner_module, "PreparedTask")


def test_task_specs_parse_task_specific_args() -> None:
    swe_spec = SweTaskSpec.from_request(
        TaskRunRequest(
            instance_id="swe-1",
            output_dir="/tmp/out",
            model_name="model",
            base_url=None,
            api_key=None,
            env_type="docker",
            sampling_params=None,
            extra_args={
                "dataset": "verified",
                "split": "test",
                "step_timeout": "300",
                "eval_timeout": 600,
                "env_timeout": 120,
                "create_timeout": 60,
                "max_iterations": 50,
            },
        )
    )
    terminal_spec = TerminalTaskSpec.from_request(
        TaskRunRequest(
            instance_id="terminal-1",
            output_dir="/tmp/out",
            model_name="model",
            base_url=None,
            api_key=None,
            env_type="docker",
            sampling_params=None,
            extra_args={
                "env_timeout": 120,
                "create_timeout": 60,
                "max_iterations": "40",
                "repo_url": "https://example.test/repo.git",
                "refresh_repo": "false",
                "timeout_multiplier": "1.5",
            },
        )
    )

    assert swe_spec.dataset == "verified"
    assert swe_spec.step_timeout == 300
    assert terminal_spec.max_iterations == 40
    assert terminal_spec.repo_url == "https://example.test/repo.git"
    assert terminal_spec.refresh_repo is False
    assert terminal_spec.timeout_args() == {"timeout_multiplier": 1.5}


def test_run_task_owns_the_common_lifecycle(tmp_path: Path) -> None:
    class FakeAdapter(TaskAdapter):
        runner_label = "fake"

        def __init__(self) -> None:
            self.calls: list[str] = []

        def prepare_task(
            self,
            request: TaskRunRequest,
            trial_dir: Path,
        ) -> TaskSpec:
            self.calls.append("prepare")
            return TaskSpec(
                id=request.instance_id,
                kind="fake",
                instruction="do it",
                payload={"id": request.instance_id},
            )

        def create_environment(
            self,
            task: TaskSpec,
            request: TaskRunRequest,
        ) -> Any:
            self.calls.append("create_env")
            return SimpleNamespace()

        def start_environment(
            self,
            env_obj: Any,
            task: TaskSpec,
            request: TaskRunRequest,
        ) -> None:
            self.calls.append("start_env")

        def setup_environment(
            self,
            env_obj: Any,
            task: TaskSpec,
            request: TaskRunRequest,
        ) -> None:
            self.calls.append("setup_env")

        def build_agent(
            self,
            env_obj: Any,
            task: TaskSpec,
            request: TaskRunRequest,
            trial_dir: Path,
        ) -> Any:
            self.calls.append("build_agent")
            return SimpleNamespace()

        def run_agent(
            self,
            agent: Any,
            task: TaskSpec,
            env_obj: Any,
        ) -> Any:
            self.calls.append("run_agent")
            return SimpleNamespace(
                history=[{"role": "assistant", "content": "done"}],
                success=True,
                message="ok",
                iterations=1,
                error=None,
                exit_status="done",
            )

        def after_agent_result(self, agent_result: Any) -> None:
            self.calls.append("after_agent")

        def evaluate(
            self,
            env_obj: Any,
            task: TaskSpec,
            request: TaskRunRequest,
            trial_dir: Path,
        ) -> tuple[dict[str, Any], Optional[str]]:
            self.calls.append("eval")
            return {"resolved": True, "resolved_status": "FULL", "reward": 1}, None

        def build_reward_payload(
            self,
            instance_id: str,
            eval_payload: dict[str, Any],
            error_msg: Optional[str],
        ) -> dict[str, Any]:
            self.calls.append("reward")
            return dict(eval_payload)

        def write_result(
            self,
            trial_dir: Path,
            instance_id: str,
            model: str,
            base_url: Optional[str],
            env_type: str,
            agent_result: Any,
            tools_json: Optional[dict[str, Any]],
            reward_payload: dict[str, Any],
            eval_output: Optional[str],
            started: float,
            metadata: dict[str, Any],
        ) -> None:
            self.calls.append("write")

        def build_exit_status(
            self,
            error_msg: Optional[str],
            agent_result: Any,
            eval_payload: dict[str, Any],
        ) -> str:
            self.calls.append("exit")
            return "done"

    adapter = FakeAdapter()
    result = run_task(
        TaskRunRequest(
            instance_id="fake-1",
            output_dir=str(tmp_path),
            model_name="model",
            base_url=None,
            api_key=None,
            env_type="fake",
            sampling_params=None,
            extra_args={},
        ),
        adapter,
    )

    assert result["reward"] == 1
    assert result["exit_status"] == "done"
    assert adapter.calls == [
        "prepare",
        "create_env",
        "start_env",
        "setup_env",
        "build_agent",
        "run_agent",
        "after_agent",
        "eval",
        "reward",
        "write",
        "exit",
    ]


def test_run_task_writes_fallback_when_adapter_write_fails(tmp_path: Path) -> None:
    class WriteFailAdapter(TaskAdapter):
        runner_label = "write-fail"

        def prepare_task(
            self,
            request: TaskRunRequest,
            trial_dir: Path,
        ) -> TaskSpec:
            return TaskSpec(
                id=request.instance_id,
                kind="fake",
                instruction="do it",
                payload={"id": request.instance_id},
            )

        def create_environment(
            self,
            task: TaskSpec,
            request: TaskRunRequest,
        ) -> Any:
            return SimpleNamespace()

        def start_environment(
            self,
            env_obj: Any,
            task: TaskSpec,
            request: TaskRunRequest,
        ) -> None:
            return None

        def build_agent(
            self,
            env_obj: Any,
            task: TaskSpec,
            request: TaskRunRequest,
            trial_dir: Path,
        ) -> Any:
            return SimpleNamespace()

        def run_agent(
            self,
            agent: Any,
            task: TaskSpec,
            env_obj: Any,
        ) -> Any:
            return SimpleNamespace(
                history=[],
                success=True,
                message="ok",
                iterations=1,
                error=None,
                exit_status="done",
            )

        def evaluate(
            self,
            env_obj: Any,
            task: TaskSpec,
            request: TaskRunRequest,
            trial_dir: Path,
        ) -> tuple[dict[str, Any], Optional[str]]:
            return {"resolved": True, "resolved_status": "FULL", "reward": 1}, None

        def build_reward_payload(
            self,
            instance_id: str,
            eval_payload: dict[str, Any],
            error_msg: Optional[str],
        ) -> dict[str, Any]:
            payload = dict(eval_payload)
            payload["instance_id"] = instance_id
            payload["error"] = error_msg
            return payload

        def write_result(
            self,
            trial_dir: Path,
            instance_id: str,
            model: str,
            base_url: Optional[str],
            env_type: str,
            agent_result: Any,
            tools_json: Optional[dict[str, Any]],
            reward_payload: dict[str, Any],
            eval_output: Optional[str],
            started: float,
            metadata: dict[str, Any],
        ) -> None:
            raise RuntimeError("disk full")

        def build_exit_status(
            self,
            error_msg: Optional[str],
            agent_result: Any,
            eval_payload: dict[str, Any],
        ) -> str:
            return "error" if error_msg else "done"

    result = run_task(
        TaskRunRequest(
            instance_id="fake-1",
            output_dir=str(tmp_path),
            model_name="model",
            base_url=None,
            api_key=None,
            env_type="fake",
            sampling_params=None,
            extra_args={},
        ),
        WriteFailAdapter(),
    )

    assert result["exit_status"] == "error"
    assert "result write failed" in result["metadata"]["error"]
    assert (tmp_path / "reward.json").is_file()
    assert (tmp_path / "metadata.json").is_file()


def test_swe_dataset_variants_are_explicit() -> None:
    assert isinstance(resolve_swe_dataset_adapter("r2e-gym"), R2EGymDatasetAdapter)
    assert isinstance(resolve_swe_dataset_adapter("pro"), SweBenchProDatasetAdapter)
    rebench_adapter = resolve_swe_dataset_adapter("rebench")
    assert isinstance(rebench_adapter, SweRebenchDatasetAdapter)
    assert rebench_adapter.dataset_revision == SWE_REBENCH_REVISION
    assert isinstance(resolve_swe_dataset_adapter("smith-py"), SweSmithDatasetAdapter)


def test_swebench_pro_adapter_uses_official_dataset_and_image() -> None:
    adapter = resolve_swe_dataset_adapter("swebench-pro")
    request = TaskRunRequest(
        instance_id="instance-1",
        output_dir="/tmp/out",
        model_name="model",
        base_url=None,
        api_key=None,
        env_type="docker",
        sampling_params=None,
        extra_args={},
    )

    assert adapter.source_name == "ScaleAI/SWE-bench_Pro"
    assert adapter.workspace_dir() == "/app"
    assert (
        adapter.image_name(
            {"instance_id": "instance-1", "dockerhub_tag": "repo-tag"},
            "docker",
            request,
        )
        == "docker.io/jefzda/sweap-images:repo-tag"
    )


def test_swebench_pro_adapter_normalizes_prompt_fields() -> None:
    adapter = resolve_swe_dataset_adapter("pro")

    row = adapter.normalize_row(
        {
            "problem_statement": '"Fix the bug\\nnow"',
            "requirements": '"Keep the public API stable"',
            "interface": "",
            "fail_to_pass": '["tests/test_bug.py::test_fix"]',
            "pass_to_pass": "['tests/test_existing.py::test_old']",
        }
    )

    assert row["problem_statement"] == (
        "Fix the bug\nnow\n\nRequirements:\nKeep the public API stable"
    )
    assert row["FAIL_TO_PASS"] == ["tests/test_bug.py::test_fix"]
    assert row["PASS_TO_PASS"] == ["tests/test_existing.py::test_old"]


def test_swe_and_terminal_specs_use_consolidated_modules() -> None:
    assert (
        resolve_runner("swe", "oh-core").module
        == "nanorollout.adapters.swe.entrypoints"
    )
    assert resolve_runner("swe", "claudecode").entrypoint == "run_installed_claude_code"
    assert (
        resolve_runner("terminal", "terminus-2").module
        == "nanorollout.adapters.terminal.entrypoints"
    )
    assert resolve_runner("terminal", "open-code").entrypoint == "run_tb_opencode"
    assert (
        resolve_runner("osworld", "qwen3vl-mmagents").module
        == "nanorollout.adapters.osworld.entrypoints"
    )
    assert (
        resolve_runner("cocoa-bench", "cocoa-agent").module
        == "nanorollout.adapters.cocoa.entrypoints"
    )


def test_legacy_runner_resolution_stays_swe_first() -> None:
    assert resolve_legacy_runner("miniswe").task == "swe"
    assert resolve_legacy_runner("terminal-miniswe").task == "terminal"


def test_build_runner_params_preserves_request_fields() -> None:
    request = RunRequest(
        instance_id="task-1",
        model_name="model",
        base_url="http://localhost:8000",
        api_key="key",
        env_type="docker",
        task="terminal",
        agent="terminus-2",
        sampling_params={"temperature": 0.1},
        extra_args={"env_timeout": 120},
    )

    params = build_runner_params(request, output_dir="/tmp/run")

    assert params["instance_id"] == "task-1"
    assert params["output_dir"] == "/tmp/run"
    assert params["base_url"] == "http://localhost:8000"
    assert params["api_key"] == "key"
    assert params["env_type"] == "docker"
    assert params["extra_args"] == {"env_timeout": 120}
