"""SWE dataset variants."""

from __future__ import annotations

import json
import logging
import os
import shlex
from typing import Any, Optional

from datasets import load_dataset

from nanorollout.runner import (
    TaskRunRequest,
    TaskSpec,
)

from .r2egym import run_r2egym_eval, setup_r2egym_env
from .rebench import run_rebench_eval
from .pro import (
    decode_literal_string,
    parse_list,
    resolve_scripts_dir,
    run_swebench_pro_eval,
)
from .swebench import run_swebench_eval

logger = logging.getLogger(__name__)

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "pro": "ScaleAI/SWE-bench_Pro",
    "swebench-pro": "ScaleAI/SWE-bench_Pro",
    "swe-bench-pro": "ScaleAI/SWE-bench_Pro",
    "gym": "SWE-Gym/SWE-Gym",
    "rebench": "nebius/SWE-rebench",
    "smith": "SWE-bench/SWE-smith",
    "smith-py": "SWE-bench/SWE-smith-py",
    "smith-go": "SWE-bench/SWE-smith-go",
    "smith-rs": "SWE-bench/SWE-smith-rs",
    "smith-cpp": "SWE-bench/SWE-smith-cpp",
    "smith-java": "SWE-bench/SWE-smith-java",
    "smith-js": "SWE-bench/SWE-smith-js",
    "smith-ts": "SWE-bench/SWE-smith-ts",
    "smith-php": "SWE-bench/SWE-smith-php",
    "r2e-gym": "R2E-Gym/R2E-Gym-V1",
}

SWE_REBENCH_REVISION = "89cdfbab4ab1bd8f5a658bb212d1b63624f4f881"


def _select_instance(
    instances: list[dict[str, Any]],
    instance_id: str,
) -> dict[str, Any]:
    for instance in instances:
        if instance.get("instance_id") == instance_id:
            return dict(instance)
    raise ValueError(f"Instance id not found: {instance_id}")


class SweDatasetAdapter:
    dataset_name: Optional[str] = None
    dataset_revision: Optional[str] = None
    namespace = "swebench"
    image_dataset = "sweb"
    image_split = "eval"
    id_separator = "_1776_"

    def __init__(self, subset: str) -> None:
        self.subset = subset

    @property
    def source_name(self) -> str:
        return self.dataset_name or DATASET_MAPPING.get(self.subset, self.subset)

    def normalize_row(self, row: dict[str, Any]) -> dict[str, Any]:
        return row

    def load_instances(self, split: str) -> list[dict[str, Any]]:
        logger.info(
            "Loading dataset %s split %s revision %s",
            self.source_name,
            split,
            self.dataset_revision or "default",
        )
        load_kwargs: dict[str, Any] = {}
        if self.dataset_revision:
            load_kwargs["revision"] = self.dataset_revision
        return [
            self.normalize_row(dict(row))
            for row in load_dataset(self.source_name, split=split, **load_kwargs)
        ]

    def resolve_instance(self, instance_id: str, split: str) -> dict[str, Any]:
        instance = _select_instance(self.load_instances(split), instance_id)
        if instance.get("instance_id") != instance_id:
            raise ValueError(
                f"Instance id mismatch: {instance.get('instance_id')} != {instance_id}"
            )
        return instance

    def image_id(self, instance: dict[str, Any]) -> str:
        return instance["instance_id"].replace("__", self.id_separator)

    def workspace_dir(self) -> str:
        return "/testbed"

    def image_name(
        self,
        instance: dict[str, Any],
        env_type: str,
        request: Optional[TaskRunRequest] = None,
    ) -> str:
        del request
        split = f".{self.image_split}" if self.image_split else ""
        image_id = self.image_id(instance)
        if env_type in ("docker", "modal"):
            return (
                f"docker.io/{self.namespace}/{self.image_dataset}{split}.x86_64."
                f"{image_id}:latest"
            ).lower()
        if env_type == "enroot":
            return (
                f"{self.namespace}+{self.image_dataset}{split}.x86_64."
                f"{image_id}+latest.sqsh"
            ).lower()
        raise ValueError(f"Unknown environment class: {env_type}")

    def setup_environment(
        self,
        env_obj: Any,
        task: TaskSpec,
        request: TaskRunRequest,
    ) -> None:
        del env_obj, task, request

    def evaluate(
        self,
        env_obj: Any,
        task: TaskSpec,
        request: TaskRunRequest,
    ) -> tuple[dict[str, Any], Optional[str]]:
        del request
        return run_swebench_eval(
            env_obj,
            task.payload,
            task.evaluation.get("eval_timeout"),
            task.environment.get("workspace_dir"),
        )


class SweGymDatasetAdapter(SweDatasetAdapter):
    namespace = "xingyaoww"
    id_separator = "_s_"


class SweRebenchDatasetAdapter(SweDatasetAdapter):
    dataset_revision = SWE_REBENCH_REVISION
    namespace = "swerebench"

    def image_name(
        self,
        instance: dict[str, Any],
        env_type: str,
        request: Optional[TaskRunRequest] = None,
    ) -> str:
        del request
        if env_type in ("docker", "modal"):
            image = instance.get("image_name") or instance.get("docker_image")
            if image:
                return str(image)
        return super().image_name(instance, env_type)

    def evaluate(
        self,
        env_obj: Any,
        task: TaskSpec,
        request: TaskRunRequest,
    ) -> tuple[dict[str, Any], Optional[str]]:
        del request
        return run_rebench_eval(
            env_obj,
            task.payload,
            task.evaluation.get("eval_timeout"),
            task.environment.get("workspace_dir"),
        )


class SweBenchProDatasetAdapter(SweDatasetAdapter):
    dataset_name = "ScaleAI/SWE-bench_Pro"

    def normalize_row(self, row: dict[str, Any]) -> dict[str, Any]:
        problem = str(decode_literal_string(row.get("problem_statement", "")))
        requirements = decode_literal_string(row.get("requirements"))
        interface = decode_literal_string(row.get("interface"))
        parts = [problem]
        if requirements:
            parts.append(f"Requirements:\n{requirements}")
        if interface:
            parts.append(f"Interface:\n{interface}")

        row["problem_statement"] = "\n\n".join(parts)
        row["FAIL_TO_PASS"] = parse_list(row.get("fail_to_pass"))
        row["PASS_TO_PASS"] = parse_list(row.get("pass_to_pass"))
        return row

    def workspace_dir(self) -> str:
        return "/app"

    def image_name(
        self,
        instance: dict[str, Any],
        env_type: str,
        request: Optional[TaskRunRequest] = None,
    ) -> str:
        if env_type not in ("docker", "modal"):
            raise ValueError("SWE-Bench Pro uses DockerHub images; use docker or modal")

        extra_args = request.extra_args if request is not None else {}
        username = (
            extra_args.get("swebench_pro_dockerhub_username")
            or extra_args.get("dockerhub_username")
            or os.environ.get("SWEBENCH_PRO_DOCKERHUB_USERNAME")
            or "jefzda"
        )
        tag = instance.get("dockerhub_tag")
        if not tag:
            raise ValueError("SWE-Bench Pro instance is missing dockerhub_tag")
        return f"docker.io/{username}/sweap-images:{tag}"

    def setup_environment(
        self,
        env_obj: Any,
        task: TaskSpec,
        request: TaskRunRequest,
    ) -> None:
        del request
        workspace_dir = shlex.quote(task.environment["workspace_dir"])
        base_commit = shlex.quote(str(task.metadata["base_commit"]))
        env_obj.execute(
            f"cd {workspace_dir} && git reset --hard {base_commit} "
            f"&& git clean -fd && git checkout {base_commit}"
        )

    def evaluate(
        self,
        env_obj: Any,
        task: TaskSpec,
        request: TaskRunRequest,
    ) -> tuple[dict[str, Any], Optional[str]]:
        scripts_dir = resolve_scripts_dir(request.extra_args)
        if scripts_dir is None:
            raise ValueError(
                "SWE-Bench Pro eval needs official run scripts. Set "
                "swebench_pro_scripts_dir or swebench_pro_repo."
            )
        return run_swebench_pro_eval(
            env_obj,
            task.payload,
            task.evaluation.get("eval_timeout"),
            task.environment.get("workspace_dir"),
            scripts_dir,
        )


class SweSmithDatasetAdapter(SweDatasetAdapter):
    namespace = "jyangballin"
    image_dataset = "swesmith"
    image_split = ""

    def image_id(self, instance: dict[str, Any]) -> str:
        instance_id = instance["instance_id"]
        return f"{instance_id.split('.')[0]}.{instance_id.split('.')[1]}".replace(
            "__",
            self.id_separator,
        )

    def image_name(
        self,
        instance: dict[str, Any],
        env_type: str,
        request: Optional[TaskRunRequest] = None,
    ) -> str:
        del request
        if env_type in ("docker", "modal"):
            image = instance.get("image_name") or instance.get("docker_image")
            if image:
                return str(image)
        return super().image_name(instance, env_type)

    def setup_environment(
        self,
        env_obj: Any,
        task: TaskSpec,
        request: TaskRunRequest,
    ) -> None:
        workspace_dir = shlex.quote(task.environment["workspace_dir"])
        instance_id = shlex.quote(request.instance_id)
        env_obj.execute(
            f"cd {workspace_dir} && "
            f"(git checkout {instance_id} || "
            f"(git fetch --all --prune && git checkout {instance_id}))"
        )


class R2EGymDatasetAdapter(SweDatasetAdapter):
    dataset_name = "R2E-Gym/R2E-Gym-V1"
    namespace = "namanjain12"

    def normalize_row(self, row: dict[str, Any]) -> dict[str, Any]:
        if "instance_id" not in row:
            repo = row.get("repo_name", "unknown")
            commit = row.get("commit_hash", "unknown")
            row["instance_id"] = f"{repo}__{commit}"

        parsed_commit_content = json.loads(row.get("parsed_commit_content", "{}"))
        old_commit_hash = parsed_commit_content.get("old_commit_hash")
        if not old_commit_hash:
            raise ValueError("R2E-Gym: old_commit_hash not found")

        row["base_commit"] = old_commit_hash
        row["patch"] = row.get("parsed_commit_content", "")
        return row

    def image_name(
        self,
        instance: dict[str, Any],
        env_type: str,
        request: Optional[TaskRunRequest] = None,
    ) -> str:
        del request
        repo_name = instance.get("repo_name", "")
        commit_hash = instance.get("commit_hash", "")
        if env_type == "enroot":
            return f"{self.namespace}+{repo_name}_final+{commit_hash}.sqsh".lower()
        if env_type in ("docker", "modal"):
            return f"docker.io/{self.namespace}/{repo_name}_final:{commit_hash}".lower()
        raise ValueError(f"Unknown environment class: {env_type}")

    def setup_environment(
        self,
        env_obj: Any,
        task: TaskSpec,
        request: TaskRunRequest,
    ) -> None:
        del request
        setup_r2egym_env(env_obj, workspace_dir=task.environment["workspace_dir"])

    def evaluate(
        self,
        env_obj: Any,
        task: TaskSpec,
        request: TaskRunRequest,
    ) -> tuple[dict[str, Any], Optional[str]]:
        del request
        return run_r2egym_eval(
            env_obj,
            task.payload,
            task.evaluation.get("eval_timeout"),
        )


def resolve_swe_dataset_adapter(subset: str) -> SweDatasetAdapter:
    subset_lower = subset.lower()
    if "r2e" in subset_lower:
        return R2EGymDatasetAdapter(subset)
    if "pro" in subset_lower:
        return SweBenchProDatasetAdapter(subset)
    if "gym" in subset_lower:
        return SweGymDatasetAdapter(subset)
    if "rebench" in subset_lower:
        return SweRebenchDatasetAdapter(subset)
    if "smith" in subset_lower:
        return SweSmithDatasetAdapter(subset)
    return SweDatasetAdapter(subset)
