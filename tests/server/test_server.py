from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

import requests


SERVER_URL = os.environ.get("NANOROLLOUT_SERVER_URL", "http://127.0.0.1:11000").rstrip("/")
MODEL_NAME = os.environ.get("MODEL_NAME", "test-model")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def _build_run_request(
    *,
    instance_id: str,
    dataset: str,
    split: str,
    run_name: str,
    agent: str = "oh-core",
) -> dict[str, Any]:
    return {
        "instance_id": instance_id,
        "task_timeout_s": 1800,
        "model_name": MODEL_NAME,
        "run_name": run_name,
        "base_url": OPENAI_BASE_URL,
        "api_key": OPENAI_API_KEY,
        "env_type": os.environ.get("ENV_TYPE", "modal"),
        "sampling_params": {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 4096,
        },
        "task": "swe",
        "agent": agent,
        "extra_args": {
            "instance_id": instance_id,
            "dataset": dataset,
            "split": split,
            "step_timeout": 600,
            "eval_timeout": 600,
            "env_timeout": 120,
            "create_timeout": 600,
            "max_iterations": 100,
            "use_fn_calling": True,
        },
    }


SWEBENCH_REQUEST = _build_run_request(
    instance_id="astropy__astropy-12907",
    dataset="verified",
    split="test",
    run_name="test-swebench-step_0",
)

SWE_GYM_REQUEST = _build_run_request(
    instance_id="getmoto__moto-7365",
    dataset="gym",
    split="train",
    run_name="test-swe-gym-step_0",
)


EXPECTED_RESPONSE_KEYS = {
    "reward",
    "messages",
    "exit_status",
    "agent_metrics",
    "metadata",
    "tools",
}


def _post_run(payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{SERVER_URL}/run", json=payload, timeout=None)
    response.raise_for_status()
    return response.json()


def _assert_miles_compatible_response(result: dict[str, Any]) -> None:
    assert set(result) == EXPECTED_RESPONSE_KEYS
    assert isinstance(result["reward"], (int, float))
    assert isinstance(result["messages"], list)
    assert result["exit_status"]
    assert isinstance(result["agent_metrics"], dict)
    assert isinstance(result["metadata"], dict)
    assert result["tools"] is None or isinstance(result["tools"], list)


def test_swebench_request() -> None:
    result = _post_run(deepcopy(SWEBENCH_REQUEST))

    _assert_miles_compatible_response(result)


def test_swe_gym_request() -> None:
    result = _post_run(deepcopy(SWE_GYM_REQUEST))

    _assert_miles_compatible_response(result)


if __name__ == "__main__":
    for payload in (SWEBENCH_REQUEST, SWE_GYM_REQUEST):
        result = _post_run(deepcopy(payload))
        print(result)
