"""Compatibility wrapper for SWE runner single-run helpers."""

from brew.harness.runner.swe.single_run import (
    DATASET_MAPPING,
    _normalize_r2e_gym_instance,
    build_eval_payload,
    build_reward_payload,
    create_environment,
    load_instances,
    resolve_instance,
    select_instance,
)

__all__ = [
    "DATASET_MAPPING",
    "_normalize_r2e_gym_instance",
    "build_eval_payload",
    "build_reward_payload",
    "create_environment",
    "load_instances",
    "resolve_instance",
    "select_instance",
]
