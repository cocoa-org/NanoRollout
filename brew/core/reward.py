from typing import Any, Dict, Optional


def _coerce_reward(value: Any, threshold: Optional[float]) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        reward = 1.0 if value else 0.0
    else:
        try:
            reward = float(value)
        except (TypeError, ValueError):
            return 0.0
    if threshold is None:
        return reward
    return 1.0 if reward >= threshold else 0.0


def resolve_reward(
    result: Dict[str, Any],
    task_type: str | None = None,
) -> float:

    if task_type in ("swe"):
        return _coerce_reward(result.get("reward"), 0.0)

    raise ValueError(f"Unknown task type: {task_type}")
