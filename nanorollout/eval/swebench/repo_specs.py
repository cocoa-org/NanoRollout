from .constants import MAP_REPO_VERSION_TO_SPECS


def _normalize_repo(repo: str | None) -> str | None:
    return repo.lower() if repo else None


def _get_repo_versions(repo: str | None) -> dict[str, dict[str, object]] | None:
    repo_key = _normalize_repo(repo)
    if not repo_key:
        return None

    versions = MAP_REPO_VERSION_TO_SPECS.get(repo_key)
    if versions:
        return versions

    if repo:
        versions = MAP_REPO_VERSION_TO_SPECS.get(repo)
        if versions:
            return versions

    for candidate_repo, candidate_versions in MAP_REPO_VERSION_TO_SPECS.items():
        if candidate_repo.lower() == repo_key:
            return candidate_versions

    return None


def _collect_version_candidates(
    version: str | None,
    instance: dict | None = None,
) -> list[str]:
    candidates: list[str] = []
    if version not in (None, ""):
        candidates.append(str(version))

    if isinstance(instance, dict):
        for field in ("base_commit", "commit"):
            commit = instance.get(field)
            if not commit:
                continue
            commit_str = str(commit)
            candidates.extend([commit_str, commit_str[:8]])

    # Default profile for repos without explicit version catalog.
    candidates.append("default")
    return candidates


def get_repo_specs(
    repo: str | None,
    version: str | None,
    instance: dict | None = None,
) -> dict[str, object] | None:
    # SWE-rebench direct override.
    install_config = instance.get("install_config") if isinstance(instance, dict) else None
    if isinstance(install_config, dict):
        return install_config

    versions = _get_repo_versions(repo)
    if not versions:
        return None

    for candidate in _collect_version_candidates(version, instance):
        specs = versions.get(candidate)
        if specs:
            return specs
        candidate_lower = candidate.lower()
        specs = versions.get(candidate_lower)
        if specs:
            return specs

    return None


def get_repo_test_cmd(
    repo: str | None,
    version: str | None,
    instance: dict | None,
) -> str:
    specs = get_repo_specs(repo, version, instance)
    if not specs:
        return ""
    test_cmd = specs.get("test_cmd", "")
    if isinstance(test_cmd, list):
        return " && ".join(str(item) for item in test_cmd)
    return str(test_cmd)


def get_repo_eval_commands(
    repo: str | None,
    version: str | None,
    instance: dict | None,
) -> list[str]:
    specs = get_repo_specs(repo, version, instance)
    if not specs:
        return []
    eval_commands = specs.get("eval_commands", [])
    return [str(item) for item in eval_commands]
