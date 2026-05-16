def grade(**kwargs) -> dict:
    """
    Grade the OpenMMLab contributor summarization task.

    Returns:
        Dict mapping criterion names to scores (0.0 to 1.0)
    """
    import json
    import re
    from pathlib import Path

    expected_contributors_file = Path("/tmp_workspace/gt/expected_contributors.jsonl")
    expected_repos_file = Path("/tmp_workspace/gt/expected_repos.txt")
    output_contributors_file = Path("/tmp_workspace/results/contributors.jsonl")
    output_repos_file = Path("/tmp_workspace/results/repos.txt")

    ALL_CRITERIA = {
        "repo_exact_match": 0.0,
        "valid_jsonl": 0.0,
        "schema_valid": 0.0,
        "user_project_match_rate": 0.0,
        "user_full_match_rate": 0.0,
        "exact_match": 0.0,
        "overall_score": 0.0,
    }

    if not expected_contributors_file.exists():
        return ALL_CRITERIA
    if not expected_repos_file.exists():
        return ALL_CRITERIA

    def canon_project(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower())

    def parse_expected(path: Path, project_alias: dict[str, str]) -> dict[str, dict[str, int]]:
        data = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            user = item["user"]
            projects = item["project"]
            normalized = {}
            for project, count in projects.items():
                normalized[project_alias[canon_project(project)]] = int(count)
            data[user] = normalized
        return data

    expected_repo_names = [
        line.strip()
        for line in expected_repos_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    project_alias = {canon_project(name): name for name in expected_repo_names}
    expected_repo_set = set(project_alias.values())
    expected_counts = parse_expected(expected_contributors_file, project_alias)
    expected_projects = {user: set(projects) for user, projects in expected_counts.items()}
    expected_users = set(expected_counts)

    def compute_overall(s):
        s["overall_score"] = round(
            0.05 * s["repo_exact_match"]
            + 0.05 * s["valid_jsonl"]
            + 0.05 * s["schema_valid"]
            + 0.05 * s["user_project_match_rate"]
            + 0.70 * s["user_full_match_rate"]
            + 0.10 * s["exact_match"],
            4,
        )
        return s

    scores = dict(ALL_CRITERIA)
    if not output_repos_file.exists():
        return compute_overall(scores)

    raw_repo_lines = output_repos_file.read_text(encoding="utf-8").splitlines()
    repo_lines = [line.strip() for line in raw_repo_lines if line.strip()]
    repo_duplicates = set()
    repo_unknown = set()
    predicted_repo_names = []
    seen_repo_names = set()

    for repo in repo_lines:
        key = canon_project(repo)
        if key not in project_alias:
            repo_unknown.add(repo)
            continue
        normalized_repo = project_alias[key]
        predicted_repo_names.append(normalized_repo)
        if normalized_repo in seen_repo_names:
            repo_duplicates.add(normalized_repo)
        seen_repo_names.add(normalized_repo)

    predicted_repo_set = set(predicted_repo_names)
    scores["repo_exact_match"] = 1.0 if predicted_repo_set == expected_repo_set and not repo_unknown and not repo_duplicates else 0.0

    if not output_contributors_file.exists():
        return compute_overall(scores)

    raw_lines = output_contributors_file.read_text(encoding="utf-8").splitlines()
    nonempty_lines = [line for line in raw_lines if line.strip()]

    parsed = []
    for idx, line in enumerate(nonempty_lines, start=1):
        try:
            parsed.append(json.loads(line))
        except Exception:
            return compute_overall(scores)
    scores["valid_jsonl"] = 1.0

    predicted_counts = {}
    schema_ok = True
    notes = []

    for idx, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            schema_ok = False
            notes.append(f"line_{idx}_not_object")
            continue

        user = item.get("user")
        projects = item.get("project")

        if not isinstance(user, str) or not user.strip():
            schema_ok = False
            notes.append(f"line_{idx}_bad_user")
            continue
        user = user.strip()

        if not isinstance(projects, dict):
            schema_ok = False
            notes.append(f"line_{idx}_bad_project")
            continue

        if not all(
            isinstance(k, str)
            and k.strip()
            and isinstance(v, int)
            and not isinstance(v, bool)
            for k, v in projects.items()
        ):
            schema_ok = False
            notes.append(f"line_{idx}_bad_project_entry")
            continue

        norm_count_map = {}
        for project, count in projects.items():
            key = canon_project(project.strip())
            if key not in project_alias:
                schema_ok = False
                notes.append(f"line_{idx}_unknown_project={project.strip()}")
                continue
            normalized_project = project_alias[key]
            norm_count_map[normalized_project] = count
            if normalized_project not in predicted_repo_set:
                schema_ok = False
                notes.append(f"line_{idx}_project_not_in_repo_output={normalized_project}")

        if user in predicted_counts:
            schema_ok = False
            notes.append(f"duplicate_user={user}")
            continue
        predicted_counts[user] = norm_count_map

    scores["schema_valid"] = 1.0 if schema_ok else 0.0
    predicted_projects = {user: set(projects) for user, projects in predicted_counts.items()}
    all_users = expected_users | set(predicted_counts)

    if all_users:
        project_match_users = sum(
            1 for user in all_users
            if predicted_projects.get(user, set()) == expected_projects.get(user, set())
        )
        full_match_users = sum(
            1 for user in all_users
            if predicted_counts.get(user, {}) == expected_counts.get(user, {})
        )
        scores["user_project_match_rate"] = round(project_match_users / len(all_users), 4)
        scores["user_full_match_rate"] = round(full_match_users / len(all_users), 4)

    full_exact = (
        scores["repo_exact_match"] == 1.0
        and schema_ok
        and predicted_counts == expected_counts
    )
    scores["exact_match"] = 1.0 if full_exact else 0.0

    return compute_overall(scores)
