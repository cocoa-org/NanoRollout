def grade(**kwargs) -> dict:
    """
    Grade the image classification task.

    Returns:
        Dict mapping criterion names to scores (0.0 to 1.0)
    """
    import json
    from collections import Counter
    from functools import lru_cache
    from pathlib import Path

    output_dir = Path("/tmp_workspace/results")
    gt_file = Path("/tmp_workspace") / "gt" / "filename_to_class.json"

    ALL_CRITERIA = {
        "output_dir_exists": 0.0,
        "five_subdirs": 0.0,
        "all_expected_files_present": 0.0,
        "no_duplicate_or_extra_files": 0.0,
        "folder_purity": 0.0,
        "class_completeness": 0.0,
        "best_match_accuracy": 0.0,
        "overall_score": 0.0,
    }

    if not gt_file.exists():
        return ALL_CRITERIA

    gt_map = json.loads(gt_file.read_text(encoding="utf-8"))
    expected_files = set(gt_map.keys())
    classes = sorted(set(gt_map.values()))
    total_expected = len(expected_files)

    if total_expected == 0:
        return ALL_CRITERIA

    scores = dict(ALL_CRITERIA)

    if not output_dir.exists() or not output_dir.is_dir():
        return scores

    scores["output_dir_exists"] = 1.0

    pred_dirs = sorted([p for p in output_dir.iterdir() if p.is_dir()])
    scores["five_subdirs"] = 1.0 if len(pred_dirs) == 5 else 0.0

    pred_file_counts = Counter()
    extra_files = []
    pred_sets = []

    for folder in pred_dirs:
        folder_files = []
        for item in folder.iterdir():
            if item.is_file():
                folder_files.append(item.name)
            else:
                extra_files.append(str(item))
        pred_sets.append(folder_files)
        pred_file_counts.update(folder_files)

    predicted_files = set(pred_file_counts.keys())
    missing_files = expected_files - predicted_files
    extra_named_files = predicted_files - expected_files
    has_duplicates = any(v != 1 for v in pred_file_counts.values())

    scores["all_expected_files_present"] = 1.0 if not missing_files else 0.0
    scores["no_duplicate_or_extra_files"] = 1.0 if (not extra_named_files and not extra_files and not has_duplicates) else 0.0

    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    matrix = []
    valid_total = 0

    for folder_files in pred_sets:
        row = [0] * len(classes)
        for filename in folder_files:
            cls = gt_map.get(filename)
            if cls is None:
                continue
            row[class_to_idx[cls]] += 1
            valid_total += 1
        matrix.append(row)

    if valid_total > 0 and matrix:
        scores["folder_purity"] = round(
            sum(max(row) for row in matrix) / valid_total,
            4,
        )

        completeness_hits = 0
        for class_idx in range(len(classes)):
            completeness_hits += max(row[class_idx] for row in matrix)
        scores["class_completeness"] = round(completeness_hits / total_expected, 4)

        @lru_cache(maxsize=None)
        def best_match(i: int, used_mask: int) -> int:
            if i == len(matrix):
                return 0

            best = best_match(i + 1, used_mask)
            for class_idx in range(len(classes)):
                if used_mask & (1 << class_idx):
                    continue
                best = max(
                    best,
                    matrix[i][class_idx] + best_match(i + 1, used_mask | (1 << class_idx)),
                )
            return best

        matched = best_match(0, 0)
        scores["best_match_accuracy"] = round(matched / total_expected, 4)

    scores["overall_score"] = round(
        0.06 * scores["five_subdirs"]
        + 0.08 * scores["all_expected_files_present"]
        + 0.06 * scores["no_duplicate_or_extra_files"]
        + 0.8 * scores["best_match_accuracy"],
        4,
    )

    return scores
