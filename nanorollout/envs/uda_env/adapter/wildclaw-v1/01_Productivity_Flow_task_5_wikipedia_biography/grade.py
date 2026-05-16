def grade(**kwargs) -> dict:
    """
    Grade the Wikipedia Biography extraction task.
    """
    from pathlib import Path
    import re
    import unicodedata

    workspace = Path("/tmp_workspace")

    ALL_CRITERIA = [
        "results_dir_exists",
        "coverage_ratio",
        "content_match_ratio",
        "no_extra_files",
        "hard_constraint_pass",
        "overall_score",
    ]

    ZERO = {k: 0.0 for k in ALL_CRITERIA}
    results_dir = workspace / "results"
    gt_dir = workspace / "gt"

    if not gt_dir.exists() or not gt_dir.is_dir():
        return ZERO

    gt_files = sorted(
        p for p in gt_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".md"
    )
    if len(gt_files) == 0:
        return ZERO

    expected_names = {p.stem for p in gt_files}
    if not expected_names:
        return ZERO

    def normalize_md_content(text: str) -> str:
        """Normalize markdown content for comparison."""
        text = unicodedata.normalize("NFKC", str(text))
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Remove reference markers: [1], [2], etc.
        text = re.sub(r"\[\d+\]", "", text)
        # Treat straight and curved quotes as equivalent: " " 「」 -> "
        text = text.replace("\u201c", '"').replace("\u201d", '"')  # " "
        text = text.replace("\u300c", '"').replace("\u300d", '"')  # 「」
        # Do NOT strip URLs — their presence/absence is part of the evaluation
        # Ignore leading "生平" / "传记" section headers (e.g. "## 生平" or "生平" at start)
        text = re.sub(r"^\s*(#{1,6}\s*)?(生平|传记)\s*\n*", "", text, flags=re.IGNORECASE)
        # Convert markdown ATX headers to normal text: ## 标题 -> 标题 (keep title as plain text for comparison)
        text = re.sub(r"^\s*#{1,6}\s*([^\n]*)", r"\1", text, flags=re.MULTILINE)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        return text.strip()

    def read_and_normalize(path: Path) -> str | None:
        try:
            return normalize_md_content(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return None

    scores = dict(ZERO)
    scores["results_dir_exists"] = 1.0 if results_dir.exists() and results_dir.is_dir() else 0.0

    if not results_dir.exists() or not results_dir.is_dir():
        return scores

    pred_md_files = [p for p in results_dir.iterdir() if p.is_file() and p.suffix.lower() == ".md"]
    pred_names = {p.stem for p in pred_md_files}

    coverage = 0
    content_match = 0

    for gt_path in gt_files:
        name = gt_path.stem
        gt_content = read_and_normalize(gt_path)
        if gt_content is None:
            continue

        pred_path = results_dir / f"{name}.md"
        if pred_path.exists() and pred_path.is_file():
            coverage += 1
            pred_content = read_and_normalize(pred_path)
            if pred_content is not None and pred_content == gt_content:
                content_match += 1

    total = len(gt_files)
    scores["coverage_ratio"] = round(coverage / total, 4) if total > 0 else 0.0
    scores["content_match_ratio"] = round(content_match / total, 4) if total > 0 else 0.0

    extra_names = pred_names - expected_names
    extra_count = len(extra_names)
    scores["no_extra_files"] = round(max(0.0, 1.0 - 0.1 * extra_count), 4)

    hard_checks = ["results_dir_exists"]
    hard_pass = all(scores[k] == 1.0 for k in hard_checks)
    scores["hard_constraint_pass"] = 1.0 if hard_pass else 0.0

    if not hard_pass:
        scores["overall_score"] = 0.0
        return scores

    base_score = round(
        0.3 * scores["coverage_ratio"] + 0.7 * scores["content_match_ratio"],
        4,
    )
    scores["overall_score"] = round(base_score * scores["no_extra_files"], 4)
    return scores
