def grade(**kwargs) -> dict:
    """
    Grade the SCP crawling task.

    Returns:
        Dict mapping criterion names to scores (0.0 to 1.0)
    """
    import json
    import re
    from pathlib import Path

    gt_file = Path("/tmp_workspace") / "gt" / "ground_truth.json"
    output_root = Path("/tmp_workspace/results")
    summary_file = output_root / "summary.jsonl"

    ALL_CRITERIA = {
        "output_root_exists": 0.0,
        "item_directories_created": 0.0,
        "summary_exists": 0.0,
        "summary_valid_jsonl": 0.0,
        "summary_item_coverage": 0.0,
        "summary_class_accuracy": 0.0,
        "summary_n_mask_accuracy": 0.0,
        "text_files_present": 0.0,
        "text_anchor_recall": 0.0,
        "image_count_accuracy": 0.0,
        "jpeg_file_validity": 0.0,
        "overall_score": 0.0,
    }

    if not gt_file.exists():
        return ALL_CRITERIA | {"error": f"missing gt file: {gt_file}"}

    gt = json.loads(gt_file.read_text(encoding="utf-8"))
    by_item = gt.get("by_item", {})
    expected_items = sorted(by_item.keys())
    expected_item_set = set(expected_items)

    def normalize_space(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def normalize_item(text: str) -> str:
        text = normalize_space(str(text)).upper()
        m = re.search(r"SCP[-\s]*0*([1-9]\d*|0)", text)
        if not m:
            return ""
        return f"SCP-{int(m.group(1)):03d}"

    def normalize_class(text: str) -> str:
        text = normalize_space(str(text)).lower()
        text = text.replace("object class:", "").replace("containment class:", "")
        return re.sub(r"[^a-z0-9]+", "", text)

    def looks_like_jpeg(path: Path) -> bool:
        try:
            data = path.read_bytes()
        except Exception:
            return False
        return len(data) >= 4 and data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"

    scores = dict(ALL_CRITERIA)

    if not output_root.exists() or not output_root.is_dir():
        return scores | {"error": f"output dir not found: {output_root}"}

    scores["output_root_exists"] = 1.0

    existing_dirs = {
        p.name for p in output_root.iterdir()
        if p.is_dir() and re.fullmatch(r"scp-\d{3}", p.name)
    }
    dir_hits = len(existing_dirs & {item.lower() for item in expected_items})
    scores["item_directories_created"] = round(dir_hits / len(expected_items), 4)

    if summary_file.exists() and summary_file.is_file():
        scores["summary_exists"] = 1.0
    else:
        scores["overall_score"] = round(
            0.10 * scores["item_directories_created"],
            4,
        )
        return scores | {"error": f"missing summary file: {summary_file}"}

    lines = [line for line in summary_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    parsed = []
    for idx, line in enumerate(lines, start=1):
        try:
            parsed.append(json.loads(line))
        except Exception as exc:
            return scores | {"error": f"invalid json on line {idx}: {exc}"}

    scores["summary_valid_jsonl"] = 1.0

    predicted = {}
    duplicate_items = set()
    malformed_rows = 0

    for row in parsed:
        if not isinstance(row, dict):
            malformed_rows += 1
            continue
        item = normalize_item(row.get("item", ""))
        if not item:
            malformed_rows += 1
            continue

        raw_class = row.get("class")
        raw_mask = row.get("n_mask")
        try:
            n_mask = int(raw_mask)
        except Exception:
            malformed_rows += 1
            continue

        entry = {
            "class": normalize_class(raw_class),
            "n_mask": n_mask,
        }
        if item in predicted:
            duplicate_items.add(item)
        predicted[item] = entry

    pred_item_set = set(predicted.keys())
    tp_items = len(pred_item_set & expected_item_set)
    item_recall = tp_items / len(expected_item_set) if expected_item_set else 0.0
    item_precision = tp_items / len(pred_item_set) if pred_item_set else 0.0
    item_f1 = (
        2 * item_recall * item_precision / (item_recall + item_precision)
        if (item_recall + item_precision) else 0.0
    )
    if duplicate_items or malformed_rows:
        item_f1 *= 0.0
    scores["summary_item_coverage"] = round(item_f1, 4)

    class_correct = 0
    mask_correct = 0
    for item in expected_items:
        pred = predicted.get(item)
        target = by_item[item]
        if pred and pred["class"] == normalize_class(target["class"]):
            class_correct += 1
        if pred and pred["n_mask"] == int(target["n_mask"]):
            mask_correct += 1

    total_items = len(expected_items) or 1
    scores["summary_class_accuracy"] = round(class_correct / total_items, 4)
    scores["summary_n_mask_accuracy"] = round(mask_correct / total_items, 4)

    text_present = 0
    anchor_total = 0
    anchor_hits = 0
    image_count_ok = 0
    expected_jpeg_files = 0
    valid_jpeg_files = 0

    for item in expected_items:
        item_dir = output_root / item.lower()
        target = by_item[item]

        text_file = item_dir / "text.md"
        if text_file.exists() and text_file.is_file():
            content = text_file.read_text(encoding="utf-8", errors="ignore").strip()
            if content:
                text_present += 1
                normalized_content = normalize_space(content).lower()
                anchors = target.get("text_anchors", [])
                for anchor in anchors:
                    anchor_total += 1
                    if normalize_space(anchor).lower() in normalized_content:
                        anchor_hits += 1
            else:
                anchor_total += len(target.get("text_anchors", []))
        else:
            anchor_total += len(target.get("text_anchors", []))

        expected_image_count = int(target.get("image_count", 0))
        expected_names = {f"{idx}.jpg" for idx in range(1, expected_image_count + 1)}
        actual_names = set()
        if item_dir.exists() and item_dir.is_dir():
            for path in item_dir.iterdir():
                if path.is_file() and path.name != "text.md":
                    actual_names.add(path.name)

        if actual_names == expected_names:
            image_count_ok += 1

        for idx in range(1, expected_image_count + 1):
            expected_jpeg_files += 1
            img_path = item_dir / f"{idx}.jpg"
            if img_path.exists() and img_path.is_file() and looks_like_jpeg(img_path):
                valid_jpeg_files += 1

    scores["text_files_present"] = round(text_present / total_items, 4)
    scores["text_anchor_recall"] = round(anchor_hits / anchor_total, 4) if anchor_total else 1.0
    scores["image_count_accuracy"] = round(image_count_ok / total_items, 4)
    scores["jpeg_file_validity"] = (
        round(valid_jpeg_files / expected_jpeg_files, 4) if expected_jpeg_files else 1.0
    )

    scores["overall_score"] = round(
        0.10 * scores["item_directories_created"]
        + 0.02 * scores["summary_exists"]
        + 0.05 * scores["summary_valid_jsonl"]
        + 0.10 * scores["summary_item_coverage"]
        + 0.15 * scores["summary_class_accuracy"]
        + 0.20 * scores["summary_n_mask_accuracy"]
        + 0.05 * scores["text_files_present"]
        + 0.10 * scores["text_anchor_recall"]
        + 0.10 * scores["image_count_accuracy"]
        + 0.13 * scores["jpeg_file_validity"],
        4,
    )

    notes = []
    if duplicate_items:
        notes.append(f"duplicate_items={sorted(duplicate_items)[:10]}")
    if malformed_rows:
        notes.append(f"malformed_rows={malformed_rows}")
    if notes:
        scores["warning"] = "; ".join(notes)

    return scores
