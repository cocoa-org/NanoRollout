def grade(iou_thresh=0.5, f1_pass=0.8, **kwargs) -> dict:
    """
    Grade the SAM3 debugging task.

    Compare predictions.json against gt_boxes.json,
    match by IoU>0.5, then judge pass/fail per case. overall_score = passed / total.

    Returns:
        dict: contains overall_score, passed, total, per_case
    """
    import json
    from pathlib import Path

    workspace_path = kwargs.get("workspace_path", "/tmp_workspace")
    pred_path = Path("/tmp_workspace/results") / "predictions.json"
    gt_path = Path(workspace_path) / "gt" / "gt_boxes.json"

    def _box_iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / union if union > 0 else 0.0

    def _match(pred_boxes, gt_boxes):
        used = set()
        tp = 0
        for gt in gt_boxes:
            best_iou, best_j = 0, -1
            for j, p in enumerate(pred_boxes):
                if j in used:
                    continue
                iou = _box_iou(p, gt)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thresh and best_j >= 0:
                used.add(best_j)
                tp += 1
        return tp, len(pred_boxes) - tp, len(gt_boxes) - tp

    def _f1(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return 2*p*r/(p+r) if p+r else 0.0

    if not pred_path.exists():
        return {"path_exists": 0.0, "overall_score": 0.0}

    with open(pred_path) as f:
        pred = json.load(f)
    with open(gt_path) as f:
        gt = json.load(f)

    per_case = {}
    passed = 0
    total = 0

    for name, gt_case in gt["cases"].items():
        gt_boxes = gt_case["boxes_xyxy"]
        pred_boxes = pred.get("cases", {}).get(name, {}).get("boxes_xyxy", [])
        tp, fp, fn = _match(pred_boxes, gt_boxes)
        f1 = _f1(tp, fp, fn)
        case_pass = f1 >= f1_pass
        per_case[name] = {
            "tp": tp, "fp": fp, "fn": fn,
            "f1": round(f1, 4),
            "pass": case_pass,
        }
        if case_pass:
            passed += 1
        total += 1

    return {
        "path_exists": 1.0,
        **{name: 1.0 if case["pass"] else 0.0 for name, case in per_case.items()},
        "overall_score": round(passed / total, 4) if total else 0.0,
    }
