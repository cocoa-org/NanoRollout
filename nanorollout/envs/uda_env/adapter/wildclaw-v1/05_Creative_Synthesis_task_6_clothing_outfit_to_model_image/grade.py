def grade(**kwargs) -> dict:
    """
    Grade the clothing outfit task by checking classification, matching and model images.

    Returns:
        Dict mapping criterion names to scores (0.0 to 1.0)
    """
    import base64
    import json
    import os
    import re
    import time
    from openai import OpenAI
    from pathlib import Path

    GT_DIR = Path("/tmp_workspace/gt")
    GT_FILE = GT_DIR / "ground_truth.json"

    VLM_MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-5.4")
    OPENROUTER_BASE_URL = os.environ["OPENROUTER_BASE_URL"]

    ALL_CRITERIA = [
        "basic_requirements",
        "outfit_accuracy",
        "visual_quality",
        "overall_score",
    ]

    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=OPENROUTER_BASE_URL)

    def _call_vlm(messages, model=None, max_tokens=2048, retries=2):
        if model is None:
            model = VLM_MODEL
        for attempt in range(retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0,
                )
                return resp.choices[0].message.content
            except Exception as e:
                if attempt < retries:
                    time.sleep(2 ** attempt)
                else:
                    return None

    def _extract_json(text):
        if text is None:
            return None
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if m:
            text = m.group(1)
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            m2 = re.search(r"\{.*\}", text, re.DOTALL)
            if m2:
                try:
                    return json.loads(m2.group(0))
                except json.JSONDecodeError:
                    pass
            return None

    def _norm(s):
        if not isinstance(s, str):
            return str(s).strip().lower() if s is not None else ""
        return s.strip().lower().replace("-", " ").replace("\u2013", " ").replace("_", " ")

    def _read_image_b64(path):
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except Exception:
            return None

    def _get_outfit_items(outfit):
        items = set()
        for role in ["top", "bottom", "shoes"]:
            f = outfit.get(role, {}).get("original_file", "") if isinstance(outfit.get(role), dict) else ""
            if f:
                items.add(f)
        return items

    GT = json.loads(GT_FILE.read_text())
    GT_OUTFITS = GT["outfits"]
    GT_MAPPING = GT["photo_mapping"]

    scores = {}
    workspace = Path("/tmp_workspace/results")

    outfits_file = workspace / "outfits.json"
    if not outfits_file.exists() or outfits_file.stat().st_size == 0:
        
        return {k: 0.0 for k in ALL_CRITERIA}

    try:
        pred_data = json.loads(outfits_file.read_text())
        if isinstance(pred_data, list):
            pred_outfits = pred_data
        elif isinstance(pred_data, dict):
            pred_outfits = pred_data.get("outfits", [])
        else:
            pred_outfits = []
    except (json.JSONDecodeError, KeyError, AttributeError):
        
        return {k: 0.0 for k in ALL_CRITERIA}

    has_4 = len(pred_outfits) == 4
    all_have_3 = all(pred.get("top") and pred.get("bottom") and pred.get("shoes") for pred in pred_outfits)

    all_files = set()
    for pred in pred_outfits:
        for role in ["top", "bottom", "shoes"]:
            f = pred.get(role, {}).get("original_file", "") if isinstance(pred.get(role), dict) else ""
            if f:
                all_files.add(f)
    items_unique = len(all_files) == 12

    images_found = 0
    image_paths = []
    for i in range(1, 5):
        for ext in ["png", "jpg", "jpeg", "webp"]:
            p = workspace / f"model_outfit_{i}.{ext}"
            if p.exists() and p.stat().st_size > 10_000:
                images_found += 1
                image_paths.append((i, p))
                break
        else:
            image_paths.append((i, None))

    basic_checks = [has_4, all_have_3, items_unique, images_found == 4]
    scores["basic_requirements"] = round(sum(basic_checks) / len(basic_checks), 2)

    # ── Outfit accuracy ──────────────────────────────────────────────

    gt_combos = [_get_outfit_items(o) for o in GT_OUTFITS]
    pred_combos = [_get_outfit_items(o) for o in pred_outfits]

    pred_to_gt = {}
    matched_gt, matched_pred = set(), set()
    for pi, ps in enumerate(pred_combos):
        if len(ps) != 3:
            continue
        for gi, gs in enumerate(gt_combos):
            if gi not in matched_gt and ps == gs:
                pred_to_gt[pi] = (gi, 1.0)
                matched_gt.add(gi)
                matched_pred.add(pi)
                break

    remaining = []
    for pi, ps in enumerate(pred_combos):
        if pi in matched_pred or len(ps) != 3:
            continue
        for gi, gs in enumerate(gt_combos):
            if gi in matched_gt:
                continue
            overlap = len(ps & gs)
            if overlap > 0:
                remaining.append((overlap / 3.0, pi, gi))
    remaining.sort(reverse=True)
    for sc, pi, gi in remaining:
        if pi not in matched_pred and gi not in matched_gt:
            pred_to_gt[pi] = (gi, sc)
            matched_gt.add(gi)
            matched_pred.add(pi)

    combo_score = round(sum(sc for _, sc in pred_to_gt.values()) / 4, 2)

    classification_correct = 0
    total_items = 0
    for pred_outfit in pred_outfits:
        for role in ["top", "bottom", "shoes"]:
            item = pred_outfit.get(role, {})
            orig_file = item.get("original_file", "")
            pred_cat = _norm(item.get("category", ""))
            if orig_file in GT_MAPPING:
                total_items += 1
                if pred_cat == _norm(GT_MAPPING[orig_file]["category"]):
                    classification_correct += 1
    class_score = round(classification_correct / 12, 2) if total_items > 0 else 0.0

    gender_correct = 0
    for pi, pred_o in enumerate(pred_outfits):
        if pi in pred_to_gt and pred_to_gt[pi][1] == 1.0:
            gi = pred_to_gt[pi][0]
            if _norm(pred_o.get("gender", "")) == _norm(GT_OUTFITS[gi].get("gender", "")):
                gender_correct += 1
    gender_score = round(gender_correct / 4, 2)

    scores["outfit_accuracy"] = round((combo_score + class_score + gender_score) / 3, 2)

    # ── Visual quality (VLM judge) ───────────────────────────────────

    gender_scores, outfit_match_scores, quality_scores = [], [], []

    for idx, path in image_paths:
        if path is None:
            gender_scores.append(0.0)
            outfit_match_scores.append(0.0)
            quality_scores.append(0.0)
            continue

        pi = idx - 1
        if pi not in pred_to_gt or pred_to_gt[pi][1] < 1.0:
            gender_scores.append(0.0)
            outfit_match_scores.append(0.0)
            quality_scores.append(0.0)
            continue

        gi = pred_to_gt[pi][0]
        gt_outfit = GT_OUTFITS[gi]
        b64 = _read_image_b64(path)
        if b64 is None:
            gender_scores.append(0.0)
            outfit_match_scores.append(0.0)
            quality_scores.append(0.0)
            continue

        ext = str(path).rsplit(".", 1)[-1].lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}.get(ext, "image/png")

        content = [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {
                "type": "text",
                "text": (
                    f"Evaluate fashion model image.\n"
                    f"Expected gender: {gt_outfit.get('gender', 'unknown')}\n"
                    f"Expected top: {gt_outfit.get('top', {}).get('description', '')}\n"
                    f"Expected bottom: {gt_outfit.get('bottom', {}).get('description', '')}\n"
                    f"Expected shoes: {gt_outfit.get('shoes', {}).get('description', '')}\n\n"
                    "Rate 0.0-1.0:\n"
                    "1. gender_correct\n2. outfit_match\n3. quality (proportions, details, full body)\n\n"
                    "Return ONLY valid JSON:\n"
                    '{"gender_correct": <float>, "outfit_match": <float>, "quality": <float>}'
                ),
            },
        ]
        result = _call_vlm([{"role": "user", "content": content}], max_tokens=512)
        data = _extract_json(result)
        if data:
            g = min(1.0, max(0.0, float(data.get("gender_correct", 0))))
            o = min(1.0, max(0.0, float(data.get("outfit_match", 0))))
            q = min(1.0, max(0.0, float(data.get("quality", 0))))
            gender_scores.append(g)
            outfit_match_scores.append(o)
            quality_scores.append(q)
        else:
            gender_scores.append(0.0)
            outfit_match_scores.append(0.0)
            quality_scores.append(0.0)

    n = max(len(gender_scores), 1)
    scores["visual_quality"] = round(
        (sum(gender_scores) / n + sum(outfit_match_scores) / n + sum(quality_scores) / n) / 3, 2,
    )

    basic = scores["basic_requirements"]
    if basic < 1.0:
        scores["overall_score"] = 0.0
    else:
        scores["overall_score"] = round(
            0.35 * scores["outfit_accuracy"] + 0.65 * scores["visual_quality"], 4,
        )

    return scores
