def grade(**kwargs) -> dict:
    """
    Jigsaw puzzle grading (max 25 points, normalized to 0-1).

    - Grid positions:  9 pts (1 pt per correct position)
    - Transforms:      5 pts (1 pt per correct key:value; 0 if count != 5)
    - Distractors:     6 pts (1 pt per correct distractor; 0 if count != 6)
    - Assembled image: 5 pts (VLM checks if assembled.png is a valid 3x3 puzzle)
    """
    import os
    import json
    import base64
    import logging
    from pathlib import Path
    from PIL import Image

    log = logging.getLogger("grade_jigsaw_puzzle")
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    TOTAL_POINTS = 25
    base = Path("/tmp_workspace")
    result_file = base / "results" / "result.json"
    assembled_path = base / "results" / "assembled.png"

    gt_solution = {
        (0, 0): "piece_13.png", (0, 1): "piece_14.png", (0, 2): "piece_08.png",
        (1, 0): "piece_10.png", (1, 1): "piece_11.png", (1, 2): "piece_07.png",
        (2, 0): "piece_04.png", (2, 1): "piece_03.png", (2, 2): "piece_01.png",
    }
    gt_transforms = {
        "piece_04.png": "rotate_270",
        "piece_07.png": "rotate_270",
        "piece_08.png": "rotate_180",
        "piece_11.png": "rotate_90",
        "piece_14.png": "rotate_90",
    }
    gt_distractors = {"piece_02.png", "piece_05.png", "piece_06.png",
                      "piece_09.png", "piece_12.png", "piece_15.png"}

    scores = {}
    points = 0

    # ========== Read prediction results ==========
    if not result_file.exists():
        log.warning("result.json not found: %s", result_file)
        scores["overall_score"] = 0.0
        return scores

    try:
        pred = json.loads(result_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        log.warning("result.json JSON parse failed: %s", e)
        scores["overall_score"] = 0.0
        return scores

    pred_grid = pred.get("grid", [])
    pred_transforms = pred.get("transforms", {})
    pred_distractors = list(pred.get("distractors", []))

    # ========== Grid positions (9 points) ==========
    grid_points = 0
    for r in range(3):
        for c in range(3):
            try:
                if pred_grid[r][c] == gt_solution[(r, c)]:
                    grid_points += 1
            except (IndexError, TypeError):
                pass

    scores["grid_points"] = grid_points
    points += grid_points
    log.info("Grid: %d/9 points", grid_points)

    # ========== Transforms (5 points) ==========
    if len(pred_transforms) != 5:
        transforms_points = 0
        log.info("Transforms: count %d ≠ 5, scoring 0 for this dimension", len(pred_transforms))
    else:
        transforms_points = 0
        for piece_name, gt_t in gt_transforms.items():
            if pred_transforms.get(piece_name) == gt_t:
                transforms_points += 1
        log.info("Transforms: %d/5 points", transforms_points)

    scores["transforms_points"] = transforms_points
    points += transforms_points

    # ========== Distractors (6 points) ==========
    if len(pred_distractors) != 6:
        distractors_points = 0
        log.info("Distractors: count %d ≠ 6, scoring 0 for this dimension", len(pred_distractors))
    else:
        pred_distractors_set = set(pred_distractors)
        distractors_points = len(pred_distractors_set & gt_distractors)
        log.info("Distractors: %d/6 points", distractors_points)

    scores["distractors_points"] = distractors_points
    points += distractors_points

    # ========== Assembled image (5 points) — VLM checks if valid 3×3 puzzle ==========
    assembly_points = 0
    if not assembled_path.exists():
        log.info("assembled.png not found, assembly dimension scores 0")
    else:
        try:
            img = Image.open(assembled_path)
            w, h = img.size
            if w != 600 or h != 600:
                log.info("assembled.png size %dx%d ≠ 600x600, assembly dimension scores 0", w, h)
            else:
                img_b64 = base64.b64encode(assembled_path.read_bytes()).decode("utf-8")
                try:
                    import time
                    from openai import OpenAI

                    client = OpenAI(
                        api_key=os.environ["OPENROUTER_API_KEY"],
                        base_url=os.environ["OPENROUTER_BASE_URL"],
                    )

                    vlm_prompt = (
                        "请判断这张图片是否是由多张小图片拼接而成的 3×3 九宫格图片。\n"
                        "只需要判断是否成功拼接成了一张完整的图片（不需要判断内容是否正确）。\n\n"
                        "判断标准：\n"
                        "- 如果图片看起来是由 9 块碎片拼在一起的（无论内容对不对、朝向对不对），回答 yes\n"
                        "- 如果图片是空白、损坏、不完整、或不是 3×3 拼接格式，回答 no\n\n"
                        '请只返回 JSON：{"assembled": true} 或 {"assembled": false}'
                    )

                    max_retries = 3
                    for attempt in range(max_retries):
                        log.info("VLM assembly check attempt %d/%d...", attempt + 1, max_retries)
                        try:
                            resp = client.chat.completions.create(
                                model=os.environ.get("JUDGE_MODEL", "openai/gpt-5.4"),
                                messages=[{"role": "user", "content": [
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                                    {"type": "text", "text": vlm_prompt},
                                ]}],
                                temperature=0.0,
                                max_tokens=128,
                            )
                            raw = resp.choices[0].message.content.strip()
                            log.info("VLM response: %s", raw[:200])
                            if raw.startswith("```"):
                                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                            vlm_result = json.loads(raw)
                            if vlm_result.get("assembled"):
                                assembly_points = 5
                                log.info("VLM verdict: assembled, assembly dimension scores 5")
                            else:
                                log.info("VLM verdict: not assembled, assembly dimension scores 0")
                            break
                        except Exception as e:
                            log.warning("VLM assembly check attempt %d failed: %s", attempt + 1, e)
                            if attempt < max_retries - 1:
                                time.sleep(2 ** attempt)

                except Exception as e:
                    log.error("OpenAI client initialization failed: %s, assembly dimension scores 0", e)

        except Exception as e:
            log.warning("assembled.png read failed: %s, assembly dimension scores 0", e)

    scores["assembly_points"] = assembly_points
    points += assembly_points

    # ========== Total score ==========
    scores["total_points"] = points
    scores["overall_score"] = round(points / TOTAL_POINTS, 4)

    log.info("Score breakdown: grid=%d + transforms=%d + distractors=%d + assembly=%d = %d/%d → overall=%.4f",
             grid_points, transforms_points, distractors_points, assembly_points,
             points, TOTAL_POINTS, scores["overall_score"])

    return scores
