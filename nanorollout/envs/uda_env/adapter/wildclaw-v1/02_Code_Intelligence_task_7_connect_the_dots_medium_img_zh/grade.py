def grade(**kwargs) -> dict:
    """
    Connect-the-dots (medium) image-only grading.

    - File existence (30%): result.png exists = 0.3
    - Image completeness (70%): VLM compares results/result.png with gt/gt.png, 0.7 weight
    """
    import os
    import json
    import base64
    import logging
    import time
    from pathlib import Path

    log = logging.getLogger("grade_connect_the_dots_medium_img")
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    base = Path("/tmp_workspace")
    result_image = base / "results" / "result.png"
    gt_image = base / "gt" / "gt.png"

    scores = {}
    image_score = 0.0
    file_exists_score = 0.0

    # ========== OpenAI client initialization ==========
    client = None
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=os.environ["OPENROUTER_BASE_URL"],
        )
    except Exception as e:
        log.error("OpenAI client initialization failed: %s", e)

    # ========== Image completeness score ==========
    if not result_image.exists():
        log.warning("result.png not found: %s", result_image)
        scores["image_score"] = 0.0
    elif not gt_image.exists():
        log.warning("gt.png not found: %s", gt_image)
        scores["image_score"] = 0.0
    else:
        file_exists_score = 1.0
        log.info("result.png exists, awarding file existence score 0.3")
        try:
            pred_b64 = base64.b64encode(result_image.read_bytes()).decode("utf-8")
            gt_b64 = base64.b64encode(gt_image.read_bytes()).decode("utf-8")

            if client:
                vlm_prompt = (
                    "你是一位评分裁判。请比较以下两张图片的相似度和完成度。\n"
                    "第一张图是标准答案（正确的连点成线结果），第二张图是待评估的答案。\n\n"
                    "这是一道连点成线谜题的解答结果。正确的解答应该是：\n"
                    "按编号顺序用线段依次连接所有带编号的点，最终形成一个可辨识的图案。\n\n"
                    "评分标准（0-10 分）：\n"
                    "- 10分：连线结果与标准答案完全一致或几乎完全一致，图案完整还原\n"
                    "- 7-9分：大部分点按正确顺序连接，图案主体清晰可辨，但有少量连线错误或遗漏\n"
                    "- 4-6分：部分点正确连接，能看出一些图案轮廓，但整体不完整或有较多错误\n"
                    "- 1-3分：只有少量连线正确，整体与标准答案差距较大\n"
                    "- 0分：完全没有连线、图片为空、或与标准答案完全不相关\n\n"
                    '请只返回一个 JSON 对象，示例：\n'
                    '{"score": 7, "reason": "大部分连线正确但右侧部分有偏差"}'
                )

                max_retries = 3
                for attempt in range(max_retries):
                    log.info("VLM image comparison attempt %d/%d...", attempt + 1, max_retries)
                    try:
                        resp = client.chat.completions.create(
                            model=os.environ.get("JUDGE_MODEL", "openai/gpt-5.4"),
                            messages=[{"role": "user", "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{gt_b64}"}},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{pred_b64}"}},
                                {"type": "text", "text": vlm_prompt},
                            ]}],
                            temperature=0.0,
                            max_tokens=256,
                        )
                        raw = resp.choices[0].message.content.strip()
                        log.info("VLM response: %s", raw[:300])
                        if raw.startswith("```"):
                            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                        result_json = json.loads(raw)
                        raw_score = int(result_json["score"])
                        image_score = round(max(0.0, min(1.0, raw_score / 10.0)), 4)
                        scores["image_judge_reason"] = result_json.get("reason", "")
                        scores["image_judge_method"] = "vlm"
                        log.info("VLM image score: %d/10, reason: %s", raw_score, scores["image_judge_reason"])
                        break
                    except Exception as e:
                        log.warning("VLM image comparison attempt %d failed: %s", attempt + 1, e)
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)

                if "image_judge_method" not in scores:
                    log.warning("VLM image comparison failed all 3 attempts, image score 0")
            else:
                log.warning("OpenAI client unavailable, image score 0")

        except Exception as e:
            log.warning("Image read failed: %s", e)

    scores["image_score"] = image_score
    scores["file_exists_score"] = file_exists_score

    # ========== Total score ==========
    scores["overall_score"] = round(0.3 * file_exists_score + 0.7 * scores["image_score"], 4)
    log.info(
        "Final scores: file_exists=%.1f × 30%% + image=%.4f × 70%% = overall=%.4f",
        file_exists_score, scores["image_score"], scores["overall_score"]
    )

    return scores
