def grade(**kwargs) -> dict:
    """
    Link-a-Pix (color pixel art) grading.

    - Image completeness (50%): VLM compares results/result.png with gt/gt.png
    - Description accuracy (50%): LLM evaluates semantic similarity to ground truth
    """
    import os
    import json
    import base64
    import logging
    import time
    from pathlib import Path

    log = logging.getLogger("grade_link_a_pix_color")
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    base = Path("/tmp_workspace")
    result_image = base / "results" / "result.png"
    description_file = base / "results" / "description.txt"
    gt_image = base / "gt" / "gt.png"

    gt_description = "在蓝天白云和太阳的背景下，一个穿着红色上衣、黑色裤子的人正站在绿色草地上骑滑板车。"

    scores = {}
    image_score = 0.0
    description_score = 0.0

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

    # ========== 1. Image completeness score (50%) ==========
    if not result_image.exists():
        log.warning("result.png not found: %s", result_image)
        scores["image_score"] = 0.0
    elif not gt_image.exists():
        log.warning("gt.png not found: %s", gt_image)
        scores["image_score"] = 0.0
    else:
        try:
            pred_b64 = base64.b64encode(result_image.read_bytes()).decode("utf-8")
            gt_b64 = base64.b64encode(gt_image.read_bytes()).decode("utf-8")

            if client:
                vlm_prompt = (
                    "你是一位评分裁判。请比较以下两张图片的相似度和完成度。\n"
                    "第一张图是标准答案（正确解答的连数字像素画），第二张图是待评估的答案。\n\n"
                    "这是一道连数字（Link-a-Pix）谜题的解答结果。正确的解答应该是：\n"
                    "每对同色同值的数字之间用路径连接，路径覆盖的方格用对应颜色填充，最终形成一幅像素画。\n\n"
                    "评分标准（0-10 分）：\n"
                    "- 10分：填色结果与标准答案完全一致或几乎完全一致，像素画完整还原\n"
                    "- 7-9分：大部分路径正确连接并填色，像素画主体清晰可辨，但有少量区域填色错误或遗漏\n"
                    "- 4-6分：部分路径正确，能看出一些填色区域，但整体像素画不完整或有较多错误\n"
                    "- 1-3分：只有少量填色正确，整体与标准答案差距较大\n"
                    "- 0分：完全没有填色、图片为空、或与标准答案完全不相关\n\n"
                    '请只返回一个 JSON 对象，示例：\n'
                    '{"score": 7, "reason": "大部分填色正确但右下角区域有偏差"}'
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

    # ========== 2. Description accuracy score (50%) ==========
    if not description_file.exists():
        log.warning("description.txt not found: %s", description_file)
        scores["description_score"] = 0.0
    else:
        pred_description = description_file.read_text(encoding="utf-8").strip()
        if not pred_description:
            log.warning("description.txt is empty")
            scores["description_score"] = 0.0
        else:
            log.info("Read description for evaluation: %s", pred_description[:200])

            # ---- Fallback: keyword-based text matching ----
            def keyword_fallback(pred: str) -> float:
                pred_lower = pred.lower()
                primary_kws = [
                    "滑板车", "滑板", "scooter", "skateboard", "kickboard"
                ]
                person_kws = [
                    "人", "人物", "小孩", "男孩", "女孩", "少年",
                    "person", "boy", "girl", "kid", "child", "rider"
                ]
                clothing_kws = [
                    "红色衣服", "红色上衣", "红衣", "红色",
                    "黑色裤子", "黑裤", "黑色"
                ]
                background_kws = [
                    "草地", "绿地", "草坪", "草", "grass",
                    "天空", "蓝天", "sky",
                    "太阳", "sun", "阳光",
                    "云", "白云", "cloud"
                ]

                has_primary = any(kw in pred_lower for kw in primary_kws)
                has_person = any(kw in pred_lower for kw in person_kws)
                clothing_hits = sum(1 for kw in clothing_kws if kw in pred_lower)
                bg_hits = sum(1 for kw in background_kws if kw in pred_lower)

                log.info(
                    "Keyword match — primary(scooter): %s, person: %s, clothing hits: %d, background hits: %d",
                    has_primary, has_person, clothing_hits, bg_hits
                )

                if has_primary and has_person:
                    return min(1.0, 0.8 + 0.03 * clothing_hits + 0.02 * bg_hits)
                if has_primary:
                    return min(0.75, 0.55 + 0.03 * clothing_hits + 0.02 * bg_hits)
                if has_person and bg_hits >= 2:
                    return min(0.45, 0.25 + 0.03 * clothing_hits + 0.02 * bg_hits)
                return min(0.2, 0.03 * clothing_hits + 0.02 * bg_hits)

            # ---- Prefer LLM as Judge ----
            llm_succeeded = False

            if client:
                judge_prompt = f"""你是一位评分裁判。请比较以下两段描述的语义相似度。

【标准答案】
{gt_description}

【待评估回答】
{pred_description}

评分标准（0-10 分）：
- 10分：完全准确地描述了相同的物体/场景，关键要素齐全（人物、骑滑板车、红色上衣、黑色裤子、蓝天白云、太阳、绿色草地）
- 7-9分：正确识别出主体是一个人在骑滑板车，且提及了部分背景或服装细节，但有遗漏或不够准确
- 4-6分：大致方向正确（如识别出人物或运动场景），但未准确识别出滑板车或关键要素有较大偏差
- 1-3分：描述有少量相关元素，但整体偏差较大
- 0分：完全不相关或为空

请只返回一个 JSON 对象，示例：
{{
    "score": 7,
    "reason": "正确识别了主体但缺少部分细节"
}}"""

                max_retries = 3
                for attempt in range(max_retries):
                    log.info("LLM Judge attempt %d/%d...", attempt + 1, max_retries)
                    try:
                        response = client.chat.completions.create(
                            model=os.environ.get("JUDGE_MODEL", "openai/gpt-5.4"),
                            messages=[{"role": "user", "content": judge_prompt}],
                            temperature=0.0,
                            max_tokens=256,
                        )

                        result_text = response.choices[0].message.content.strip()
                        log.info("LLM raw response: %s", result_text[:300])

                        if result_text.startswith("```"):
                            result_text = result_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

                        result_json = json.loads(result_text)
                        raw_score = int(result_json["score"])
                        description_score = round(max(0.0, min(1.0, raw_score / 10.0)), 4)
                        scores["desc_judge_reason"] = result_json.get("reason", "")
                        scores["desc_judge_method"] = "llm"
                        scores["llm_attempts"] = attempt + 1
                        llm_succeeded = True
                        log.info(
                            "LLM Judge succeeded — score: %d/10, reason: %s",
                            raw_score, scores["desc_judge_reason"]
                        )
                        break

                    except Exception as e:
                        log.warning("LLM Judge attempt %d failed: %s", attempt + 1, e)
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)

            if not llm_succeeded:
                log.warning("LLM Judge failed, falling back to keyword matching")
                description_score = keyword_fallback(pred_description)
                scores["desc_judge_method"] = "keyword_fallback"
                log.info("Keyword fallback score: %s", description_score)

    scores["description_score"] = description_score

    # ========== Total score ==========
    scores["overall_score"] = round(0.5 * scores["image_score"] + 0.5 * scores["description_score"], 4)
    log.info(
        "Final scores: image=%.4f × 50%% + description=%.4f × 50%% = overall=%.4f",
        scores["image_score"], scores["description_score"], scores["overall_score"]
    )

    return scores
