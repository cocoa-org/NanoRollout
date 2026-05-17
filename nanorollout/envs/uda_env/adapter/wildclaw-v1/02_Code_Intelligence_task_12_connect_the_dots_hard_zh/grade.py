def grade(**kwargs) -> dict:
    """
    Connect-the-dots (hard) grading.

    - Image file existence (15%): result.png exists = 0.15
    - Description file existence (15%): description.txt exists and is non-empty = 0.15
    - Image completeness (35%): VLM compares results/result.png with gt/gt.png
    - Description accuracy (35%): LLM evaluates semantic similarity to ground truth
    """
    import os
    import json
    import base64
    import logging
    import time
    from pathlib import Path

    log = logging.getLogger("grade_connect_the_dots_hard")
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    base = Path("/tmp_workspace")
    result_image = base / "results" / "result.png"
    description_file = base / "results" / "description.txt"
    gt_image = base / "gt" / "gt.png"

    gt_description = "画面由多组彩色线条连成的图案组成：顶部是一个大型环形轮廓，中间有多条蜿蜒曲线相互交织，底部散布着若干个小五边形，整体构成一幅复杂的抽象装饰图案。"

    scores = {}
    image_score = 0.0
    description_score = 0.0
    img_file_exists = 0.0
    desc_file_exists = 0.0

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

    # ========== 1. Image completeness score (15% existence + 35% quality) ==========
    if not result_image.exists():
        log.warning("result.png not found: %s", result_image)
        scores["image_score"] = 0.0
    elif not gt_image.exists():
        log.warning("gt.png not found: %s", gt_image)
        scores["image_score"] = 0.0
    else:
        img_file_exists = 1.0
        log.info("result.png exists, awarding file existence score 0.15")
        try:
            pred_b64 = base64.b64encode(result_image.read_bytes()).decode("utf-8")
            gt_b64 = base64.b64encode(gt_image.read_bytes()).decode("utf-8")

            if client:
                vlm_prompt = (
                    "你是一位评分裁判。请比较以下两张图片的相似度和完成度。\n"
                    "第一张图是标准答案（正确的连点成线结果），第二张图是待评估的答案。\n\n"
                    "这是一道连点成线谜题的解答结果（困难难度，162个点按颜色分组连接）。正确的解答应该是：\n"
                    "按颜色分组，每组内按编号顺序用线段依次连接所有带编号的点，最终形成多组可辨识的图案。\n\n"
                    "评分标准（0-10 分）：\n"
                    "- 10分：连线结果与标准答案完全一致或几乎完全一致，所有组的图案完整还原\n"
                    "- 7-9分：大部分组的点按正确顺序连接，主体图案清晰可辨，但有少量连线错误或遗漏\n"
                    "- 4-6分：部分组正确连接，能看出一些图案轮廓，但整体不完整或有较多错误\n"
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

    # ========== 2. Description accuracy score (15% existence + 35% quality) ==========
    if not description_file.exists():
        log.warning("description.txt not found: %s", description_file)
        scores["description_score"] = 0.0
    else:
        pred_description = description_file.read_text(encoding="utf-8").strip()
        if not pred_description:
            log.warning("description.txt is empty")
            scores["description_score"] = 0.0
        else:
            desc_file_exists = 1.0
            log.info("description.txt exists and is non-empty, awarding description file existence score 0.15")
            log.info("Read description for evaluation: %s", pred_description[:200])

            # ---- Fallback: keyword-based text matching ----
            def keyword_fallback(pred: str) -> float:
                pred_lower = pred.lower()
                primary_kws = ["五边形", "pentagon", "多边形", "polygon"]
                secondary_kws = ["环形", "环", "loop", "circle", "圆"]
                context_kws = [
                    "曲线", "curve", "路径", "path",
                    "抽象", "abstract", "装饰", "decorative",
                    "彩色", "颜色", "color", "多组", "分组"
                ]

                has_primary = any(kw in pred_lower for kw in primary_kws)
                has_secondary = any(kw in pred_lower for kw in secondary_kws)
                context_hits = sum(1 for kw in context_kws if kw in pred_lower)

                log.info(
                    "Keyword match — primary(pentagon): %s, secondary(loop): %s, context hits: %d",
                    has_primary, has_secondary, context_hits
                )

                if has_primary and has_secondary:
                    return min(1.0, 0.8 + 0.05 * context_hits)
                if has_primary:
                    return min(0.8, 0.6 + 0.05 * context_hits)
                if has_secondary:
                    return min(0.4, 0.2 + 0.05 * context_hits)
                return min(0.2, 0.05 * context_hits)

            # ---- Prefer LLM as Judge ----
            llm_succeeded = False

            if client:
                judge_prompt = f"""你是一位评分裁判。请比较以下两段描述的语义相似度。

【标准答案】
{gt_description}

【待评估回答】
{pred_description}

评分标准（0-10 分）：
- 10分：完全准确地描述了相同的物体/场景，关键要素齐全
- 7-9分：正确识别出主体图案，但部分细节有遗漏或不够准确
- 4-6分：大致方向正确（如识别出是多组图形/抽象图案），但未准确描述具体特征
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
    scores["img_file_exists"] = img_file_exists
    scores["desc_file_exists"] = desc_file_exists

    # ========== Total score ==========
    scores["overall_score"] = round(
        0.15 * img_file_exists + 0.15 * desc_file_exists
        + 0.35 * scores["image_score"] + 0.35 * scores["description_score"],
        4
    )
    log.info(
        "Final scores: img_exists=%.1f×15%% + desc_exists=%.1f×15%% + image=%.4f×35%% + description=%.4f×35%% = overall=%.4f",
        img_file_exists, desc_file_exists,
        scores["image_score"], scores["description_score"], scores["overall_score"]
    )

    return scores
