def grade(**kwargs) -> dict:
    """
    04_Search_Retrieval_task_1_google_scholar_search

    Args:

    Returns:
        0 or 1
    """
    import os
    import json
    import logging
    from pathlib import Path

    log = logging.getLogger("04_Search_Retrieval_task_1_google_scholar_search Grading Start!")
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    description_file = Path("/tmp_workspace/results/results.md")
    scores = {}

    if not description_file.exists():
        log.warning("results.md not found: %s", description_file)
        scores["overall_score"] = 0.0
        return scores

    pred_description = description_file.read_text(encoding="utf-8").strip()
    if not pred_description:
        log.warning("results.md is empty")
        scores["overall_score"] = 0.0
        return scores

    log.info("Content in MD: %s", pred_description[:200])

    llm_succeeded = False
    last_error = None

    try:
        import time
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=os.environ["OPENROUTER_BASE_URL"],
        )

        gt_description = "\n".join([
            "1. Ziyu Liu → Dahua Lin → Boyang Deng →  Geoffrey Hinton",
            "2. Ziyu Liu → Yuhang Zang → Joshua M. Susskind → Geoffrey Hinton",
            "3. Ziyu Liu → Zuxuan Wu → Leonid Sigal → Geoffrey Hinton",
            "4. Ziyu Liu → Yu-Gang Jiang → Leonid Sigal → Geoffrey Hinton"
        ])
        judge_prompt = f"""你是一位评分裁判。请你依据标准答案和待评估回答进行打分。

其中标准答案是几条关系链（包含起点、终点和中间节点），而待评估回答中可能存在一条或多条关系链，也可能不存在任何关系链。
两条关系链相同的判定标准为：起点、终点以及中间节点完全一致，如果任一节点不同或节点数目不同，不可以认为是同一关系链。


评分具体要求是：
1. 如果待评估回答中不包含任何关系链条，返回 score 为 0。
2. 如果待评估回答中存在任一条关系链条，和标准答案中的任意一条关系链相同(要求起点、终点和中间节点完全一致)，就返回 score 为 1。
3. 如果待评估回答中的所有关系链条，不存在任何一条和标准答案中的相同(要求起点、终点和中间节点完全一致)，就返回 score 为 0。
4. 学者的名字可以是缩写。

【标准答案】
{gt_description}

【待评估回答】
{pred_description}

请只返回一个 JSON 对象，格式：{{"score": <0或1>, "reason": "<简要理由>"}}"""

        max_retries = 3
        for attempt in range(max_retries):
            log.info("LLM Judge request %d/%d...", attempt + 1, max_retries)
            try:
                response = client.chat.completions.create(
                    model=os.environ.get("JUDGE_MODEL", "openai/gpt-5.4"),
                    messages=[{"role": "user", "content": judge_prompt}],
                    temperature=0,
                )

                result_text = response.choices[0].message.content.strip()
                log.info("LLM raw response: %s", result_text[:300])

                if result_text.startswith("```"):
                    result_text = result_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

                result_json = json.loads(result_text)
                raw_score = float(result_json.get("score", 0))
                scores["overall_score"] = raw_score
                scores["judge_reason"] = result_json.get("reason", "")
                llm_succeeded = True
                log.info("LLM Judge succeeded — score: %s, reason: %s",
                         raw_score, scores["judge_reason"])
                break

            except Exception as e:
                last_error = e
                log.warning("LLM Judge attempt %d failed: %s", attempt + 1, e)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

    except Exception as e:
        last_error = e
        log.error("OpenAI client initialization failed: %s", e)

    if not llm_succeeded and last_error:
        scores["judge_error"] = str(last_error)

    # ---- Fall back to keyword matching when all LLM attempts fail ----
    if not llm_succeeded:
        log.warning("LLM Judge failed all 3 attempts, scoring 0")
        scores["overall_score"] = 0

    log.info("Final score: overall_score=%s",
             scores["overall_score"])

    return scores
