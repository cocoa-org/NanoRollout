def grade(**kwargs) -> dict:
    """
    Academic homepage style transfer grading (resume version).
    Send generated homepage screenshot to VLM for rubric-based evaluation.

    Returns:
        dict of per-dimension scores (0.0-1.0)
    """
    import os
    import json
    import time
    import base64
    import logging
    from pathlib import Path

    log = logging.getLogger("grade_resume_homepage")
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    base = Path("/tmp_workspace")
    homepage_dir = base / "results"
    screenshot_path = homepage_dir / "screenshot.png"

    scores = {}

    # ========== Programmatic pre-checks ==========

    if not homepage_dir.exists():
        log.warning("results directory not found: %s", homepage_dir)
        return {"overall_score": 0.0, "error": "results directory not found"}

    html_files = list(set(homepage_dir.rglob("*.html")))
    scores["has_html"] = 1.0 if html_files else 0.0
    if not html_files:
        log.warning("No HTML files found in results directory")

    if not screenshot_path.exists():
        log.warning("screenshot.png not found: %s", screenshot_path)
        return {"overall_score": 0.0, "error": "screenshot.png not found"}

    screenshot_size = screenshot_path.stat().st_size
    if screenshot_size < 1024:
        log.warning("screenshot.png too small (%d bytes), may be invalid", screenshot_size)
        return {"overall_score": 0.0, "error": "screenshot.png too small, likely invalid"}

    scores["screenshot_exists"] = 1.0
    log.info("screenshot.png exists, size: %d bytes", screenshot_size)

    # ========== Read screenshot ==========

    img_b64 = base64.b64encode(screenshot_path.read_bytes()).decode("utf-8")

    # ========== Define VLM evaluation rubric ==========

    rubrics = {
        # ===== Style consistency =====
        "style_1_dual_column": "页面采用左侧边栏 + 右侧主内容区的双栏布局",
        "style_2_photo_round": "左侧栏包含圆形的个人头像或照片（非方形、非圆角矩形），且不能是灰色默认剪影占位符",
        "style_3_name_tagline": "左侧栏显示姓名，并紧跟一行简短的研究方向或身份标语",
        "style_4_social_vertical": "左侧栏的社交链接采用竖向排列，每行一个链接由图标加文字标签组成（如 Email、GitHub、Google Scholar 各占一行），而非仅小图标横向排列",
        "style_5_emoji_match": "各板块标题使用与 AcadHomepage 模板一致的 emoji：🔥 News、📝 Publications、🎖 Honors and Awards 或 Awards、📖 Educations、💻 Internships 或 Research Experience，不应使用其他替代 emoji（如 📄、🏆、🎓、🎤、💼 等均为不一致）",
        "style_6_paper_cards": "重点论文采用卡片式展示，左侧有缩略图或论文配图（非灰色占位符或 broken image），右侧有标题、作者、会议标签和链接按钮",
        "style_7_bold_author": "作者本人姓名 Shengyuan Ding 在论文作者列表中加粗显示",
        "style_8_news_date_format": "News 部分的日期使用 YYYY.MM 格式（如 2025.07），使用斜体样式，不使用方括号包裹的月份缩写格式（如 [Jan. 2026]）",
        "style_9_academic_style": "整体为简洁学术风格，配色协调，无花哨装饰",
        "style_10_section_order": "板块结构与学术主页模板一致，大致遵循 About、News、Publications、Awards 等顺序",
        "style_11_no_about_header": "主内容区顶部的个人简介直接展示文字，没有独立的板块标题（不应出现 About Me 等标题头）",
        "style_12_no_extra_sidebar": "左侧栏不包含模板中不存在的多余独立元素（如不应有单独的 CV 文字链接按钮或下载链接）",
        "style_13_edu_date_format": "Educations 部分的时间使用纯文字格式（如 2021.09 - 2025.06, B.E.），不使用彩色标签或徽章样式显示年份",
        "style_14_venue_tag_style": "论文的会议标签使用素雅文字样式（如直接文字 ICCV 2025），不使用彩色胶囊或亮色徽章样式",
        "style_15_no_topnav": "页面顶部不包含水平导航菜单栏（如 ABOUT / NEWS / PUBLICATIONS 等横排链接），AcadHomepage 模板不使用顶部导航菜单",
        "style_16_no_extra_sections": "页面不包含 AcadHomepage 模板中不存在的多余内容板块（如 Skills、技能、Languages 等自行添加的板块均不应出现）",
        "style_17_sections_vertical": "所有主内容板块在页面中采用上下垂直排列（与模板一致），不将多个板块（如 Awards 和 Services）并排为左右双列布局",
        "style_18_edu_standalone": "教育经历（Educations）作为主内容区的独立板块展示（带有板块标题如 📖 Educations），而非嵌入在个人简介或 Biography 区域内",
        # ===== Prompt requirements =====
        "prompt_19_three_papers": "Publications 部分恰好展示三篇论文：MM-IFEngine（ICCV 2025）、ARM-Thinker（CVPR 2026）、OmniAlign-V（ACL 2025），不多不少",
        "prompt_20_mmifengine": "包含论文 MM-IFEngine: Towards Multimodal Instruction Following（ICCV 2025），且 Shengyuan Ding 标记为一作（排名第一）",
        "prompt_21_armthinker": "包含论文 ARM-Thinker: Reinforcing Multimodal Generative Reward Models with Agentic Tool Use and Visual Reasoning（CVPR 2026），且 Shengyuan Ding 标记为一作（排名第一）",
        "prompt_22_omnialign": "包含论文 OmniAlign-V: Towards Enhanced Alignment of MLLMs with Human Preference（ACL 2025），且 Shengyuan Ding 标记为共同一作（* 标记或等价标注）",
        "prompt_23_news_filter": "News 部分只展示 2025 年的动态，不包含 2024 年及更早或 2026 年及之后的条目",
        # ===== Content consistency =====
        "content_24_name": "姓名正确显示为 Shengyuan Ding",
        "content_25_affiliation": "身份显示为复旦大学博士生（Fudan University Ph.D. student）",
        "content_26_advisor": "导师信息提及 Prof. Dahua Lin",
        "content_27_research": "研究方向描述包含 Multimodal Large Language Models 或 instruction following 或 reward modeling 相关表述",
        "content_28_education": "教育经历包含复旦大学博士（2025.09 起）和南京大学本科（2021.09-2025.06）信息",
        "content_29_internship": "页面提及上海人工智能实验室研究实习经历（Shanghai AI Laboratory）",
        "content_30_contact": "页面包含联系方式区域，展示了邮箱、Google Scholar、GitHub 等入口",
        "content_31_paper_info": "每篇论文的标题、作者列表、会议名称和年份信息均完整展示",
        "content_32_paper_links": "每篇论文旁边有链接入口文字或按钮（如 arXiv、code、project page 等）",
        "content_33_paper_desc": "每篇论文附有一句话描述或简介",
        "content_34_awards": "获奖信息展示了至少 3 项奖项（国家奖学金 National Scholarship、华为奖学金 Huawei Scholarship、MCM M Award、南京大学优秀毕业生 Outstanding Graduate）",
        "content_35_services": "页面包含 Professional Services 或 Reviewer 相关信息（如 CVPR、COLM、TMLR 审稿人）",
        "content_36_opensource": "页面包含开源贡献信息（VLMEvalKit Maintainer）",
        # ===== Visual resource loading =====
        "visual_37_real_photo": "左侧栏的头像显示了真实的人物照片，非灰色默认头像剪影、非 broken image 图标、非空白占位符",
        "visual_38_paper_thumbs": "三篇论文卡片的左侧缩略图均实际显示了与论文内容相关的图片（如论文 figure、方法示意图），非灰色占位符、非写有 placeholder 文字的色块、非 broken image",
        "visual_39_social_icons": "左侧栏的社交链接图标均正常加载并显示为可辨识的品牌或功能图形（如邮件信封、GitHub logo 等），非缺失图标、非 broken image 方框、非空白",
    }

    rubric_list = "\n".join(
        f"  - {rid}: {desc}" for rid, desc in rubrics.items()
    )

    # ========== VLM as Judge ==========

    vlm_succeeded = False
    last_error = None

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=os.environ["OPENROUTER_BASE_URL"],
        )

        judge_prompt = (
            "你是一位网页评估专家。下面是一张学术个人主页的完整截图。\n"
            "请根据截图内容，逐条评估以下 rubric，判断该网页是否满足每条要求。\n\n"
            "评估规则：\n"
            "- 对每条 rubric 给出 1（满足）或 0（不满足）\n"
            "- 仔细观察截图中的文字、布局、板块结构等细节\n"
            "- 如有不确定，偏向给 0\n\n"
            f"Rubric 列表：\n{rubric_list}\n\n"
            "请严格按以下 JSON 格式返回（不要包含任何其他文字）：\n"
            "{\n"
            '    "rubric_id": 0 或 1,\n'
            "    ...\n"
            "}\n"
            "确保返回的 JSON 包含上述所有 rubric_id 作为键。"
        )

        max_retries = 3
        for attempt in range(max_retries):
            log.info("VLM Judge request %d/%d...", attempt + 1, max_retries)
            try:
                resp = client.chat.completions.create(
                    model=os.environ.get("JUDGE_MODEL", "openai/gpt-5.4"),
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_b64}",
                                },
                            },
                            {"type": "text", "text": judge_prompt},
                        ],
                    }],
                    temperature=0.0,
                    max_tokens=2048,
                )

                raw = resp.choices[0].message.content.strip()
                log.info("VLM raw response: %s", raw[:600])

                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

                result = json.loads(raw)

                for rid in rubrics:
                    scores[rid] = float(max(0, min(1, int(result.get(rid, 0)))))

                scores["judge_method"] = "vlm"
                scores["vlm_attempts"] = attempt + 1
                vlm_succeeded = True
                log.info("VLM Judge succeeded")
                break

            except Exception as e:
                last_error = e
                log.warning("VLM Judge attempt %d failed: %s", attempt + 1, e)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

    except Exception as e:
        last_error = e
        log.error("OpenAI client initialization failed: %s", e)

    if not vlm_succeeded:
        log.warning("VLM Judge all attempts failed, all rubrics scored 0")
        if last_error:
            scores["judge_error"] = str(last_error)
        for rid in rubrics:
            scores[rid] = 0.0
        scores["judge_method"] = "failed"

    # ========== Aggregate scores ==========

    all_rubric_keys = list(rubrics.keys())
    style_keys = [k for k in all_rubric_keys if k.startswith("style_")]
    prompt_keys = [k for k in all_rubric_keys if k.startswith("prompt_")]
    content_keys = [k for k in all_rubric_keys if k.startswith("content_")]
    visual_keys = [k for k in all_rubric_keys if k.startswith("visual_")]

    style_avg = (
        sum(scores.get(k, 0.0) for k in style_keys) / len(style_keys)
        if style_keys else 0.0
    )
    prompt_avg = (
        sum(scores.get(k, 0.0) for k in prompt_keys) / len(prompt_keys)
        if prompt_keys else 0.0
    )
    content_avg = (
        sum(scores.get(k, 0.0) for k in content_keys) / len(content_keys)
        if content_keys else 0.0
    )
    visual_avg = (
        sum(scores.get(k, 0.0) for k in visual_keys) / len(visual_keys)
        if visual_keys else 0.0
    )

    scores["style_score"] = round(style_avg, 4)
    scores["prompt_score"] = round(prompt_avg, 4)
    scores["content_score"] = round(content_avg, 4)
    scores["visual_score"] = round(visual_avg, 4)

    total_pass = sum(scores.get(k, 0.0) for k in all_rubric_keys)
    total_count = len(all_rubric_keys)
    scores["overall_score"] = round(total_pass / total_count, 4)

    log.info(
        "Final scores: style=%.4f, prompt=%.4f, content=%.4f, visual=%.4f, overall=%.4f (%d/%d)",
        scores["style_score"], scores["prompt_score"],
        scores["content_score"], scores["visual_score"],
        scores["overall_score"],
        int(total_pass), total_count,
    )

    return scores
