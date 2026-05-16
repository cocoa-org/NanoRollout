def grade(**kwargs) -> dict:
    """
    Grade the arxiv digest task.

    Returns:
        Dict mapping criterion names to scores (0.0 to 1.0)
    """
    from pathlib import Path
    import re

    ALL_CRITERIA = [
        "classify_multimodal",
        "classify_medical",
        "classify_generation",
        "classify_driving",
        "classify_3d",
        "metadata_appendix_flags",
        "metadata_split_consistency",
        "metadata_simpleocr",
        "metadata_exploring_multimodal_lmms",
        "metadata_nolan",
        "metadata_weavetime",
        "metadata_global_local_dual_perception",
        "metadata_see_it_say_it_sorted",
        "metadata_dynamicgtr",
        "metadata_dynamic_multimodal_activation_steering",
        "metadata_dr_seg",
        "metadata_cccaption",
        "interest_count",
        "interest_selection",
        "benchmark_charxiv",
        "benchmark_caprl_avg",
        "benchmark_cccaption_avg",
        "benchmark_caprl_infovqa",
        "classify_score",
        "metadata_score",
        "interest_score",
        "overall_score",
    ]

    scores = {}
    workspace = Path("/tmp_workspace/results")
    digest = workspace / "arxiv_digest.md"

    if not digest.exists() or len(digest.read_text().strip()) < 100:
        return {k: 0.0 for k in ALL_CRITERIA}

    content = digest.read_text()

    def extract_section(markdown_text: str, heading_pattern: str) -> str:
        """
        Extract section content under the first heading that matches heading_pattern.
        Section ends at the next heading of the same or higher level.
        """
        h = re.search(heading_pattern, markdown_text, re.MULTILINE | re.IGNORECASE)
        if not h:
            return ""

        heading_line = h.group(0)
        level_match = re.match(r"^#{1,6}", heading_line)
        if not level_match:
            return ""
        level = len(level_match.group(0))

        rest = markdown_text[h.end():]
        next_h = re.search(rf"^#{{1,{level}}}\s+", rest, re.MULTILINE)
        return rest[: next_h.start()] if next_h else rest

    # --- Classification accuracy ---
    # Strictly scope to the "### Classification" block, then require exact
    # matches for the category "####" sub-headings.
    strict_headings = [
        ("multimodal", "Multimodal / Vision-Language Models"),
        ("medical", "Medical Image Analysis"),
        ("generation", "Image / Video Generation & Editing"),
        ("driving", "Autonomous Driving / Robotics / Embodied AI"),
        ("3d", "3D Vision / Reconstruction / Gaussian Splatting"),
    ]

    classification_section = extract_section(content, r"^###\s+Classification\s*$")

    # Map strict headings to their positions
    strict_heading_map = {}
    for cat_id, heading_text in strict_headings:
        m = re.search(
            rf"^####\s+{re.escape(heading_text)}\s*$",
            classification_section,
            re.MULTILINE,
        )
        if m:
            strict_heading_map[m.start()] = (m.start(), m.end(), cat_id)

    # Find ALL #### headings (including "Others") to use as boundaries
    all_h4 = [(m.start(), m.end()) for m in re.finditer(r"^####\s+", classification_section, re.MULTILINE)]
    all_h4.sort()

    # Extract section text: from each strict heading to the next #### heading
    sections = {}
    for idx, (h_start, h_end) in enumerate(all_h4):
        if h_start not in strict_heading_map:
            continue
        _, s_end, cat_id = strict_heading_map[h_start]
        next_starts = [s for s, _ in all_h4 if s > h_start]
        section_end = next_starts[0] if next_starts else len(classification_section)
        sections[cat_id] = classification_section[s_end:section_end]

    # Ground-truth: checkpoint papers per category (distinctive keywords)
    ground_truth = {
        "multimodal": [
            r"SimpleOCR",
            r"Exploring Multimodal LMMs",
            r"NoLan",
            r"WeaveTime",
            r"Global.Local Dual Perception",
            r"See It, Say It, Sorted",
            r"DynamicGTR",
            r"Dynamic Multimodal Activation Steering",
            r"Dr\.\s*Seg",
            r"CCCaption",
        ],
        "medical": [
            r"(?:Diagnostic Trace|Visual Cognition.guided.*Chest X.Ray)",
            r"(?:Brain Tumor Segmentation.*Non.Enhancing)",
            r"SigVLP",
        ],
        "generation": [
            r"SkyReels.?V4",
            r"MultiAnimate",
            r"(?:Accelerating Diffusion.*Pipeline|Hybrid Data.Pipeline.*Diffusion)",
        ],
        "driving": [
            r"(?:World Guidance|World Modeling.*Condition Space)",
            r"LiLo.VLA",
            r"SEF.MAP",
        ],
        "3d": [
            r"(?:Visual Geometry Priors.*Gaussian|Sparse Gaussian Occupancy)",
            r"(?:Cryo.EM|Protein.*Cryo)",
            r"UniHand",
        ],
    }

    for cat, papers in ground_truth.items():
        section_text = sections.get(cat, "")
        correct = sum(1 for p in papers if re.search(p, section_text, re.IGNORECASE))
        scores[f"classify_{cat}"] = round(correct / len(papers), 2)

    # --- Multimodal metadata audit ---
    metadata_section = extract_section(
        content,
        r"^###\s+Multimodal Paper Metadata Audit\s*$",
    )
    metadata_section_exists = bool(metadata_section.strip())

    metadata_ground_truth = {
        "metadata_simpleocr": {
            "paper": r"SimpleOCR",
            "authors": [
                r"Yibo\s+Peng",
                r"Peng\s+Xia",
                r"Ding\s+Zhong",
                r"Kaide\s+Zeng",
                r"Siwei\s+Han",
                r"Yiyang\s+Zhou",
                r"Jiaqi\s+Liu",
                r"Ruiyi\s+Zhang",
                r"Huaxiu\s+Yao",
            ],
            "has_appendix": True,
            "main_figures": 5,
            "appendix_figures": 0,
            "main_tables": 4,
            "appendix_tables": 4,
            "total_figures": 5,
            "total_tables": 8,
            "appendix_evidence": r"Appendix A Dataset Details",
        },
        "metadata_exploring_multimodal_lmms": {
            "paper": r"Exploring Multimodal LMMs",
            "authors": [
                r"Giuseppe\s+Lando",
                r"Rosario\s+Forte",
                r"Antonino\s+Furnari",
            ],
            "has_appendix": False,
            "main_figures": 3,
            "appendix_figures": 0,
            "main_tables": 5,
            "appendix_tables": 0,
            "total_figures": 3,
            "total_tables": 5,
            "appendix_evidence": r"None",
        },
        "metadata_nolan": {
            "paper": r"NoLan",
            "authors": [
                r"Lingfeng\s+Ren",
                r"Weihao\s+Yu",
                r"Runpeng\s+Yu",
                r"Xinchao\s+Wang",
            ],
            "has_appendix": True,
            "main_figures": 4,
            "appendix_figures": 2,
            "main_tables": 5,
            "appendix_tables": 17,
            "total_figures": 6,
            "total_tables": 22,
            "appendix_evidence": r"Appendix A Appendix",
        },
        "metadata_weavetime": {
            "paper": r"WeaveTime",
            "authors": [
                r"Yulin\s+Zhang",
                r"Cheng\s+Shi",
                r"Sibei\s+Yang",
            ],
            "has_appendix": False,
            "main_figures": 7,
            "appendix_figures": 0,
            "main_tables": 5,
            "appendix_tables": 0,
            "total_figures": 7,
            "total_tables": 5,
            "appendix_evidence": r"None",
        },
        "metadata_global_local_dual_perception": {
            "paper": r"Global.Local Dual Perception",
            "authors": [
                r"Junxin\s+Lu",
                r"Tengfei\s+Song",
                r"Zhanglin\s+Wu",
                r"Pengfei\s+Li",
                r"Xiaowei\s+Liang",
                r"Hui\s+Yang",
                r"Kun\s+Chen",
                r"Ning\s+Xie",
                r"Yunfei\s+Lu",
                r"Jing\s+Zhao",
                r"Shiliang\s+Sun",
                r"Daimeng\s+Wei",
            ],
            "has_appendix": False,
            "main_figures": 7,
            "appendix_figures": 0,
            "main_tables": 4,
            "appendix_tables": 0,
            "total_figures": 7,
            "total_tables": 4,
            "appendix_evidence": r"None",
        },
        "metadata_see_it_say_it_sorted": {
            "paper": r"See It, Say It, Sorted",
            "authors": [
                r"Yongchang\s+Zhang",
                r"Oliver\s+Ma",
                r"Tianyi\s+Liu",
                r"Guangquan\s+Zhou",
                r"Yang\s+Chen",
            ],
            "has_appendix": False,
            "main_figures": 5,
            "appendix_figures": 0,
            "main_tables": 6,
            "appendix_tables": 0,
            "total_figures": 5,
            "total_tables": 6,
            "appendix_evidence": r"None",
        },
        "metadata_dynamicgtr": {
            "paper": r"DynamicGTR",
            "authors": [
                r"Yanbin\s+Wei",
                r"Jiangyue\s+Yan",
                r"Chun\s+Kang",
                r"Yang\s+Chen",
                r"Hua\s+Liu",
                r"James\s+Kwok",
                r"Yu\s+Zhang",
            ],
            "has_appendix": True,
            "main_figures": 3,
            "appendix_figures": 1,
            "main_tables": 8,
            "appendix_tables": 11,
            "total_figures": 4,
            "total_tables": 19,
            "appendix_evidence": r"A\.?\s*GTR Generation",
        },
        "metadata_dynamic_multimodal_activation_steering": {
            "paper": r"Dynamic Multimodal Activation Steering",
            "authors": [
                r"Jianghao\s+Yin",
                r"Qin\s+Chen",
                r"Kedi\s+Chen",
                r"Jie\s+Zhou",
                r"Xingjiao\s+Wu",
                r"Liang\s+He",
            ],
            "has_appendix": True,
            "main_figures": 4,
            "appendix_figures": 2,
            "main_tables": 5,
            "appendix_tables": 9,
            "total_figures": 6,
            "total_tables": 14,
            "appendix_evidence": r"Appendix A Appendix",
        },
        "metadata_dr_seg": {
            "paper": r"Dr\.\s*Seg",
            "authors": [
                r"Haoxiang\s+Sun",
                r"Tao\s+Wang",
                r"Chenwei\s+Tang",
                r"Li\s+Yuan",
                r"Jiancheng\s+Lv",
            ],
            "has_appendix": True,
            "main_figures": 7,
            "appendix_figures": 7,
            "main_tables": 6,
            "appendix_tables": 6,
            "total_figures": 14,
            "total_tables": 12,
            "appendix_evidence": r"Appendix A More Experiment Details and Ablations",
        },
        "metadata_cccaption": {
            "paper": r"CCCaption",
            "authors": [
                r"Zhijiang\s+Tang",
                r"Linhua\s+Wang",
                r"Jiaxin\s+Qi",
                r"Weihao\s+Jiang",
                r"Peng\s+Hou",
                r"Anxiang\s+Zeng",
                r"Jianqiang\s+Huang",
            ],
            "has_appendix": False,
            "main_figures": 5,
            "appendix_figures": 0,
            "main_tables": 5,
            "appendix_tables": 0,
            "total_figures": 5,
            "total_tables": 5,
            "appendix_evidence": r"None",
        },
    }

    appendix_flags_ok = True
    split_consistency_ok = True
    if metadata_section_exists:
        for score_key, spec in metadata_ground_truth.items():
            row_match = re.search(
                rf"^\|[^\n]*{spec['paper']}[^\n]*\|\s*$",
                metadata_section,
                re.IGNORECASE | re.MULTILINE,
            )
            if not row_match:
                appendix_flags_ok = False
                split_consistency_ok = False
                scores[score_key] = 0.0
                continue

            row_text = row_match.group(0)
            author_ok = all(re.search(author_pat, row_text, re.IGNORECASE) for author_pat in spec["authors"])
            appendix_ok = bool(
                re.search(
                    r"\|\s*(?:yes|true)\s*\|",
                    row_text,
                    re.IGNORECASE,
                )
            ) if spec["has_appendix"] else bool(
                re.search(
                    r"\|\s*(?:no|false)\s*\|",
                    row_text,
                    re.IGNORECASE,
                )
            )
            main_figures_ok = bool(re.search(rf"\|\s*{spec['main_figures']}\s*\|", row_text))
            appendix_figures_ok = bool(re.search(rf"\|\s*{spec['appendix_figures']}\s*\|", row_text))
            main_tables_ok = bool(re.search(rf"\|\s*{spec['main_tables']}\s*\|", row_text))
            appendix_tables_ok = bool(re.search(rf"\|\s*{spec['appendix_tables']}\s*\|", row_text))
            total_figures_ok = bool(re.search(rf"\|\s*{spec['total_figures']}\s*\|", row_text))
            total_tables_ok = bool(re.search(rf"\|\s*{spec['total_tables']}\s*\|", row_text))
            evidence_ok = bool(re.search(spec["appendix_evidence"], row_text, re.IGNORECASE))

            cells = [c.strip() for c in row_text.strip().strip("|").split("|")]
            if len(cells) >= 10:
                try:
                    row_main_figures = int(cells[4])
                    row_appendix_figures = int(cells[5])
                    row_main_tables = int(cells[6])
                    row_appendix_tables = int(cells[7])
                    row_total_figures = int(cells[8])
                    row_total_tables = int(cells[9])
                    split_figures_consistent = row_main_figures + row_appendix_figures == row_total_figures
                    split_tables_consistent = row_main_tables + row_appendix_tables == row_total_tables
                except ValueError:
                    split_figures_consistent = False
                    split_tables_consistent = False
            else:
                split_figures_consistent = False
                split_tables_consistent = False
            appendix_flags_ok = appendix_flags_ok and appendix_ok
            split_consistency_ok = split_consistency_ok and split_figures_consistent and split_tables_consistent
            scores[score_key] = 1.0 if (
                author_ok
                and main_figures_ok
                and appendix_figures_ok
                and main_tables_ok
                and appendix_tables_ok
                and total_figures_ok
                and total_tables_ok
                and evidence_ok
            ) else 0.0
    else:
        for score_key in metadata_ground_truth:
            scores[score_key] = 0.0
        appendix_flags_ok = False
        split_consistency_ok = False

    scores["metadata_appendix_flags"] = 1.0 if (metadata_section_exists and appendix_flags_ok) else 0.0
    scores["metadata_split_consistency"] = 1.0 if (metadata_section_exists and split_consistency_ok) else 0.0

    # --- Interest selection ---
    # "### Papers of Interest" must contain exactly one recommended paper, and
    # that paper should be CCCaption.
    # We first scope to "## Personalized Recommendations", then search inside it.
    recommendations_section = extract_section(
        content,
        r"^#{2,4}\s+[^\n]*(?:[Pp]ersonali|[Rr]ecommend)[^\n]*$",
    )
    interest_section = extract_section(
        recommendations_section if recommendations_section.strip() else content,
        r"^#{2,4}\s+[^\n]*[Pp]apers?\s+[Oo]f\s+[Ii]nterest[^\n]*$",
    )
    recommended_papers = re.findall(
        r"(?m)^(?:-\s+\*\*.+?\*\*|\d+\.\s+\*\*.+?\*\*|-\s+.+?—.+|\d+\.\s+.+?—.+)$",
        interest_section,
    )
    scores["interest_count"] = 1.0 if len(recommended_papers) == 1 else 0.0
    scores["interest_selection"] = (
        1.0
        if (
            scores["interest_count"] == 1.0
            and re.search(r"CCCaption", interest_section, re.IGNORECASE)
        )
        else 0.0
    )

    # --- Benchmark extraction sub-items ---
    benchmark_source = recommendations_section if recommendations_section.strip() else content
    benchmark_section = extract_section(
        benchmark_source,
        r"^#{2,4}\s+[^\n]*(?:[Bb]enchmark|[Cc]omparison|[Pp]rism)[^\n]*$",
    )

    # Sub-item 1: benchmark table contains CharXiv
    has_charxiv = bool(
        re.search(r"^\|[^\n]*CharXiv[^\n]*\|\s*$", benchmark_section, re.IGNORECASE | re.MULTILINE)
    )
    scores["benchmark_charxiv"] = 1.0 if has_charxiv else 0.0

    # Sub-item 2: CapRL-3B average score is 51.07
    scores["benchmark_caprl_avg"] = (
        1.0 if re.search(r"CapRL.{0,10}3B.*?51\.07", benchmark_section, re.IGNORECASE) else 0.0
    )

    # Sub-item 3: CCCaption-2B average score is 52.80
    scores["benchmark_cccaption_avg"] = (
        1.0
        if re.search(r"CCCaption.{0,10}2B.*?52\.80", benchmark_section, re.IGNORECASE)
        else 0.0
    )

    # Sub-item 4: CapRL row has InfoVQA score 55.94
    has_infovqa_header = bool(re.search(r"\|\s*InfoVQA\s*\|", benchmark_section, re.IGNORECASE))
    caprl_row_has_5594 = bool(
        re.search(r"^\|[^\n]*CapRL[^\n]*55\.94[^\n]*\|\s*$", benchmark_section, re.IGNORECASE | re.MULTILINE)
    )
    scores["benchmark_caprl_infovqa"] = 1.0 if (has_infovqa_header and caprl_row_has_5594) else 0.0

    classify_keys = [
        "classify_multimodal",
        "classify_medical",
        "classify_generation",
        "classify_driving",
        "classify_3d",
    ]
    metadata_keys = [
        "metadata_appendix_flags",
        "metadata_split_consistency",
        "metadata_simpleocr",
        "metadata_exploring_multimodal_lmms",
        "metadata_nolan",
        "metadata_weavetime",
        "metadata_global_local_dual_perception",
        "metadata_see_it_say_it_sorted",
        "metadata_dynamicgtr",
        "metadata_dynamic_multimodal_activation_steering",
        "metadata_dr_seg",
        "metadata_cccaption",
    ]
    interest_keys = [
        "interest_count",
        "interest_selection",
        "benchmark_charxiv",
        "benchmark_caprl_avg",
        "benchmark_cccaption_avg",
        "benchmark_caprl_infovqa",
    ]

    scores["classify_score"] = round(
        sum(scores.get(k, 0.0) for k in classify_keys) / len(classify_keys), 4
    )
    scores["metadata_score"] = round(
        (
            sum(scores.get(k, 0.0) for k in metadata_keys) / len(metadata_keys)
            if metadata_section_exists
            else 0.0
        ),
        4,
    )
    scores["interest_score"] = round(
        sum(scores.get(k, 0.0) for k in interest_keys) / len(interest_keys), 4
    )

    scores["overall_score"] = round(
        0.3 * scores["classify_score"]
        + 0.4 * scores["metadata_score"]
        + 0.3 * scores["interest_score"],
        4,
    )

    return scores
