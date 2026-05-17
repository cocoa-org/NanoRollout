def grade(**kwargs) -> dict:
    import base64
    import json
    import os
    import re
    import time
    from openai import OpenAI
    from pathlib import Path

    OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
    VLM_MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-5.4")
    OPENROUTER_BASE_URL = os.environ["OPENROUTER_BASE_URL"]

    ALL_CRITERIA = [
        "responsive_design",
        "content_completeness",
        "visual_quality",
        "overall_score",
    ]


    def _call_vlm(messages, model=None, max_tokens=2048, retries=2):
        if model is None:
            model = VLM_MODEL
        client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
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
                print(f"  [VLM call attempt {attempt + 1} failed: {e}]")
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

    workspace = Path("/tmp_workspace/results")
    scores = {}

    html_file = workspace / "index.html"
    if not html_file.exists() or html_file.stat().st_size < 1000:
        print(f"[FAIL] index.html not found or too small")
        return {k: 0.0 for k in ALL_CRITERIA}

    html_content = html_file.read_text(encoding="utf-8", errors="ignore")
    print(f"\n=== HTML Info ===")
    print(f"  Size: {len(html_content):,} chars")

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")

    all_css = ""
    for style_tag in soup.find_all("style"):
        all_css += (style_tag.string or "") + "\n"
    combined_source = html_content + "\n" + all_css

    # ── Gating Requirements ───────────────────────────────────────────

    screenshot_path = workspace / "screenshot.png"

    checks = {}
    checks["project_name"] = "vlmevalkit" in html_content.lower()

    links = soup.find_all("a", href=True)
    checks["github_link"] = any("github.com" in a.get("href", "") for a in links)
    checks["navigation"] = bool(soup.find("nav")) or bool(soup.find(class_=re.compile(r"nav", re.I)))

    img_tags = soup.find_all("img")
    svg_tags = [s for s in soup.find_all("svg") if len(s.find_all()) >= 2]
    bg_images = re.findall(r"background(?:-image)?\s*:[^;]*url\s*\(", combined_source)
    total_images = len(img_tags) + len(svg_tags) + len(bg_images)
    checks["images_5+"] = total_images >= 5

    sections = soup.find_all("section")
    h_tags = soup.find_all(["h1", "h2"])
    section_count = max(len(sections), len(h_tags))
    checks["sections_4+"] = section_count >= 4
    
    checks["screenshot_exists"] = screenshot_path.exists() and screenshot_path.stat().st_size > 1000

    print(f"\n=== Gating Checks ===")
    for k, v in checks.items():
        print(f"  {k}: {'OK' if v else 'FAIL'}")
    print(f"  Images: {total_images}, Sections: {section_count}")

    if not all(checks.values()):
        print("\n[FAIL] Gating condition not met")
        return {k: 0.0 for k in ALL_CRITERIA}

    # ── Responsive Design ───────────────────────────────────────────
    has_viewport = bool(soup.find("meta", attrs={"name": "viewport"}))
    has_media = bool(re.search(r"@media", combined_source))
    scores["responsive_design"] = 1.0 if (has_viewport and has_media) else 0.0
    print(f"\n=== Responsive Design: {scores['responsive_design']} ===")

    # ── Content completeness ─────────────────────────────────────────

    print(f"\n=== Content Completeness (VLM Judge) ===")
    text_content = soup.get_text(separator="\n", strip=True)[:5000]
    prompt = (
        "Evaluate a VLMEvalKit homepage. Rate content completeness 0.0-1.0:\n"
        "1. Project introduction\n2. Key features\n"
        "3. Supported models/benchmarks\n4. Quick start\n5. Citation/community\n\n"
        f"=== Text ===\n{text_content}\n\n"
        "Return ONLY valid JSON:\n"
        '{"content_completeness": <float>}'
    )
    result = _call_vlm([{"role": "user", "content": prompt}], max_tokens=512)
    data = _extract_json(result)
    scores["content_completeness"] = round(
        min(1.0, max(0.0, float(data.get("content_completeness", 0)))) if data else 0.0, 2,
    )
    print(f"  content_completeness: {scores['content_completeness']}")

    # ── Visual quality ───────────────────────────────────────────────

    print(f"\n=== Visual Quality (VLM Judge) ===")
    design_score = 0.0

    if screenshot_path.exists() and screenshot_path.stat().st_size > 1000:
        print(f"  Screenshot found: {screenshot_path} ({screenshot_path.stat().st_size:,} bytes)")
        try:
            with open(screenshot_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            vlm_content = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {
                    "type": "text",
                    "text": (
                        "Rate this project homepage design 0.0-1.0:\n"
                        "Evaluate color harmony, typography, layout, visual elements, and polish.\n"
                        "Be strict: 1.0=Apple-level professional, 0.8=great, 0.6=functional but basic, 0.4=poor design, 0.0=broken.\n\n"
                        "Return ONLY valid JSON:\n"
                        '{"design_quality": <float>}'
                    ),
                },
            ]
            vlm_result = _call_vlm([{"role": "user", "content": vlm_content}], max_tokens=512)
            vlm_data = _extract_json(vlm_result)
            if vlm_data:
                design_score = min(1.0, max(0.0, float(vlm_data.get("design_quality", 0))))
        except Exception as e:
            print(f"  [Screenshot VLM error: {e}]")
    else:
        print(f"  [Screenshot not found at {screenshot_path}]")

    if design_score == 0.0:
        print("  [Falling back to source analysis]")
        source_excerpt = combined_source[:8000]
        fb_prompt = (
            "Rate this HTML/CSS homepage design quality 0.0-1.0 from source:\n"
            f"=== Source ===\n{source_excerpt}\n\n"
            "Evaluate color harmony, typography, layout, visual elements, and polish.\n"
            "Be strict: 1.0=Apple-level professional, 0.8=great, 0.6=functional but basic, 0.4=poor design, 0.0=broken.\n\n"
            "Return ONLY valid JSON:\n"
            '{"design_quality": <float>}'
        )
        fb_result = _call_vlm([{"role": "user", "content": fb_prompt}], max_tokens=512)
        fb_data = _extract_json(fb_result)
        if fb_data:
            design_score = min(1.0, max(0.0, float(fb_data.get("design_quality", 0))))

    scores["visual_quality"] = round(design_score, 2)
    print(f"  visual_quality: {scores['visual_quality']}")

    w = {"responsive_design": 1, "content_completeness": 1, "visual_quality": 2}
    total_w = sum(w.values())
    scores["overall_score"] = round(
        sum(scores.get(k, 0.0) * w.get(k, 1) for k in w) / total_w, 4,
    )

    return scores
