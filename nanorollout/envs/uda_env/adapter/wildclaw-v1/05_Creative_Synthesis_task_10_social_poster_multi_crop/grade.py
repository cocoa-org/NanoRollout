def grade(**kwargs) -> dict:
    import base64
    import json
    import os
    import re
    import time
    from openai import OpenAI
    from pathlib import Path

    OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
    GT_DIR = Path("/tmp_workspace/gt")
    VLM_MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-5.4")
    OPENROUTER_BASE_URL = os.environ["OPENROUTER_BASE_URL"]

    ALL_CRITERIA = [
        "basic_requirements",
        "subject_preserved",
        "visual_quality",
        "overall_score",
    ]


    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

    def _call_vlm(messages, model=None, max_tokens=1024, retries=2):
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


    def _read_image_b64(path):
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except Exception:
            return None


    def _mime_for(path):
        ext = str(path).rsplit(".", 1)[-1].lower()
        return {"png": "image/png", "jpg": "image/jpeg",
                "jpeg": "image/jpeg", "webp": "image/webp"}.get(ext, "image/png")

    workspace = Path("/tmp_workspace/results")
    scores = {}

    EXPECTED = {
        "crop_ins_square.png":   {"ratio": 1.0,    "tol": 0.02},
        "crop_tiktok.png":       {"ratio": 9 / 16, "tol": 0.02},
        "crop_ins_portrait.png": {"ratio": 4 / 5,  "tol": 0.02},
    }

    original_poster = GT_DIR / "poster_original.png"
    if not original_poster.exists():
        original_poster = Path("/tmp_workspace/poster_original.png")

    from PIL import Image

    found = {}
    ratio_ok = 0
    for fname, spec in EXPECTED.items():
        base = fname.rsplit(".", 1)[0]
        for ext in ("png", "jpg", "jpeg", "webp"):
            p = workspace / f"{base}.{ext}"
            if p.exists() and p.stat().st_size > 1000:
                found[fname] = p
                try:
                    w, h = Image.open(p).size
                    actual_ratio = w / h
                    if abs(actual_ratio - spec["ratio"]) <= spec["tol"]:
                        ratio_ok += 1
                        print(f"  {fname}: {w}x{h} ratio={actual_ratio:.4f} target={spec['ratio']:.4f} -> OK")
                    else:
                        print(f"  {fname}: {w}x{h} ratio={actual_ratio:.4f} target={spec['ratio']:.4f} -> FAIL")
                except Exception as e:
                    print(f"  {fname}: [read error: {e}]")
                break

    checks = {
        "files_found": len(found) == 3,
        "ratios_correct": ratio_ok == 3,
    }

    gate_pass = all(checks.values())
    scores["basic_requirements"] = 1.0 if gate_pass else round(
        sum(checks.values()) / len(checks), 2,
    )
    print(f"\n=== Basic Requirements (gating): {scores['basic_requirements']} ===")
    print(f"  Files found: {len(found)}/3 -> {'OK' if checks['files_found'] else 'FAIL'}")
    print(f"  Ratios correct: {ratio_ok}/3 -> {'OK' if checks['ratios_correct'] else 'FAIL'}")
    print(f"  => gate={'PASS' if gate_pass else 'FAIL'}")

    if not gate_pass:
        scores.update({k: 0.0 for k in ALL_CRITERIA if k not in scores})
        scores["overall_score"] = 0.0
        print("  *** GATING FAILED — all subsequent scores set to 0 ***")
        return scores

    # ── Subject preserved (VLM) ──────────────────────────────────────

    print(f"\n=== Subject Preserved (VLM Judge) ===")
    original_b64 = _read_image_b64(original_poster) if original_poster.exists() else None
    if original_b64 is None:
        print(f"  [WARN] Original poster not found at {original_poster}")

    subject_scores = []
    for fname in EXPECTED:
        if fname not in found or original_b64 is None:
            subject_scores.append(0.0)
            continue
        b64 = _read_image_b64(found[fname])
        if b64 is None:
            subject_scores.append(0.0)
            continue

        content = [
            {"type": "text", "text": "Image 1 is the original poster. Image 2 is a cropped output."},
            {"type": "image_url", "image_url": {"url": f"data:{_mime_for(original_poster)};base64,{original_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:{_mime_for(found[fname])};base64,{b64}"}},
            {
                "type": "text",
                "text": (
                    "Is the primary visual subject preserved in the crop?\n"
                    "1.0=fully preserved, 0.7=mostly, 0.3=partially, 0.0=lost\n\n"
                    "Return ONLY valid JSON:\n"
                    '{"subject_score": <float>}'
                ),
            },
        ]
        result = _call_vlm([{"role": "user", "content": content}])
        data = _extract_json(result)
        sc = min(1.0, max(0.0, float(data.get("subject_score", 0)))) if data else 0.0
        subject_scores.append(sc)
        print(f"  {fname}: {sc:.2f}")

    scores["subject_preserved"] = round(sum(subject_scores) / 3, 2)

    # ── Visual quality (VLM) ─────────────────────────────────────────

    print(f"\n=== Visual Quality (VLM Judge) ===")
    aesthetic_scores = []
    for fname in EXPECTED:
        if fname not in found:
            aesthetic_scores.append(0.0)
            continue
        b64 = _read_image_b64(found[fname])
        if b64 is None:
            aesthetic_scores.append(0.0)
            continue

        content = [
            {"type": "image_url", "image_url": {"url": f"data:{_mime_for(found[fname])};base64,{b64}"}},
            {
                "type": "text",
                "text": (
                    "Rate this cropped social media poster 0.0-1.0:\n"
                    "Composition, edge quality, professional appearance.\n"
                    "1.0=excellent, 0.7=good, 0.4=mediocre, 0.0=poor\n\n"
                    "Return ONLY valid JSON:\n"
                    '{"aesthetic_score": <float>}'
                ),
            },
        ]
        result = _call_vlm([{"role": "user", "content": content}])
        data = _extract_json(result)
        sc = min(1.0, max(0.0, float(data.get("aesthetic_score", 0)))) if data else 0.0
        aesthetic_scores.append(sc)
        print(f"  {fname}: {sc:.2f}")

    scores["visual_quality"] = round(sum(aesthetic_scores) / 3, 2)

    # ── Overall (basic_requirements excluded — it's a gate) ──

    w = {"subject_preserved": 1, "visual_quality": 2}
    total_w = sum(w.values())
    scores["overall_score"] = round(
        sum(scores.get(k, 0.0) * w.get(k, 1) for k in w) / total_w, 4,
    )

    return scores
