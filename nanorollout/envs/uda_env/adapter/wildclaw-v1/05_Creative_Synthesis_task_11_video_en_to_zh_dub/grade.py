def grade(**kwargs) -> dict:
    import base64
    import json
    import os
    import re
    import subprocess
    import tempfile
    import time
    from openai import OpenAI
    from pathlib import Path

    OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
    GT_DIR = Path("/tmp_workspace/gt")
    GT_FILE = GT_DIR / "ground_truth.json"

    VLM_MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-5.4")
    AUDIO_MODEL = "openai/gpt-4o-mini-audio-preview"
    OPENROUTER_BASE_URL = os.environ["OPENROUTER_BASE_URL"]

    ALL_CRITERIA = [
        "basic_requirements",
        "transcript_accuracy",
        "translation_quality",
        "audio_quality",
        "visual_consistency",
        "overall_score",
    ]


    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

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


    def _get_duration(path):
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                capture_output=True, text=True, timeout=30,
            )
            return float(r.stdout.strip())
        except Exception:
            return None

    GT = json.loads(GT_FILE.read_text())
    GT_TRANSCRIPT = GT["english_transcript"]
    MAX_DURATION = GT["max_duration_seconds"]
    SPEECH_START = GT.get("audio_check_start_seconds", 85)
    AUDIO_CHECK_DURATION = GT.get("audio_check_duration_seconds", 35)
    FRAME_TIMESTAMPS = GT["frame_check_timestamps_seconds"]
    SPEAKER_GENDER = GT.get("speaker_gender", "unknown")

    scores = {}
    workspace = Path("/tmp_workspace/")

    en_file = workspace / "results" / "transcript_en.txt"
    zh_file = workspace / "results" / "transcript_zh.txt"
    output_file = workspace / "results" / "output.mp4"
    source_video = workspace / "recording.mp4"

    # ── 1. Basic requirements (GATING) ────────────────────────────────

    en_text = ""
    if en_file.exists() and en_file.stat().st_size > 0:
        en_text = en_file.read_text(encoding="utf-8", errors="ignore").strip()

    zh_text = ""
    if zh_file.exists() and zh_file.stat().st_size > 0:
        zh_text = zh_file.read_text(encoding="utf-8", errors="ignore").strip()

    video_ok = output_file.exists() and output_file.stat().st_size > 100_000
    dur = _get_duration(output_file) if output_file.exists() else None
    duration_ok = dur is not None and dur <= MAX_DURATION + 1

    basic_checks = {
        "en_transcript": bool(en_text),
        "zh_transcript": bool(zh_text) and bool(re.search(r"[\u4e00-\u9fff]", zh_text)),
        "video_exists": video_ok,
        "duration_ok": duration_ok,
    }

    gate_pass = all(basic_checks.values())
    scores["basic_requirements"] = 1.0 if gate_pass else round(
        sum(basic_checks.values()) / len(basic_checks), 2,
    )

    if not gate_pass:
        scores.update({k: 0.0 for k in ALL_CRITERIA if k not in scores})
        scores["overall_score"] = 0.0
        return scores

    # ── 2. English transcript accuracy (exact match) ─────────────────

    if en_text:
        gt_norm = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", GT_TRANSCRIPT.lower())).strip()
        pred_norm = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", en_text.lower())).strip()
        scores["transcript_accuracy"] = 1.0 if gt_norm == pred_norm else 0.0
    else:
        scores["transcript_accuracy"] = 0.0

    # ── 3. Translation quality (VLM judge, 4-dimension) ──────────────

    if zh_text:
        src = en_text or GT_TRANSCRIPT
        prompt = (
            "You are a professional translation evaluator. "
            "Evaluate the following English-to-Chinese translation on four dimensions, "
            "each scored 0.0-1.0.\n\n"
            f"=== English Source ===\n{src}\n\n"
            f"=== Chinese Translation ===\n{zh_text}\n\n"
            "Dimensions:\n"
            "1. accuracy: Does the Chinese faithfully convey ALL information from the English? "
            "Penalize any omissions, additions, or distortions.\n"
            "2. fluency: Is the Chinese natural, idiomatic, and grammatically correct? "
            "Penalize awkward phrasing, translationese, or unnatural word choices.\n"
            "3. terminology: Are domain-specific terms (tech, brand names, etc.) translated "
            "correctly and consistently?\n"
            "4. style: Does the translation preserve the tone, register, and rhetorical "
            "intent of the original?\n\n"
            "Return ONLY valid JSON:\n"
            '{"accuracy": <float>, "fluency": <float>, "terminology": <float>, "style": <float>}'
        )
        result = _call_vlm([{"role": "user", "content": prompt}], max_tokens=512)
        data = _extract_json(result)
        if data:
            sub = {k: min(1.0, max(0.0, float(data.get(k, 0)))) for k in
                   ["accuracy", "fluency", "terminology", "style"]}
            tw = {"accuracy": 3, "fluency": 3, "terminology": 2, "style": 2}
            scores["translation_quality"] = round(
                sum(sub[k] * tw[k] for k in tw) / sum(tw.values()), 2,
            )
        else:
            scores["translation_quality"] = 0.0
    else:
        scores["translation_quality"] = 0.0

    # ── 4. Audio quality (audio model — original vs dubbed) ──────────

    scores["audio_quality"] = 0.0

    if output_file.exists() and video_ok:
        try:
            def _extract_wav(video_path, start, duration):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    wav_path = f.name
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", str(start),
                     "-i", str(video_path), "-t", str(duration),
                     "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", wav_path],
                    capture_output=True, timeout=60,
                )
                file_size = os.path.getsize(wav_path)
                wav_header = 44
                sample_rate, channels, bits = 16000, 1, 16
                min_bytes = wav_header + int(0.1 * sample_rate * channels * (bits // 8))
                with open(wav_path, "rb") as fh:
                    b64 = base64.b64encode(fh.read()).decode()
                os.unlink(wav_path)
                if file_size < min_bytes:
                    return None
                return b64

            dubbed_b64 = _extract_wav(output_file, SPEECH_START, AUDIO_CHECK_DURATION)
            if dubbed_b64 is None:
                raise Exception("skip")

            has_original = source_video.exists()
            original_b64 = None
            if has_original:
                original_b64 = _extract_wav(source_video, SPEECH_START, AUDIO_CHECK_DURATION)

            content_parts = []
            if has_original and original_b64:
                content_parts.append({"type": "text", "text": "[ORIGINAL ENGLISH AUDIO]:"})
                content_parts.append({"type": "input_audio", "input_audio": {"data": original_b64, "format": "wav"}})
            content_parts.append({"type": "text", "text": "[DUBBED CHINESE AUDIO]:"})
            content_parts.append({"type": "input_audio", "input_audio": {"data": dubbed_b64, "format": "wav"}})

            gender_hint = ""
            if SPEAKER_GENDER != "unknown":
                gender_hint = (
                    f"IMPORTANT: The original speaker is {SPEAKER_GENDER}. "
                    f"The dubbed voice MUST also be {SPEAKER_GENDER}. "
                    "If the gender does not match, speaker_match MUST be ≤ 0.2.\n\n"
                )

            content_parts.append({
                "type": "text",
                "text": (
                    "You are a strict audio quality evaluator for video dubbing. "
                    "You are given the original English audio and the dubbed Chinese audio "
                    "from the same video. Be very critical — only studio-quality dubbing "
                    "deserves >0.85 on any dimension.\n\n"
                    f"{gender_hint}"
                    f'Expected Chinese content: "{zh_text or GT_TRANSCRIPT}"\n\n'
                    "Rate each dimension 0.0-1.0:\n"
                    "1. is_chinese: Is the dubbed speech clearly in Mandarin Chinese?\n"
                    "2. content_match: Does the dubbed content accurately convey the "
                    "meaning of the expected Chinese text?\n"
                    "3. speaker_match: Does the dubbed voice match the original speaker's "
                    "characteristics? Check: (a) gender match — this is critical, a gender "
                    "mismatch alone should result in ≤ 0.2; (b) similar vocal energy and "
                    "tone; (c) similar speaking pace and style.\n"
                    "4. naturalness: Is the dubbed speech natural-sounding with proper "
                    "prosody, intonation, and rhythm? Heavily penalize robotic TTS, "
                    "unnatural pauses, monotone delivery, or mispronunciations.\n"
                    "5. audio_clarity: Is the audio clean without artifacts, distortion, "
                    "clipping, or background noise?\n\n"
                    "Return ONLY valid JSON:\n"
                    '{"is_chinese": <float>, "content_match": <float>, '
                    '"speaker_match": <float>, "naturalness": <float>, '
                    '"audio_clarity": <float>}'
                ),
            })

            result = _call_vlm(
                [{"role": "user", "content": content_parts}], model=AUDIO_MODEL, max_tokens=512,
            )
            data = _extract_json(result)
            if data:
                aq_keys = ["is_chinese", "content_match", "speaker_match", "naturalness", "audio_clarity"]
                aq_sub = {k: min(1.0, max(0.0, float(data.get(k, 0)))) for k in aq_keys}
                aq_w = {"is_chinese": 1, "content_match": 2, "speaker_match": 3,
                        "naturalness": 2, "audio_clarity": 1}
                scores["audio_quality"] = round(
                    sum(aq_sub[k] * aq_w[k] for k in aq_w) / sum(aq_w.values()), 2,
                )
        except Exception:
            pass

    # ── 5. Visual consistency (frame comparison: MSE + SSIM) ─────────

    scores["visual_consistency"] = 0.0

    if output_file.exists() and source_video.exists():
        try:
            from PIL import Image
            import numpy as np

            def _extract_frame(video, ts, out):
                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-ss", str(ts), "-i", str(video),
                         "-frames:v", "1", "-q:v", "2", out],
                        capture_output=True, timeout=30,
                    )
                    return os.path.exists(out) and os.path.getsize(out) > 0
                except Exception:
                    return False

            def _ssim_channel(a, b, C1=6.5025, C2=58.5225):
                mu_a = np.mean(a)
                mu_b = np.mean(b)
                sig_a2 = np.var(a)
                sig_b2 = np.var(b)
                sig_ab = np.mean((a - mu_a) * (b - mu_b))
                num = (2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)
                den = (mu_a ** 2 + mu_b ** 2 + C1) * (sig_a2 + sig_b2 + C2)
                return num / den

            def _ssim_rgb(a, b):
                return np.mean([_ssim_channel(a[:, :, c], b[:, :, c]) for c in range(3)])

            sims = []
            with tempfile.TemporaryDirectory() as td:
                for ts in FRAME_TIMESTAMPS:
                    sf = os.path.join(td, f"src_{ts}.jpg")
                    of = os.path.join(td, f"out_{ts}.jpg")
                    if _extract_frame(source_video, ts, sf) and _extract_frame(output_file, ts, of):
                        a = np.array(Image.open(sf).convert("RGB").resize((320, 240)), dtype=float)
                        b = np.array(Image.open(of).convert("RGB").resize((320, 240)), dtype=float)
                        mse = np.mean((a - b) ** 2)
                        mse_sim = max(0.0, 1.0 - mse / (255.0 ** 2))
                        ssim_val = max(0.0, _ssim_rgb(a, b))
                        combined = 0.4 * mse_sim + 0.6 * ssim_val
                        sims.append(combined)
            if sims:
                scores["visual_consistency"] = round(sum(sims) / len(sims), 2)
        except ImportError:
            pass

    # ── Overall score (weighted, basic_requirements excluded) ────────

    w = {"transcript_accuracy": 2, "translation_quality": 1, "audio_quality": 3, "visual_consistency": 1}
    total_w = sum(w.values())
    scores["overall_score"] = round(
        sum(scores.get(k, 0.0) * w.get(k, 1) for k in w) / total_w, 4,
    )

    return scores
