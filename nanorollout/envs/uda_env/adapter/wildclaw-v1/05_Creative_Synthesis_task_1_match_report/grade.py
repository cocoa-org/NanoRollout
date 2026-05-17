def grade(**kwargs) -> dict:
    """
    Grade the match report task (text + video clips).

    Returns:
        Dict mapping criterion names to scores (0.0 to 1.0)
    """
    import os
    import json
    import subprocess
    from pathlib import Path

    grading_model = os.environ.get("JUDGE_MODEL", "openai/gpt-5.4")
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=os.environ["OPENROUTER_BASE_URL"])

    workspace = Path("/tmp_workspace/results")
    scores = {}
    zero = {"text_content_accuracy": 0.0, "video_content_alignment": 0.0, "overall_score": 0.0}

    # ========== 1. Pre-checks ==========
    report_path = workspace / "report.md"
    clips_dir = workspace / "clips"
    events_json_path = workspace / "events.json"

    if not report_path.exists() or report_path.stat().st_size == 0:
        scores.update(zero)
        return scores
    if not clips_dir.exists() or not any(clips_dir.glob("event_*.mp4")):
        scores.update(zero)
        return scores

    report_text = report_path.read_text(encoding="utf-8")
    clip_files = sorted(clips_dir.glob("event_*.mp4"))

    # ========== 2. LLM text grading ==========
    reference_text = (
        "Barcelona 4-1 Betis at half-time\n"
        "Real Betis vs FC Barcelona. In the first half, Antony opened the scoring. "
        "Ferran Torres then scored twice to put Barca ahead. Pedri set up Baldé for a powerful strike. "
        "Pedri assisted again as Ferran completed his hat-trick with a long-range effort.\n\n"
        "Key Events:\n"
        "6th minute: Antony scores, Betis 1-0 Barcelona\n"
        "12th minute: Koundé assists, Ferran Torres taps in to equalize, Barcelona 1-1\n"
        "14th minute: Baldé crosses, Ferran Torres volleys in with an acrobatic finish, Barcelona 2-1\n"
        "32nd minute: Pedri assists, Baldé scores, Barcelona 3-1\n"
        "39th minute: Pedri passes, Ferran Torres scores a deflected long-range shot to complete his hat-trick, Barcelona 4-1\n\n"
        "Highlights:\n"
        "Players shake hands, match about to begin\n"
        "Pre-match ceremony, moment of silence for a deceased legend\n"
        "22nd minute: Altimira fouls Yamal, receives a yellow card\n"
        "26th minute: Baldé's acrobatic shot cleared by Betis defender\n"
        "45+2 minute: Ruibal crosses, Kucho's header goes just wide\n"
        "45+5 minute: Abde's aerial challenge knocks down Koundé"
    )

    llm_grading_prompt = f"""You are an expert football match report reviewer. Compare the following report against the reference and score **text_content_accuracy** (0.0 – 1.0):

- Does the report contain Overview, Key Events, and Highlights sections?
- Does it cover all 5 goal events (6th, 12th, 14th, 32nd, 39th minute)?
- Are the goal times, scorers, assist providers, and score changes correct?
- Is the writing clear and well-organized?

=== Reference Content ===
{reference_text}

=== Report to Evaluate ===
{report_text}

Respond strictly in JSON with no other content:
{{"text_content_accuracy": 0.0}}"""

    try:
        response = client.chat.completions.create(
            model=grading_model,
            messages=[{"role": "user", "content": llm_grading_prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.strip("`").removeprefix("json").strip()
        llm_scores = json.loads(raw)
        scores["text_content_accuracy"] = float(llm_scores.get("text_content_accuracy", 0.0))
    except Exception as e:
        scores["text_content_accuracy"] = 0.0
        scores["llm_error"] = str(e)

    # ========== 4. Video content alignment (Pred-GT Matching) ==========
    gt_events = [
        {"time": 678,  "type": "goal", "desc": "Antony scores"},
        {"time": 985 , "type": "goal", "desc": "Ferran Torres taps in"},
        {"time": 1097, "type": "goal", "desc": "Ferran Torres acrobatic volley"},
        {"time": 2201, "type": "goal", "desc": "Baldé scores"},
        {"time": 2707, "type": "goal", "desc": "Ferran Torres long-range hat-trick goal"},
    ]

    def _parse_timestamp(s):
        """Parse 'MM:SS' or raw seconds into total seconds."""
        s = str(s).strip()
        if ':' in s:
            parts = s.split(':')
            try:
                return int(parts[0]) * 60 + int(parts[1])
            except (ValueError, IndexError):
                return None
        try:
            return int(float(s))
        except (ValueError, TypeError):
            return None

    # Step 1: Read predicted events from events.json
    pred_events = []
    if events_json_path.exists():
        try:
            raw_events = json.loads(events_json_path.read_text(encoding="utf-8"))
            if isinstance(raw_events, list):
                for item in raw_events:
                    if isinstance(item, dict) and "video_timestamp" in item:
                        ts = _parse_timestamp(item["video_timestamp"])
                        if ts is not None:
                            pred_events.append({
                                "time": ts,
                                "type": str(item.get("type", "")),
                                "text": str(item.get("description", "")),
                                "clip_file": str(item.get("clip_file", "")) or None,
                            })
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Step 2: Match predicted events to GT events (±30 seconds tolerance on video timestamp)
    gt_scores = [0.0] * len(gt_events)
    gt_matched_clips = [None] * len(gt_events)
    gt_matched_pred = [None] * len(gt_events)
    used_pred = set()

    for gi, gt in enumerate(gt_events):
        best_pi = None
        best_diff = float("inf")
        for pi, pred in enumerate(pred_events):
            if pi not in used_pred:
                diff = abs(pred["time"] - gt["time"])
                if diff <= 30 and diff < best_diff:
                    best_diff = diff
                    best_pi = pi
        if best_pi is not None:
            used_pred.add(best_pi)
            gt_matched_pred[gi] = pred_events[best_pi]
            if pred_events[best_pi]["clip_file"]:
                clip_path = clips_dir / pred_events[best_pi]["clip_file"]
                if clip_path.exists():
                    gt_matched_clips[gi] = clip_path

    # Step 3: Use LLM to verify matched clips using multiple frames from each clip
    try:
        import base64

        for gi, gt in enumerate(gt_events):
            clip_path = gt_matched_clips[gi]
            pred = gt_matched_pred[gi]
            if clip_path is None or pred is None:
                gt_scores[gi] = 0.0
                continue

            frames_b64 = []
            for fi in range(10):
                frame_path = workspace / f"_tmp_frame_{gi}_{fi}.jpg"
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", str(fi), "-i", str(clip_path),
                     "-vframes", "1", "-q:v", "2", str(frame_path)],
                    capture_output=True, timeout=10,
                )
                if frame_path.exists() and frame_path.stat().st_size > 0:
                    with open(frame_path, "rb") as f:
                        frames_b64.append(base64.b64encode(f.read()).decode())
                frame_path.unlink(missing_ok=True)

            if not frames_b64:
                gt_scores[gi] = 0.0
                continue

            content_parts = [
                {"type": "text", "text": (
                    f"These are frames extracted at 1fps from a football match video clip (the full clip). "
                    f"The clip should correspond to a '{gt['type']}' event.\n"
                    f"The report describes this clip as: '{pred['text'][:200]}'\n\n"
                    f"Please judge:\n"
                    f"1. Do the frames show a scene related to a {gt['type']} in football "
                    f"(e.g., a shot on goal, ball hitting the net, goal replay, etc.)?\n"
                    f"2. Does the clip content generally match the description?\n\n"
                    f"Reply with only a score between 0 and 1. 1 = highly matching, 0 = completely unrelated. "
                    f"Reply with the number only."
                )},
            ]
            for b64 in frames_b64:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })

            resp = client.chat.completions.create(
                model=grading_model,
                messages=[{"role": "user", "content": content_parts}],
                temperature=0,
            )
            try:
                s = float(resp.choices[0].message.content.strip())
                gt_scores[gi] = min(max(s, 0.0), 1.0)
            except ValueError:
                gt_scores[gi] = 0.0
    except Exception:
        pass

    scores["video_content_alignment"] = round(
        sum(gt_scores) / len(gt_scores), 4
    ) if gt_scores else 0.0

    # ========== 5. Overall score ==========
    scores["overall_score"] = round(
        0.30 * scores["text_content_accuracy"]
        + 0.70 * scores["video_content_alignment"], 4
    )

    return scores