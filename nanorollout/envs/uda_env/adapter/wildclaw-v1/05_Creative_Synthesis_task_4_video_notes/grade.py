def grade(**kwargs) -> dict:
    """
    Grade the video lecture notes task by checking factual checkpoints via LLM-as-judge.

    Returns:
        Dict mapping criterion names to scores (0.0 to 1.0)
    """
    import os
    import json
    from pathlib import Path

    grading_model = os.environ.get("JUDGE_MODEL", "openai/gpt-5.4")
    workspace = Path("/tmp_workspace/results")
    notes_path = workspace / "notes.md"

    scores = {}
    zero = {f"cp_{i}": 0.0 for i in range(1, 9)}
    zero.update({"checkpoint_avg": 0.0, "overall_score": 0.0})

    # ========== 1. Pre-checks ==========
    if not notes_path.exists() or notes_path.stat().st_size == 0:
        scores.update(zero)
        return scores

    notes_content = notes_path.read_text(encoding="utf-8")
    word_count = len(notes_content.split())

    # ========== 2. Checkpoint evaluation via LLM-as-judge ==========
    checkpoints = {
        "cp_1": "The notes define an LLM as a mathematical function that predicts the next word/token; it outputs a probability distribution over all possible next words, not a single deterministic answer.",
        "cp_2": "The notes describe how chatbots work by prepending a system prompt + appending the user's message + repeatedly predicting the next word; sampling from less likely words makes output more natural and non-deterministic.",
        "cp_3": "The notes explain that model behavior is determined by parameters/weights, large models have hundreds of billions of them, and models are trained on enormous internet text.",
        "cp_4": "The notes describe pre-training: feed all-but-the-last word, compare prediction with the actual last word; backpropagation adjusts parameters; parameters start random (gibberish) and are iteratively refined.",
        "cp_5": "The notes explain RLHF (Reinforcement Learning from Human Feedback): workers flag unhelpful or problematic predictions, and their corrections further change the model's parameters to align preferences.",
        "cp_6": "The notes mention that before 2017, models processed text one word at a time; Google introduced transformers which enable parallelization.",
        "cp_7": "The notes explain that each word is converted into a vector/embedding; the attention mechanism lets vectors communicate and refine meanings based on context.",
        "cp_8": "The notes mention feed-forward networks (MLPs) providing additional capacity to store language patterns; the model's behavior is an emergent phenomenon from parameter tuning, making it hard to explain specific predictions.",
    }

    cp_scores = {}
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=os.environ["OPENROUTER_BASE_URL"])

        checkpoint_list = "\n".join(
            f"- **{key}**: {desc}" for key, desc in checkpoints.items()
        )

        prompt = (
            "You are a STRICT grading assistant. Below are study notes about Large Language Models. "
            "Your job is to verify whether the notes reflect SPECIFIC content from the source video, "
            "not just generic textbook knowledge about LLMs.\n\n"
            "IMPORTANT grading rules:\n"
            "- Each checkpoint contains multiple specific claims. ALL claims must be present for full marks.\n"
            "- Generic/vague statements that happen to overlap with a checkpoint should score 0.3 or below.\n"
            "- If any specific claim within a checkpoint is missing, deduct proportionally.\n"
            "- If information is incorrect or contradicts the checkpoint, score 0.0.\n"
            "- Only give 1.0 if every detail in the checkpoint is clearly and accurately covered.\n\n"
            "=== STUDENT NOTES ===\n"
            f"{notes_content}\n"
            "=== END NOTES ===\n\n"
            "Score each checkpoint from 0.0 to 1.0:\n\n"
            f"{checkpoint_list}\n\n"
            "Also evaluate:\n"
            "- **structure_quality**: Are the notes well-organized with clear headings, "
            "logical flow, and readable formatting? (0.0 to 1.0)\n\n"
            "Respond strictly in JSON format with no other content. Example:\n"
            '{"cp_1": 1.0, "cp_2": 0.5, ..., "cp_8": 0.0, "structure_quality": 0.8}'
        )

        resp = client.chat.completions.create(
            model=grading_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.strip("`").removeprefix("json").strip()
        cp_scores = json.loads(raw)
    except Exception as e:
        scores["llm_error"] = str(e)

    for key in checkpoints:
        scores[key] = round(float(cp_scores.get(key, 0.0)), 4)
    scores["structure_quality"] = round(float(cp_scores.get("structure_quality", 0.0)), 4)

    # ========== 3. Overall score ==========
    checkpoint_avg = sum(scores[f"cp_{i}"] for i in range(1, 9)) / 8.0
    scores["checkpoint_avg"] = round(checkpoint_avg, 4)

    # Length penalty: discount if outside 800-3000 word range
    if word_count < 800:
        length_penalty = max(0.0, word_count / 800)
    elif word_count > 3000:
        length_penalty = max(0.0, 1.0 - (word_count - 3000) / 3000)
    else:
        length_penalty = 1.0

    scores["overall_score"] = round(
        (0.85 * checkpoint_avg + 0.15 * scores["structure_quality"]) * length_penalty, 4
    )

    return scores
