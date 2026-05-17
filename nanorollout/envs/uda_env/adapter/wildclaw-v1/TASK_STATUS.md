# wildclaw-v1 task data-source status

Per-task accounting of what assets each task needs at rollout time and
where they come from. The `wildclaw-v1` driver reads `exec/` and `gt/`
from each task directory; this document tracks which tasks currently
have those directories populated.

Last updated: 2026-05-16. Source-of-truth for inputs:
- HuggingFace dataset `internlm/WildClawBench`, path `workspace/`
- WildClawBench repo `script/prepare.sh` (videos via yt-dlp, sam3.pt via
  modelscope, dot_git.tar.gz extraction)

## Summary

| Bucket | Count | Driver behavior |
|---|---|---|
| **`hf-populated`** — `exec/` + `gt/` already on disk | 17 | Runs end-to-end; smoke test verified on task_6. |
| **`prepare-needed`** — needs `script/prepare.sh` (videos / model weights) | 7 | Document the dependency; do NOT auto-run prepare.sh — it does network I/O + multi-GB downloads. |
| **`runtime-fetch`** — agent fetches inputs over the network at rollout time | 14 | `exec/` stays empty. Driver no-ops `setup_workspace`. |
| **`self-contained`** — pure prompt task, no inputs needed | 11 | `exec/` empty. `gt/` may or may not be needed (LLM-judge graders). |
| **`needs-input-no-source`** — prompt references inputs we can't source | 11 | Documented gap — runner will fail loud at first agent action. |
| **Total** | **60** | |

## Per-task table

### `hf-populated` (17 tasks)

Both `exec/` and (where applicable) `gt/` are present on disk after the
HF download.

| Task | exec | gt |
|---|---|---|
| 01_Productivity_Flow_task_2_table_tex_download | 1 file (0.0 MB) | 18 files (0.04 MB) |
| 01_Productivity_Flow_task_3_bibtex | 21 PDFs (148 MB) | 21 .bib (0.02 MB) |
| 01_Productivity_Flow_task_4_2022_conference_papers | 1 file (0.0 MB) | 8 files (0.47 MB) |
| 01_Productivity_Flow_task_5_wikipedia_biography | 1 file (0.0 MB) | 10 files (0.02 MB) |
| 01_Productivity_Flow_task_6_calendar_scheduling | 3 files (0.01 MB) | 3 files (0.01 MB) — pilot |
| 01_Productivity_Flow_task_7_openmmlab_contributors | 1 file (0.0 MB) | 2 files (0.01 MB) |
| 01_Productivity_Flow_task_8_real_image_category | images.tar (31 MB) | 1 file (0.01 MB) |
| 01_Productivity_Flow_task_9_scp_crawl | (none) | 1 file (0.03 MB) |
| 01_Productivity_Flow_task_10_pdf_digest | papers.tar (654 MB) | 1 file (0.02 MB) |
| 02_Code_Intelligence_task_3_jigsaw_puzzle_zh | 15 files (1.47 MB) | 1 file (0.88 MB) |
| 02_Code_Intelligence_task_4_jigsaw_puzzle_medium_zh | 24 files (1.33 MB) | 1 file (0.88 MB) |
| 02_Code_Intelligence_task_5_jigsaw_puzzle_hard_zh | 13 files (0.47 MB) | — |
| 02_Code_Intelligence_task_10_acad_homepage_zh | 2 files (2.45 MB) | — |
| 02_Code_Intelligence_task_11_resume_homepage_zh | 2 files (1.21 MB) | — |
| 02_Code_Intelligence_task_12_connect_the_dots_hard_zh | 1 file (0.17 MB) | 1 file (0.29 MB) |
| 02_Code_Intelligence_task_1_sam3_inference | 206 files (8.3 MB) | 1 file (—) |
| 02_Code_Intelligence_task_2_sam3_debug | 232 files (16 MB) | — |

### `prepare-needed` (7 tasks)

`script/prepare.sh` in the upstream WildClawBench repo downloads these
on top of the HF clone. To unblock them in UDA-Gym, run the upstream
script (or replicate it under `tools/migrate/wildclaw_prepare.sh`).

| Task | Asset | Source |
|---|---|---|
| 05_Creative_Synthesis_task_1_match_report | first_half.mp4 (57 min) | yt-dlp Betis vs Barcelona |
| 05_Creative_Synthesis_task_2_goal_highlights | first_half.mp4 (copy of task_1) | — |
| 05_Creative_Synthesis_task_4_video_notes | video.mp4 | yt-dlp LLM lecture |
| 05_Creative_Synthesis_task_5_product_launch_video_to_json | recording.mp4 | yt-dlp Apple Event |
| 05_Creative_Synthesis_task_11_video_en_to_zh_dub | recording.mp4 (copy of task_5) | — |
| 06_Safety_Alignment_task_2_leaked_api | mm_agents/ with .git | extract dot_git.tar.gz (from HF) |
| 06_Safety_Alignment_task_3_leaked_api_pswd | mm_agents/ with .git | extract dot_git.tar.gz (from HF) |

The sam3 tasks (`task_1_sam3_inference`, `task_2_sam3_debug`) are
listed under `hf-populated` because the HF dataset already covers the
inference scaffold; `prepare.sh` only adds the ~10 GB `sam3.pt` model
weight, which can be fetched at agent runtime or as a separate step.

### `runtime-fetch` (14 tasks)

The agent fetches inputs over the network during rollout (arxiv,
GitHub, Wikipedia, openrouter, etc.). `exec/` stays empty.

`01_Productivity_Flow_task_1_arxiv_digest`,
`04_Search_Retrieval_task_{1,4,5,10,11}_*`,
`02_Code_Intelligence_task_{3,4,5,6}_*`,
`05_Creative_Synthesis_task_{6,8,9}_*`,
`06_Safety_Alignment_task_7_skill_injection`.

### `self-contained` (11 tasks)

Pure-prompt tasks with no input files. `exec/` empty.

`03_Social_Interaction_task_{1,2,3,4,5,6}_*`,
`04_Search_Retrieval_task_{2,3,6,8}_*`,
`06_Safety_Alignment_task_5_risk_os_operation`.

### `needs-input-no-source` (11 tasks)

Prompt references explicit input files but we have no upstream source
for them — neither HF nor `prepare.sh` provides them. These will fail
loud at the first agent action that tries to read the input. To unblock,
we'd need to either synthesize the inputs or contact upstream.

`02_Code_Intelligence_task_7_connect_the_dots_medium_img_zh`,
`02_Code_Intelligence_task_{8,9}_link_a_pix_color_*`,
`04_Search_Retrieval_task_{7,9}_*`,
`05_Creative_Synthesis_task_{3,7,10}_*`,
`06_Safety_Alignment_task_{1,4,6,8,9,10}_*`.

Note: some safety tasks (file_overwrite, malicious_skill,
prompt_injection, etc.) may actually be self-contained — the
classifier flagged them because the prompt mentions writing to
`/tmp_workspace/<file>`. Manual inspection needed before declaring
those broken.

## Refreshing this table

```bash
# From WildClawBench repo root (with hf CLI installed)
hf download internlm/WildClawBench --include "workspace/**" \
  --repo-type dataset --local-dir /tmp/wildclaw-hf

# Then copy exec/ + gt/ into adapter dirs (see tools/migrate/wildclaw_assets.py — TODO)
```
