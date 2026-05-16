#!/usr/bin/env bash
# Fetch WildClawBench v1 task assets (exec/ + gt/) from HuggingFace.
#
# The task SCHEMA (meta.json, task.yaml, grade.py, etc.) lives in this
# repo. The actual INPUT data (PDFs, tarballs, calendar files, ...) and
# GROUND TRUTH (per-task expected outputs) live on the HF dataset
# ``internlm/WildClawBench`` under the ``workspace/`` subtree. They're
# .gitignored here because they'd add ~850 MB.
#
# Layout on HF:    workspace/<category>/<task_short_id>/{exec,gt}/...
# Layout on disk:  <repo>/.../adapter/wildclaw-v1/<full_task_id>/{exec,gt}/...
# Where full_task_id = "<category>_<task_short_id>".
#
# Usage:
#   bash _fetch_assets.sh                # all 17 tasks that have HF assets
#   bash _fetch_assets.sh task_6         # specific short id
#
# Requires: pip install "huggingface_hub[cli]"

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TMP_DOWNLOAD="${TMP_DOWNLOAD:-/tmp/wildclaw-hf}"

mkdir -p "$TMP_DOWNLOAD"
echo "Downloading WildClawBench workspace from HF → $TMP_DOWNLOAD"
hf download internlm/WildClawBench \
    --include "workspace/**" \
    --repo-type dataset \
    --local-dir "$TMP_DOWNLOAD"

echo
echo "Copying exec/ + gt/ into adapter dirs..."
python3 - <<PY
import shutil
from pathlib import Path

SRC = Path("$TMP_DOWNLOAD") / "workspace"
DST = Path("$HERE")

moved = 0
for cat_dir in sorted(SRC.iterdir()):
    if not cat_dir.is_dir(): continue
    for task_dir in sorted(cat_dir.iterdir()):
        if not task_dir.is_dir(): continue
        full_id = f"{cat_dir.name}_{task_dir.name}"
        adapter_task = DST / full_id
        if not adapter_task.is_dir():
            print(f"  SKIP {full_id}: no adapter dir")
            continue
        for sub in ("exec", "gt"):
            src_sub = task_dir / sub
            if src_sub.is_dir():
                dst_sub = adapter_task / sub
                if dst_sub.exists():
                    shutil.rmtree(dst_sub)
                shutil.copytree(src_sub, dst_sub)
                n = sum(1 for _ in dst_sub.rglob("*") if _.is_file())
                sz_mb = sum(p.stat().st_size for p in dst_sub.rglob("*") if p.is_file()) / 1024 / 1024
                print(f"  {full_id}/{sub:4s}  {n:4d} files  {sz_mb:8.2f} MB")
                moved += 1

print(f"\n{moved} dir(s) populated. {len(list(DST.iterdir()))} task dir(s) total in adapter.")
PY

echo
echo "Done. Remaining task assets (videos, sam3.pt) require WildClawBench's script/prepare.sh."
