"""Pre-migrated benchmark task corpora.

Each immediate subdirectory holds one benchmark's tasks rewritten to
run on the unified ``uda-desktop`` container image (see ``README.md``
for the per-task schema). Subdirectory names use hyphens (e.g.
``cocoa-v1``) and therefore aren't importable as Python modules; resolve
them from disk relative to this package's ``__file__``::

    from pathlib import Path
    from nanorollout.envs.uda_env import adapter
    ADAPTER_ROOT = Path(adapter.__file__).parent
    COCOA_V1 = ADAPTER_ROOT / "cocoa-v1"
"""

from pathlib import Path

ADAPTER_ROOT: Path = Path(__file__).resolve().parent
"""Root of the migrated task corpora. Use ``ADAPTER_ROOT / "<benchmark>" / "<task_id>"``."""
