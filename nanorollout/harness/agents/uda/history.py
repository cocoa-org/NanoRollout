"""Message-history transformations applied before each LLM call.

Two independent concerns:

1. **Image pruning** — bound the screenshot count carried in context. Ported
   from ``anthropic-quickstarts/computer-use-best-practices/formatters.py``.
   The interesting one is :class:`StripImagesAtIntervals` — see its docstring
   for why naive "keep last N" is cache-hostile and how the interval scheme
   fixes it.

2. **Conversation-window compaction** — drop oldest iteration blocks once the
   message stream gets long. :class:`SlidingWindowCompactor` keeps the first
   user message (task description) plus the last K iteration blocks
   (user/assistant/tool triplets), drops the middle.

Two message shapes coexist in this codebase:

* **OpenAI Chat-Completions** — used by OpenAILLM and its Kimi/Qwen/GLM
  subclasses. Tool feedback is ``role="tool"`` messages; images are
  ``{"type": "image_url", "image_url": {"url": "data:..."}}`` inside a
  ``user`` message's content list.
* **Anthropic-native** — used by ClaudeLLM. Tool feedback is a ``role="user"``
  message whose content list contains ``{"type": "tool_result", "content":
  [...]}`` blocks; images are ``{"type": "image", "source": {...}}`` blocks,
  either at the top level of a user message or nested inside a
  ``tool_result``'s content.

Both helpers below auto-detect format. The compactor is shape-agnostic since
it works on user-message boundaries which exist in both formats.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


# Placeholder text inserted in place of a pruned image. Keep the exact string
# stable across runs — that's what lets ``StripImagesAtIntervals`` produce
# cache-friendly prefixes.
IMAGE_PLACEHOLDER_TEXT = "[Image Omitted]"


_IMAGE_BLOCK_TYPES = frozenset({"image_url", "image"})


def _image_slots(messages: List[Dict[str, Any]]) -> List[Tuple[list, int]]:
    """Return ``(container, index)`` pairs for every image block inside user
    message content, in document order, so callers can replace them in place.

    Handles both OpenAI (``image_url``) and Anthropic-native (``image``)
    block shapes, including images nested one level deep inside a
    ``tool_result`` block's ``content`` list (the shape ClaudeLLM uses for
    post-action screenshots).

    Tool messages (``role="tool"``) and assistant messages are skipped —
    only user-role multimodal content lists carry images in either format.
    """
    slots: List[Tuple[list, int]] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for i, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype in _IMAGE_BLOCK_TYPES:
                slots.append((content, i))
                continue
            # Anthropic tool_result wrapper: images can live one level deep.
            if btype == "tool_result":
                inner = block.get("content")
                if isinstance(inner, list):
                    for j, sub in enumerate(inner):
                        if isinstance(sub, dict) and sub.get("type") in _IMAGE_BLOCK_TYPES:
                            slots.append((inner, j))
    return slots


def _placeholder_block() -> Dict[str, Any]:
    """Return a fresh placeholder dict (callers mutate in place).

    Same shape regardless of source format — ``{"type": "text", ...}``
    is valid as a content block in both OpenAI and Anthropic schemas.
    """
    return {"type": "text", "text": IMAGE_PLACEHOLDER_TEXT}


class StripOldestImages:
    """Keep only the most recent ``keep`` images.

    Simple but **cache-hostile**: every turn past ``keep`` shifts a different
    old image into the placeholder slot, so the serialized request's byte
    prefix changes each call and the prompt cache misses.

    Use :class:`StripImagesAtIntervals` instead whenever Anthropic prompt
    caching is enabled. This class is kept for non-Claude providers and as a
    fallback.
    """

    def __init__(self, keep: int) -> None:
        if keep < 0:
            raise ValueError(f"keep must be non-negative, got {keep}")
        self.keep = keep

    def __call__(self, messages: List[Dict[str, Any]]) -> None:
        slots = _image_slots(messages)
        for container, i in slots[: max(len(slots) - self.keep, 0)]:
            container[i] = _placeholder_block()


class StripImagesAtIntervals:
    """Cache-friendly image bounding.

    Keeps ``(total_images % interval) + min_images`` images. As new
    screenshots arrive the kept-count steps ``min, min+1, …, min+interval-1,
    min, min+1, …`` so the set of *removed* images (and therefore the
    serialized request prefix) only changes once every ``interval`` turns.

    You pay one cache write every ``interval`` turns instead of every turn,
    which on Claude with ephemeral caching is roughly 10x cheaper because
    cache reads cost 10% of fresh input tokens.
    """

    def __init__(self, min_images: int = 3, interval: int = 8) -> None:
        if min_images < 0 or interval < 1:
            raise ValueError(
                f"min_images>=0 and interval>=1 required, got {min_images=}, {interval=}"
            )
        self.min_images = min_images
        self.interval = interval

    def __call__(self, messages: List[Dict[str, Any]]) -> None:
        slots = _image_slots(messages)
        total = len(slots)
        keep = (total % self.interval) + self.min_images
        if total <= keep:
            return
        for container, i in slots[: total - keep]:
            container[i] = _placeholder_block()


class SlidingWindowCompactor:
    """Keep the first user message + the last ``window`` iteration blocks.

    An "iteration block" starts at each ``role="user"`` message after the
    first. We always preserve ``messages[0]`` (the task description) so the
    model never loses the original ask. Everything between the task message
    and the cutoff is dropped — both intervening assistant turns AND their
    tool messages.

    Why iteration-aligned rather than token-aligned: the OpenAI Chat
    Completions API requires that each ``tool`` message reference an
    ``assistant`` message containing a matching ``tool_call_id``. Dropping a
    tool message without its parent assistant message (or vice versa) is a
    400. Aligning to user-message boundaries keeps assistant+tool pairs
    intact by construction.

    Parameters
    ----------
    window:
        Number of trailing iteration blocks to keep. A value of 0 means
        "keep only the task message + the very latest user message".

    Notes
    -----
    If the conversation hasn't grown past ``window`` iterations yet, this is
    a no-op — so it's safe to call before every API request.
    """

    def __init__(self, window: int = 8) -> None:
        if window < 0:
            raise ValueError(f"window must be non-negative, got {window}")
        self.window = window

    def __call__(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return a new list with the middle iterations dropped.

        Operates out-of-place because callers may want to keep the full
        history elsewhere (e.g. for trajectory logging). Cheap — references
        the same message dicts, just a different list spine.
        """
        if not messages or self.window == 0:
            return list(messages)

        user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        # Need at least: task user (0) + window+1 more user msgs to trigger.
        # len(user_indices) >= window + 2  →  there's an iteration block we
        # can safely drop without touching the task or the last `window`.
        if len(user_indices) <= self.window + 1:
            return list(messages)

        cutoff = user_indices[-self.window]
        return [messages[0]] + messages[cutoff:]


def compact_messages(
    messages: List[Dict[str, Any]],
    *,
    window: int = 8,
    image_pruner: Any = None,
) -> List[Dict[str, Any]]:
    """One-shot helper: apply sliding-window compaction then image pruning.

    ``image_pruner`` is any callable with the same shape as
    :class:`StripImagesAtIntervals` / :class:`StripOldestImages`. Pass
    ``None`` to skip image pruning.

    Returns a fresh list (compaction is out-of-place); the image pruner
    mutates the message dicts in place (replacing image blocks with
    placeholders).
    """
    compacted = SlidingWindowCompactor(window=window)(messages)
    if image_pruner is not None:
        image_pruner(compacted)
    return compacted
