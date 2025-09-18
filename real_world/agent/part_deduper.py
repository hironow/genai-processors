from __future__ import annotations

"""A real‑world PartProcessor example: per‑part content de‑duplication.

Use cases
- Avoid reprocessing identical frames/samples/messages in realtime pipelines
- Reduce cost by hashing content and filtering duplicates early

Behavior
- For each incoming part, compute a stable content hash (text or bytes)
- If the hash was never seen in the recent window: attach `content_hash`
  to metadata and pass the part through
- If the hash was seen: drop the part and emit a status log indicating a dup

This processor operates on individual parts (PartProcessor), so it can be
inserted anywhere in a chain and works for text, images, audio, and video.
"""

import collections
import asyncio
from collections.abc import AsyncIterable
from typing import Optional

from genai_processors import content_api, processor
from loguru import logger
import xxhash


ProcessorPart = content_api.ProcessorPart


class ContentDeduper(processor.PartProcessor):
  def __init__(self, *, window_size: int = 10_000, name: str = "deduper"):
    self._name = name
    self._window_size = window_size
    # LRU set implemented via OrderedDict (value is unused True)
    self._seen = collections.OrderedDict[str, bool]()
    self._lock = asyncio.Lock()

  def match(self, part: ProcessorPart) -> bool:  # Match all parts
    return True

  def _hash_part(self, part: ProcessorPart) -> str:
    data: Optional[bytes]
    if content_api.is_text(part.mimetype):
      data = part.text.encode("utf-8")
    else:
      data = part.bytes
      if data is None:
        # Fallback: hash the structured Part JSON
        data = str(part.part.to_json_dict()).encode("utf-8")
    return xxhash.xxh64(data).hexdigest()

  def _remember(self, h: str) -> None:
    # Bump to most-recent
    self._seen.pop(h, None)
    self._seen[h] = True
    # Enforce window size (remove oldest)
    while len(self._seen) > self._window_size:
      self._seen.popitem(last=False)

  async def call(
      self, part: ProcessorPart
  ) -> AsyncIterable[content_api.ProcessorPart]:
    h = self._hash_part(part)
    logger.debug("{}: incoming part role={} sub={} mime={} hash={}", self._name, part.role, part.substream_name, part.mimetype, h)
    # Ensure atomic check+remember to avoid races across concurrent parts
    async with self._lock:
      is_dup = h in self._seen
      if not is_dup:
        self._remember(h)

    if is_dup:
      # Duplicate: emit a status entry and drop the part
      logger.info("{}: drop duplicate hash={}", self._name, h)
      yield processor.status(f"[deduper] duplicate content hash={h}")
      return
    # First occurrence: annotate and forward
    meta = dict(part.metadata)
    meta["content_hash"] = h
    logger.debug("{}: forward first-occurrence hash={}", self._name, h)
    yield ProcessorPart(part, metadata=meta)
