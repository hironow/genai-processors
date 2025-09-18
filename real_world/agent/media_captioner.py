from __future__ import annotations

"""PartProcessor that converts media (image/audio/video) to factual alt-text.

Usage
- Provide a turn-based text model (`caption_model`) that accepts multimodal
  input and outputs TEXT only (e.g., Gemini via GenaiModel with a strong
  system instruction like "Extract facts only; describe concisely as alt-text").

Behavior
- For text parts: passthrough
- For image/audio/video parts: call the caption_model with the single part and
  stream back text outputs only. Non-text model outputs are dropped.
- Optionally keep original media alongside generated text (keep_original).
"""

from collections.abc import AsyncIterable

from genai_processors import content_api, processor, streams
from loguru import logger


ProcessorPart = content_api.ProcessorPart


class MediaCaptioner(processor.PartProcessor):
  def __init__(
      self,
      caption_model: processor.Processor,
      *,
      keep_original: bool = False,
      log_substream: str | None = None,
  ):
    self._model = caption_model
    self._keep_original = keep_original
    self._log_substream = log_substream

  def match(self, part: ProcessorPart) -> bool:
    # Apply to all parts; dispatch in call
    return True

  async def call(
      self, part: ProcessorPart
  ) -> AsyncIterable[ProcessorPart]:
    mime = part.mimetype
    # Passthrough for text
    if content_api.is_text(mime):
      yield part
      return

    # For media, call the model and pass back text only
    logged = False
    logger.info("[caption] processing media mime={} keep_original={} log_sub={}", mime, self._keep_original, self._log_substream)
    async for out in self._model(streams.stream_content([part])):
      if content_api.is_text(out.mimetype):
        # Optionally emit a dedicated caption log once per media part, containing the actual caption text.
        if self._log_substream and not logged:
          logger.debug("[caption] emitting caption log for {}", mime)
          yield ProcessorPart(
              f"[caption] {out.text or ''}",
              role="model",
              substream_name=self._log_substream,
              metadata={
                "original_mimetype": mime,
                "caption_len": len(out.text or ""),
              },
          )
          logged = True
        # Ensure default stream for the alt-text unless the model sets one
        logger.info("[caption] alt-text emitted sub={} len={}", out.substream_name or '', len(out.text or ''))
        # Tag this part so observers can treat it as a generic custom_event.
        meta = dict(out.metadata)
        meta.update({
          "caption_event": True,  # backward-compat flag
          "original_mimetype": mime,
          "source": "media_captioner",
          # New generic event payload. Observers should prefer this key.
          "custom_event": {
            "type": "caption",
            # Optional instructions for injection; consumers may ignore.
            "inject": {"role": "user", "end_of_turn": False},
            # Additional data payload for consumers.
            "data": {"original_mimetype": mime},
          },
        })
        yield ProcessorPart(out, substream_name=out.substream_name or '', metadata=meta)

    if self._keep_original:
      logger.debug("[caption] keeping original media {}", mime)
      yield part
