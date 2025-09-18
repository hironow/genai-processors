from __future__ import annotations

"""Modality-aware metrics logger Processor.

Counts incoming parts by modality (text/audio/image/video/other), logs each
incoming part (lightweight), and emits summary logs:

- Non-streaming mode: once at the end of the input stream.
- Streaming mode: every `log_interval_sec` seconds (default: 60) and on close.

This Processor is a passthrough: it yields the original parts unchanged.
"""

import asyncio
import time
from typing import AsyncIterable

from loguru import logger
import math
import struct
from io import BytesIO
from PIL import Image, ImageDraw
from genai_processors import content_api, processor


ProcessorPart = content_api.ProcessorPart


class ModalityMetricsLogger(processor.Processor):
  def __init__(
      self,
      *,
      streaming: bool = False,
      log_interval_sec: float = 60.0,
      name: str = "modality_metrics",
  ):
    self._streaming = streaming
    self._interval = log_interval_sec
    self._name = name

    # Counters
    self._total = 0
    self._text = 0
    self._audio = 0
    self._image = 0
    self._video = 0
    self._other = 0

    # Control
    self._stop = asyncio.Event()
    self._periodic_task: asyncio.Task | None = None

  def _classify_and_count(self, part: ProcessorPart) -> str:
    # Treat tool/function messages as 'other' regardless of MIME.
    if part.part.function_call is not None or part.part.function_response is not None:
      self._other += 1
      return "other"

    mime = part.mimetype
    if content_api.is_text(mime):
      self._text += 1
      return "text"
    if content_api.is_audio(mime):
      self._audio += 1
      return "audio"
    if content_api.is_image(mime):
      self._image += 1
      return "image"
    if content_api.is_video(mime):
      self._video += 1
      return "video"
    self._other += 1
    return "other"

  def _snapshot(self) -> dict[str, int]:
    return {
        "total": self._total,
        "text": self._text,
        "audio": self._audio,
        "image": self._image,
        "video": self._video,
        "other": self._other,
    }

  def _log_snapshot(self, label: str) -> None:
    snap = self._snapshot()
    logger.info("{}: {} | totals={} ", self._name, label, snap)

  async def _periodic_logger(self):
    try:
      while True:
        try:
          await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
          break
        except asyncio.TimeoutError:
          # Time tick
          self._log_snapshot("periodic")
    except asyncio.CancelledError:
      pass

  async def call(
      self, content: AsyncIterable[ProcessorPart]
  ) -> AsyncIterable[ProcessorPart]:
    start = time.perf_counter()
    logger.info("{}: start streaming={} interval={}s", self._name, self._streaming, self._interval)
    if self._streaming:
      # Start background periodic logger.
      self._periodic_task = processor.create_task(self._periodic_logger())

    try:
      async for part in content:
        self._total += 1
        modality = self._classify_and_count(part)
        # Lightweight per-part log.
        logger.debug(
            "{}: part role={} sub={} mime={} as={}",
            self._name,
            part.role,
            part.substream_name,
            part.mimetype,
            modality,
        )
        # Emit a per-part log as a reserved 'status' substream in the same
        # modality as the incoming part: text/image/audio/video.
        try:
          log_part = self._make_log_part(modality)
          if log_part is not None:
            # Ensure reserved substream so it won't be fed into the model.
            log_part.substream_name = processor.STATUS_STREAM
            yield log_part
        except Exception as e:  # pylint: disable=broad-except
          logger.debug("{}: failed to emit log part: {}", self._name, e)
        yield part
    finally:
      self._stop.set()
      if self._periodic_task is not None:
        self._periodic_task.cancel()
        try:
          await self._periodic_task
        except Exception:  # pylint: disable=broad-except
          pass

      elapsed = time.perf_counter() - start
      # Log a final snapshot (both modes) for completeness.
      self._log_snapshot(f"final ({elapsed:0.2f}s)")

  # -------- Helpers to build modality-specific log parts --------
  def _make_log_part(self, modality: str) -> ProcessorPart | None:
    if modality == "text":
      return ProcessorPart(
          f"[metrics] text_count={self._text}", role="model", mimetype="text/plain"
      )
    if modality == "image":
      img = self._make_image_log(f"images: {self._image}")
      return ProcessorPart(img, role="model")
    if modality == "audio":
      pcm = self._make_audio_beeps(self._audio)
      return ProcessorPart(pcm, role="model", mimetype="audio/l16;rate=24000")
    if modality == "video":
      # Produce a small binary payload flagged as video for logging purposes.
      payload = (f"metrics_video_count={self._video}").encode("utf-8")
      return ProcessorPart(payload, role="model", mimetype="video/mp4")
    return None

  def _make_image_log(self, text: str) -> Image.Image:
    w, h = 360, 120
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Center-ish text; using default font to avoid external deps.
    draw.text((10, h // 2 - 10), text, fill=(0, 0, 0))
    img.format = "PNG"
    return img

  def _make_audio_beeps(self, count: int, *, sr: int = 24000) -> bytes:
    # Generate `count` short beeps (100ms each) separated by 50ms silence.
    if count <= 0:
      count = 1
    tone_hz = 880.0
    tone_len = int(0.1 * sr)
    gap_len = int(0.05 * sr)
    amplitude = 0.2  # scale <1 to avoid clipping
    frames: list[int] = []
    for _ in range(count):
      for n in range(tone_len):
        sample = int(amplitude * 32767 * math.sin(2 * math.pi * tone_hz * n / sr))
        frames.append(sample)
      frames.extend([0] * gap_len)
    # Pack as 16-bit little endian PCM
    buf = BytesIO()
    for s in frames:
      buf.write(struct.pack('<h', max(-32768, min(32767, s))))
    return buf.getvalue()
