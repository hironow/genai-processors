from __future__ import annotations

"""PartProcessor that limits media size for images, audio, and video.

Goals
- Prevent overly large payloads from propagating downstream
- Provide predictable upper-bounds on processing time and memory
- Apply best-effort conversion per modality

Behavior
- Text parts are passed through unchanged
- Images: resized to fit within max dimensions and recompressed to target format
  while trying to stay under max_image_bytes
- Audio (PCM streaming: audio/l16;rate=... or audio/l24;rate=...): truncated
  to max_audio_duration_sec
- Video: policy-based handling if larger than max_video_bytes
  * passthrough: leave as-is (default)
  * drop: emit status part and drop the video part
  * truncate: slice bytes to limit (note: truncated container might be invalid)

Notes
- This processor operates on individual parts and can be placed anywhere in a
  chain. It preserves role/substream and augments metadata to indicate changes.
"""

from collections.abc import AsyncIterable
import io
import re
from typing import Literal

from genai_processors import content_api, processor
from PIL import Image
from loguru import logger


ProcessorPart = content_api.ProcessorPart


class MediaSizeLimiter(processor.PartProcessor):
  def __init__(
      self,
      *,
      # Image constraints
      max_image_width: int = 1024,
      max_image_height: int = 1024,
      max_image_bytes: int = 400 * 1024,  # 400KB
      image_format: Literal['JPEG', 'WEBP'] = 'JPEG',
      image_quality: int = 85,
      min_image_quality: int = 50,
      # Audio constraints
      max_audio_duration_sec: float = 10.0,
      default_pcm_rate: int = 24000,
      default_pcm_bytes_per_sample: int = 2,
      # Video constraints
      max_video_bytes: int = 5 * 1024 * 1024,  # 5MB
      video_policy: Literal['passthrough', 'drop', 'truncate'] = 'passthrough',
  ):
    self._max_w = max_image_width
    self._max_h = max_image_height
    self._max_image_bytes = max_image_bytes
    self._img_fmt = image_format
    self._img_q = image_quality
    self._img_q_min = min(image_quality, min_image_quality)

    self._max_audio_sec = max_audio_duration_sec
    self._default_rate = default_pcm_rate
    self._default_bps = default_pcm_bytes_per_sample

    self._max_video_bytes = max_video_bytes
    self._video_policy = video_policy

  def match(self, part: ProcessorPart) -> bool:
    return True  # Apply to all parts; modality dispatch in call().

  async def call(
      self, part: ProcessorPart
  ) -> AsyncIterable[ProcessorPart]:
    mime = part.mimetype
    if content_api.is_text(mime):
      yield part
      return

    if content_api.is_image(mime):
      out = self._limit_image(part)
      yield out
      return

    if content_api.is_audio(mime):
      out = self._limit_audio_pcm(part)
      yield out
      return

    if content_api.is_video(mime):
      for out in self._limit_video(part):
        yield out
      return

    # Unknown binary types: pass through unchanged.
    yield part

  # ---------------- Image ----------------
  def _limit_image(self, part: ProcessorPart) -> ProcessorPart:
    try:
      img = part.pil_image
    except Exception:
      return part  # Not convertible; pass through.

    orig_w, orig_h = img.size
    resized = False
    # Resize to fit box while maintaining aspect ratio.
    if orig_w > self._max_w or orig_h > self._max_h:
      img = img.copy()
      img.thumbnail((self._max_w, self._max_h))
      resized = True
      logger.info(
          "[media_size] image resized {}x{} -> {}x{}",
          orig_w, orig_h, img.size[0], img.size[1],
      )

    # Ensure mode is compatible with target encoder.
    target_fmt = self._img_fmt
    if target_fmt == 'JPEG':
      # JPEG does not support alpha or certain modes like 'LA'/'RGBA'/'P'.
      # Convert/flatten to an RGB-compatible image.
      mode = img.mode
      try:
        if mode in ('RGBA', 'LA'):
          # Flatten alpha over white background to preserve appearance.
          if mode == 'LA':
            img_rgba = img.convert('RGBA')
          else:
            img_rgba = img
          background = Image.new('RGB', img_rgba.size, (255, 255, 255))
          alpha = img_rgba.split()[-1]
          background.paste(img_rgba.convert('RGB'), mask=alpha)
          img = background
        elif mode == 'P':
          # Paletted image: convert to RGB (handles optional transparency internally).
          img = img.convert('RGB')
        elif mode not in ('RGB', 'L'):
          img = img.convert('RGB')
      except Exception:
        # If conversion fails, pass original part through to avoid breaking the pipeline.
        return part
      if mode != img.mode:
        logger.info("[media_size] image mode {} -> {} for {}", mode, img.mode, target_fmt)

    # Recompress iteratively to meet max bytes.
    q = self._img_q
    buf = io.BytesIO()
    img.save(buf, format=target_fmt, quality=q, optimize=True)
    data = buf.getvalue()
    while len(data) > self._max_image_bytes and q > self._img_q_min:
      q = max(self._img_q_min, q - 10)
      buf = io.BytesIO()
      img.save(buf, format=target_fmt, quality=q, optimize=True)
      data = buf.getvalue()
    logger.info(
        "[media_size] image recompressed fmt={} quality={} size={}B (limit={}B)",
        target_fmt, q, len(data), self._max_image_bytes,
    )

    new = ProcessorPart(
        data,
        mimetype=f"image/{target_fmt.lower()}",
        role=part.role,
        substream_name=part.substream_name,
        metadata=dict(part.metadata),
    )
    new.metadata.update({
        'resized': resized or (orig_w, orig_h) != img.size,
        'original_dimensions': (orig_w, orig_h),
        'new_dimensions': img.size,
        'recompressed': True,
        'image_quality': q,
        'original_mimetype': part.mimetype,
        'new_size_bytes': len(data),
    })
    return new

  # ---------------- Audio (PCM streaming) ----------------
  def _parse_pcm(self, mime: str) -> tuple[int, int]:
    # Parses 'audio/l16;rate=24000' or 'audio/l24;rate=48000'
    m = re.match(r"audio/l(\d+);\s*rate=(\d+)", mime)
    if not m:
      return self._default_rate, self._default_bps
    bits = int(m.group(1))
    rate = int(m.group(2))
    bps = max(1, bits // 8)
    return rate, bps

  def _limit_audio_pcm(self, part: ProcessorPart) -> ProcessorPart:
    if part.bytes is None:
      return part
    rate, bps = self._parse_pcm(part.mimetype)
    max_bytes = int(self._max_audio_sec * rate * bps)
    data = part.bytes
    if len(data) <= max_bytes:
      return part

    new = ProcessorPart(
        data[:max_bytes],
        mimetype=part.mimetype,
        role=part.role,
        substream_name=part.substream_name,
        metadata=dict(part.metadata),
    )
    new.metadata.update({
        'truncated': True,
        'max_duration_sec': self._max_audio_sec,
        'original_size_bytes': len(data),
        'new_size_bytes': max_bytes,
    })
    logger.info(
        "[media_size] audio truncated {}B -> {}B (rate={} bps={})",
        len(data), max_bytes, rate, bps,
    )
    return new

  # ---------------- Video ----------------
  def _limit_video(self, part: ProcessorPart) -> list[ProcessorPart]:
    if part.bytes is None:
      return [part]
    size = len(part.bytes)
    if size <= self._max_video_bytes:
      return [part]

    if self._video_policy == 'passthrough':
      # Too large but policy allows passing.
      warn = processor.status(
          f"[media_size] video too large={size} > {self._max_video_bytes} (passthrough)"
      )
      logger.info("[media_size] video passthrough size={}B limit={}B", size, self._max_video_bytes)
      return [warn, part]
    elif self._video_policy == 'drop':
      warn = processor.status(
          f"[media_size] video dropped size={size} limit={self._max_video_bytes}"
      )
      logger.info("[media_size] video drop size={}B limit={}B", size, self._max_video_bytes)
      return [warn]
    elif self._video_policy == 'truncate':
      new = ProcessorPart(
          part.bytes[: self._max_video_bytes],
          mimetype=part.mimetype,
          role=part.role,
          substream_name=part.substream_name,
          metadata=dict(part.metadata),
      )
      new.metadata.update({
          'truncated': True,
          'original_size_bytes': size,
          'new_size_bytes': self._max_video_bytes,
      })
      warn = processor.status(
          f"[media_size] video truncated {size}->{self._max_video_bytes} bytes"
      )
      logger.info("[media_size] video truncate {} -> {} bytes", size, self._max_video_bytes)
      return [warn, new]
    return [part]
