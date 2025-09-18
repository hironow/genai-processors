from __future__ import annotations

"""Pipeline builders shared by REST and WebSocket examples."""

from genai_processors import processor
from genai_processors.core import preamble
from genai_processors.core import realtime
from . import models
from .metrics_logger import ModalityMetricsLogger
from .part_deduper import ContentDeduper
from .media_size_limiter import MediaSizeLimiter
from .media_captioner import MediaCaptioner
from loguru import logger


DEFAULT_SI = [
    "You are a helpful assistant. Keep responses concise and clear.",
]


def build_chat_pipeline() -> processor.Processor:
  """Simple turn-based pipeline with an optional preamble."""
  turn_model = models.build_turn_model(system_instruction=DEFAULT_SI)
  # Example of preamble if you want static content added before the user msgs
  # return preamble.Preamble(content=DEFAULT_SI) + turn_model
  w, h = 1024, 1024
  img_bytes = 400 * 1024
  audio_sec = 15.0
  video_bytes = 6 * 1024 * 1024
  video_policy = 'passthrough'
  logger.info(
      "Building chat pipeline: image={}x{} <= {}B, audio<= {}s, video<= {}B policy={}",
      w, h, img_bytes, audio_sec, video_bytes, video_policy,
  )
  pipe = (
      ModalityMetricsLogger(streaming=False, name="chat_metrics")
      + ContentDeduper(window_size=5000)
      + MediaSizeLimiter(
          max_image_width=w,
          max_image_height=h,
          max_image_bytes=img_bytes,
          max_audio_duration_sec=audio_sec,
          max_video_bytes=video_bytes,
          video_policy=video_policy,
        )
      + MediaCaptioner(models.build_caption_model(), keep_original=False, log_substream='caption')
      + turn_model
  )
  logger.info("Chat pipeline ready: metrics -> dedupe -> limit_media -> caption -> turn_model")
  return pipe


def build_live_pipeline() -> processor.Processor:
  """Realtime pipeline that wraps the turn-based model on the client side.

  reuses the same turn-based model to enable bidirectional streaming.
  """
  turn_model = models.build_turn_model(system_instruction=DEFAULT_SI)
  w, h = 1024, 1024
  img_bytes = 400 * 1024
  audio_sec = 15.0
  video_bytes = 6 * 1024 * 1024
  video_policy = 'passthrough'
  logger.info(
      "Building live pipeline: image={}x{} <= {}B, audio<= {}s, video<= {}B policy={}",
      w, h, img_bytes, audio_sec, video_bytes, video_policy,
  )
  pipe = (
      ModalityMetricsLogger(streaming=True, name="live_metrics")
      + ContentDeduper(window_size=20_000)
      + MediaSizeLimiter(
          max_image_width=w,
          max_image_height=h,
          max_image_bytes=img_bytes,
          max_audio_duration_sec=audio_sec,
          max_video_bytes=video_bytes,
          video_policy=video_policy,
        )
      + MediaCaptioner(models.build_caption_model(), keep_original=False, log_substream='caption')
      + realtime.LiveProcessor(turn_processor=turn_model)
  )
  logger.info("Live pipeline ready: metrics -> dedupe -> limit_media -> caption -> live(turn)")
  return pipe


def build_mainline_pipeline() -> processor.Processor:
  """Mainline for human↔AI 1:1 conversation (simple).

  - No metrics, no media transforms; expects text and end_of_turn only.
  - Use the same turn-based model wrapped in LiveProcessor.
  """
  turn_model = models.build_turn_model(system_instruction=DEFAULT_SI)
  pipe = realtime.LiveProcessor(turn_processor=turn_model)
  logger.info("Mainline pipeline ready: live(turn)")
  return pipe


def build_monitor_pipeline() -> processor.Processor:
  """Monitoring pipeline that observes inputs and emits logs/captions.

  - Logs modality counts/status.
  - Limits media size.
  - Generates alt-text via MediaCaptioner and logs to 'caption' substream.
  - Does NOT invoke the turn model; outputs are for observation only.
  """
  w, h = 1024, 1024
  img_bytes = 400 * 1024
  audio_sec = 15.0
  video_bytes = 6 * 1024 * 1024
  video_policy = 'passthrough'
  logger.info(
      "Building monitor pipeline: image={}x{} <= {}B, audio<= {}s, video<= {}B policy={}",
      w, h, img_bytes, audio_sec, video_bytes, video_policy,
  )
  pipe = (
      ModalityMetricsLogger(streaming=True, name="monitor_metrics")
      + ContentDeduper(window_size=20_000)
      + MediaSizeLimiter(
          max_image_width=w,
          max_image_height=h,
          max_image_bytes=img_bytes,
          max_audio_duration_sec=audio_sec,
          max_video_bytes=video_bytes,
          video_policy=video_policy,
        )
      + MediaCaptioner(models.build_caption_model(), keep_original=False, log_substream='caption')
  )
  logger.info("Monitor pipeline ready: metrics -> dedupe -> limit_media -> caption")
  return pipe
