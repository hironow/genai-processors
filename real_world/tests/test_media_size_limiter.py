import io
import math
import pytest

from genai_processors import content_api, streams
from real_world.agent.media_size_limiter import MediaSizeLimiter
from PIL import Image


def _make_image_bytes(w=1200, h=800, fmt='PNG') -> bytes:
  img = Image.new('RGB', (w, h), color=(128, 200, 255))
  bio = io.BytesIO()
  img.save(bio, format=fmt)
  return bio.getvalue()


def _make_pcm(duration_sec: float, rate=24000, freq=440.0) -> bytes:
  # 16-bit mono PCM
  total = int(duration_sec * rate)
  out = bytearray()
  for n in range(total):
    s = int(0.3 * 32767 * math.sin(2 * math.pi * freq * n / rate))
    out += int.to_bytes(max(-32768, min(32767, s)) & 0xFFFF, 2, 'little', signed=False)
  return bytes(out)


@pytest.mark.asyncio
async def test_image_resized_and_recompressed():
  limiter = MediaSizeLimiter(max_image_width=300, max_image_height=300, max_image_bytes=50*1024, image_format='JPEG')
  img_bytes = _make_image_bytes(1200, 800, 'PNG')
  p = content_api.ProcessorPart(img_bytes, mimetype='image/png', role='user')
  out = await streams.gather_stream(limiter.to_processor()(streams.stream_content([p])))
  assert len(out) == 1
  q = out[0]
  assert q.mimetype in ('image/jpeg', 'image/webp')
  # Dimensions reduced
  w, h = q.pil_image.size
  assert w <= 300 and h <= 300
  # Metadata indicates recompression/resized
  assert q.get_metadata('recompressed') is True


@pytest.mark.asyncio
async def test_image_with_alpha_to_jpeg_no_error():
  # Create a small grayscale+alpha PNG and ensure JPEG recompression does not crash.
  img = Image.new('LA', (32, 32), color=(128, 120))
  bio = io.BytesIO()
  img.save(bio, format='PNG')
  png_bytes = bio.getvalue()

  limiter = MediaSizeLimiter(max_image_width=64, max_image_height=64, max_image_bytes=100*1024, image_format='JPEG')
  p = content_api.ProcessorPart(png_bytes, mimetype='image/png', role='user')
  out = await streams.gather_stream(limiter.to_processor()(streams.stream_content([p])))
  assert len(out) == 1
  q = out[0]
  # Should be re-encoded as JPEG and have bytes
  assert q.mimetype == 'image/jpeg'
  assert isinstance(q.bytes, (bytes, bytearray)) and len(q.bytes) > 0
  # PIL should be able to load it
  _ = q.pil_image


@pytest.mark.asyncio
async def test_audio_pcm_truncated():
  limiter = MediaSizeLimiter(max_audio_duration_sec=0.2)
  audio = _make_pcm(0.5, rate=24000)
  p = content_api.ProcessorPart(audio, mimetype='audio/l16;rate=24000', role='user')
  out = await streams.gather_stream(limiter.to_processor()(streams.stream_content([p])))
  assert len(out) == 1
  q = out[0]
  assert q.mimetype.startswith('audio/l16')
  assert q.get_metadata('truncated') is True
  # Expect ~0.2s of audio -> 0.2 * 24000 * 2 bytes
  assert len(q.bytes) == int(0.2 * 24000 * 2)


@pytest.mark.asyncio
async def test_video_policy_drop_and_truncate():
  big = b'x' * (2 * 1024 * 1024)
  # Drop policy
  drop = MediaSizeLimiter(max_video_bytes=100*1024, video_policy='drop')
  p = content_api.ProcessorPart(big, mimetype='video/mp4', role='user')
  out = await streams.gather_stream(drop.to_processor()(streams.stream_content([p])))
  # Only status part
  assert len([o for o in out if o.substream_name == '']) == 0
  assert any(o.substream_name == 'status' for o in out)

  # Truncate policy
  trunc = MediaSizeLimiter(max_video_bytes=100*1024, video_policy='truncate')
  out2 = await streams.gather_stream(trunc.to_processor()(streams.stream_content([p])))
  data_parts = [o for o in out2 if o.substream_name == '']
  assert len(data_parts) == 1
  assert len(data_parts[0].bytes) == 100*1024
  assert data_parts[0].get_metadata('truncated') is True
