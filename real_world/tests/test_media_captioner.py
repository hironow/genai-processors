import pytest

from genai_processors import content_api, processor, streams
from real_world.agent.media_captioner import MediaCaptioner


@processor.processor_function
async def _stub_caption_model(content):
  async for part in content:
    if content_api.is_image(part.mimetype):
      yield content_api.ProcessorPart("ALT(image facts)", role="model")
    elif content_api.is_audio(part.mimetype):
      yield content_api.ProcessorPart("ALT(audio facts)", role="model")
    elif content_api.is_video(part.mimetype):
      yield content_api.ProcessorPart("ALT(video facts)", role="model")
    else:
      yield part


@pytest.mark.asyncio
async def test_media_captioner_converts_media_to_text_only():
  c = MediaCaptioner(_stub_caption_model)
  inputs = [
      content_api.ProcessorPart("hello", role="user"),
      content_api.ProcessorPart(b"\x89PNG\r\n\x1a\n", mimetype="image/png"),
      content_api.ProcessorPart(b"\x00\x01", mimetype="audio/wav"),
      content_api.ProcessorPart(b"\x00\x01\x02", mimetype="video/mp4"),
  ]

  out = await streams.gather_stream(c.to_processor()(streams.stream_content(inputs)))

  # Expect: original text stays, media parts replaced by text alts (and no media retained)
  texts = [p for p in out if content_api.is_text(p.mimetype) and p.substream_name == ""]
  assert [t.text for t in texts] == [
      "hello",
      "ALT(image facts)",
      "ALT(audio facts)",
      "ALT(video facts)",
  ]
  # No non-text outputs
  assert all(content_api.is_text(p.mimetype) for p in out)


@pytest.mark.asyncio
async def test_media_captioner_emits_log_only_for_media():
  c = MediaCaptioner(_stub_caption_model, log_substream='caption')
  inputs = [
      content_api.ProcessorPart("hi", role="user"),
      content_api.ProcessorPart(b"\x89PNG\r\n\x1a\n", mimetype="image/png"),
  ]

  out = await streams.gather_stream(c.to_processor()(streams.stream_content(inputs)))
  logs = [p for p in out if p.substream_name == 'caption']
  assert len(logs) == 1
  assert logs[0].text.startswith('[caption]')
  # Ensure text input did not generate a caption log
  assert not any(p.text == 'hi' and p.substream_name == 'caption' for p in out)
