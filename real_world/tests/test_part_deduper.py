import asyncio
import pytest

from genai_processors import content_api, processor, streams
from real_world.agent.part_deduper import ContentDeduper


@pytest.mark.asyncio
async def test_deduper_drops_duplicates_and_emits_status():
  # Given: a content deduper
  d = ContentDeduper(window_size=100)

  # And: mixed modalities with duplicates
  img = content_api.ProcessorPart(b"\x89PNG\r\n\x1a\n", mimetype="image/png")
  aud = content_api.ProcessorPart(b"\x00\x01", mimetype="audio/wav")
  seq = [
      content_api.ProcessorPart("t1"),
      content_api.ProcessorPart("t1"),  # dup
      img,
      img,  # dup
      aud,
      aud,  # dup
      content_api.ProcessorPart("t2"),
  ]

  # When: apply deduper
  out = await streams.gather_stream(d.to_processor()(streams.stream_content(seq)))

  # Then: first occurrences pass through with content_hash, duplicates produce status only
  # Extract by substreams
  passed = [p for p in out if p.substream_name == ""]
  status = [p for p in out if p.substream_name == "status"]

  # We expect 4 unique outputs (t1, img, aud, t2)
  assert len(passed) == 4
  assert content_api.is_text(passed[0].mimetype) and passed[0].text == "t1"
  assert passed[0].get_metadata("content_hash")
  assert passed[1].mimetype == "image/png" and passed[1].get_metadata("content_hash")
  assert passed[2].mimetype == "audio/wav" and passed[2].get_metadata("content_hash")
  assert passed[3].text == "t2" and passed[3].get_metadata("content_hash")

  # And 3 status entries for dups
  assert len(status) == 3
  assert all(content_api.is_text(s.mimetype) for s in status)
  assert all("duplicate" in s.text for s in status)

