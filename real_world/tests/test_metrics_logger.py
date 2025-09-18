import asyncio
import time

import pytest

from genai_processors import content_api, streams
from real_world.agent.metrics_logger import ModalityMetricsLogger


@pytest.mark.asyncio
async def test_status_logs_immediate_for_text():
  # Given: a metrics logger and two text inputs spaced in time
  logger = ModalityMetricsLogger(streaming=False, name="test_metrics")
  inputs = [
      content_api.ProcessorPart("first", role="user"),
      content_api.ProcessorPart("second", role="user"),
  ]
  in_stream = streams.stream_content(inputs, with_delay_sec=0.05, delay_first=False, delay_end=False)

  # When: consume first 4 outputs (status, first, status, second)
  out_parts = []
  out_times = []
  t0 = time.perf_counter()
  async for part in logger(in_stream):
    out_parts.append(part)
    out_times.append(time.perf_counter() - t0)
    if len(out_parts) >= 4:
      break

  # Then: order is [status(log1), first, status(log2), second]
  assert out_parts[0].substream_name == "status"
  assert content_api.is_text(out_parts[0].mimetype)
  assert "text_count=1" in out_parts[0].text

  assert out_parts[1].substream_name == ""
  assert content_api.is_text(out_parts[1].mimetype)
  assert out_parts[1].text == "first"

  assert out_parts[2].substream_name == "status"
  assert content_api.is_text(out_parts[2].mimetype)
  assert "text_count=2" in out_parts[2].text

  assert out_parts[3].substream_name == ""
  assert content_api.is_text(out_parts[3].mimetype)
  assert out_parts[3].text == "second"

  # And: the second status log is emitted promptly after the first data part
  # (roughly equal to the input inter-part delay; allow generous upper bound).
  dt = out_times[2] - out_times[1]
  assert dt < 0.2, f"status log appears delayed (dt={dt:0.3f}s)"

