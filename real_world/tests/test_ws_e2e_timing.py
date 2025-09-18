import time
import pytest
from fastapi.testclient import TestClient

from genai_processors import content_api, processor
from real_world.app.main import app
from real_world.agent import models


@processor.processor_function
async def _slow_turn_model(content):
  # Simulate slower model to observe reserved-stream early emission.
  buf = content_api.ProcessorContent()
  async for p in content:
    buf += p
  # Delay to ensure status/caption logs arrive first.
  import asyncio
  await asyncio.sleep(0.2)
  yield content_api.ProcessorPart(f"model:{content_api.as_text(buf)}", role="model")


@processor.processor_function
async def _fast_caption_model(content):
  # Return caption immediately.
  async for p in content:
    if content_api.is_image(p.mimetype):
      yield content_api.ProcessorPart("ALT(image)")
    elif content_api.is_audio(p.mimetype):
      yield content_api.ProcessorPart("ALT(audio)")
    elif content_api.is_video(p.mimetype):
      yield content_api.ProcessorPart("ALT(video)")


@pytest.fixture(autouse=True)
def patch_models(monkeypatch):
  monkeypatch.setattr(models, "build_turn_model", lambda *a, **k: _slow_turn_model)
  monkeypatch.setattr(models, "build_caption_model", lambda *a, **k: _fast_caption_model)


def _recv_until(ws, predicate, timeout=1.0):
  start = time.perf_counter()
  while time.perf_counter() - start < timeout:
    msg = ws.receive_json()
    if predicate(msg):
      return msg, time.perf_counter() - start
  raise AssertionError("expected message not received before timeout")


def test_ws_text_status_latency_and_count():
  client = TestClient(app)
  with client.websocket_connect("/ws") as ws:
    # First text
    ws.send_json({"type": "text", "text": "A"})
    # Read until first status text for text_count=1
    msg1, dt1 = _recv_until(
        ws,
        lambda m: m.get("substream") == "status" and m.get("type") == "text" and "text_count=1" in m.get("text", ""),
    )
    assert dt1 < 0.15, f"status log delayed too much: {dt1:0.3f}s"

    # Second text
    ws.send_json({"type": "text", "text": "B"})
    msg2, dt2 = _recv_until(
        ws,
        lambda m: m.get("substream") == "status" and m.get("type") == "text" and "text_count=2" in m.get("text", ""),
    )
    assert dt2 < 0.15, f"second status log delayed too much: {dt2:0.3f}s"


def test_ws_caption_log_latency_single_per_media():
  client = TestClient(app)
  with client.websocket_connect("/ws") as ws:
    # Send an image
    ws.send_json({
      "type": "image",
      "mimetype": "image/png",
      "data_b64": "iVBORw0KGgo=",
      "substream": "realtime",
    })
    # Expect caption log quickly
    msg, dt = _recv_until(ws, lambda m: m.get("substream") == "caption", timeout=1.0)
    assert dt < 0.2, f"caption log delayed too much: {dt:0.3f}s"

    # Trigger a turn so that model produces a response
    ws.send_json({"type": "end_of_turn"})

    # Read until next non-reserved message (the model output)
    for _ in range(10):
      m = ws.receive_json()
      if m.get("substream") in ("status", "debug", "caption"):
        continue
      assert m["type"] == "text"
      break
    else:
      raise AssertionError("expected non-reserved model output not received")
