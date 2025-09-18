import pytest
from fastapi.testclient import TestClient

import time

from genai_processors import content_api, processor
from real_world.app.main import app
from real_world.agent import models, pipelines


@processor.processor_function
async def _stub_turn_model(content):
  buf = content_api.ProcessorContent()
  async for p in content:
    buf += p
  yield content_api.ProcessorPart(f"model:{content_api.as_text(buf)}", role="model")


@processor.processor_function
async def _stub_caption_model(content):
  async for _ in content:
    yield content_api.ProcessorPart("ALT", role="model")


@processor.processor_function
async def _stub_ws_pipeline(content):
  buffer = content_api.ProcessorContent()
  async for part in content:
    part = content_api.ProcessorPart(part)
    if content_api.is_end_of_turn(part):
      if buffer:
        yield content_api.ProcessorPart(
            f"model:{content_api.as_text(buffer)}", role="model"
        )
        buffer = content_api.ProcessorContent()
    else:
      buffer += part


@processor.processor_function
async def _stub_monitor(content):
  async for p in content:
    yield p


@pytest.fixture(autouse=True)
def patch_models(monkeypatch):
  # Mainline uses per-turn stub pipeline; monitor is no-op; models are stubbed.
  monkeypatch.setattr(models, "build_turn_model", lambda *a, **k: _stub_turn_model)
  monkeypatch.setattr(models, "build_caption_model", lambda *a, **k: _stub_caption_model)
  monkeypatch.setattr(pipelines, "build_mainline_pipeline", lambda: _stub_ws_pipeline)
  monkeypatch.setattr(pipelines, "build_monitor_pipeline", lambda: _stub_monitor)


def _recv_non_reserved(ws, expected_min=1, max_reads=50):
  """Receive up to max_reads messages, skipping reserved and turn_complete.

  NOTE: starlette's WebSocketTestSession.receive_json() does not support
  timeouts. This helper relies on the test stub pipeline to always produce a
  response; otherwise this call may block on the first receive.
  """
  out = []
  for _ in range(max_reads):
    msg = ws.receive_json()
    if msg.get("substream") in ("status", "debug", "caption"):
      continue
    meta = msg.get("metadata") or {}
    if meta.get("turn_complete") is True:
      continue
    out.append(msg)
    if len(out) >= expected_min:
      break
  return out


def test_ws_two_turns_connection_stays_open():
  client = TestClient(app)
  with client.websocket_connect("/ws") as ws:
    # Turn 1
    ws.send_json({"type": "text", "text": "first"})
    ws.send_json({"type": "end_of_turn"})
    out1 = _recv_non_reserved(ws, expected_min=1)
    assert out1 and out1[0]["type"] == "text"
    assert "model:first" in out1[0]["text"]

    # Turn 2 (same connection)
    ws.send_json({"type": "text", "text": "second"})
    ws.send_json({"type": "end_of_turn"})
    out2 = _recv_non_reserved(ws, expected_min=1)
    assert out2 and out2[0]["type"] == "text"
    assert "model:second" in out2[0]["text"]

    # Ensure connection still works for a third quick check
    ws.send_json({"type": "text", "text": "third"})
    ws.send_json({"type": "end_of_turn"})
    out3 = _recv_non_reserved(ws, expected_min=1)
    assert out3 and "model:third" in out3[0]["text"]
