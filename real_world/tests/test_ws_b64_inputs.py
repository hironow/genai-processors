import base64
import pytest
from fastapi.testclient import TestClient

from genai_processors import content_api, processor
from real_world.app.main import app
from real_world.agent import pipelines


@processor.processor_function
async def _stub_live(content):
  # Echo text; pass through non-text (image/audio/video) unchanged
  async for part in content:
    if content_api.is_text(part.mimetype):
      yield content_api.ProcessorPart(f"model:{part.text}", role="model")
    else:
      yield part


@processor.processor_function
async def _stub_monitor(content):
  async for p in content:
    yield p


@pytest.fixture(autouse=True)
def patch_pipelines(monkeypatch):
  monkeypatch.setattr(pipelines, "build_mainline_pipeline", lambda: _stub_live)
  monkeypatch.setattr(pipelines, "build_monitor_pipeline", lambda: _stub_monitor)


def _recv_first_non_status(ws, max_msgs=10):
  for _ in range(max_msgs):
    msg = ws.receive_json()
    if msg.get("substream") == "status":
      continue
    return msg
  raise AssertionError("no non-status message received")


def test_ws_image_b64_missing_padding():
  client = TestClient(app)
  with client.websocket_connect("/ws") as ws:
    # Same payload as other tests but with padding removed
    b64 = "iVBORw0KGgo"  # was "iVBORw0KGgo="
    ws.send_json({
      "type": "image",
      "mimetype": "image/png",
      "data_b64": b64,
      "substream": "realtime",
    })
    ws.send_json({"type": "end_of_turn"})

    msg = _recv_first_non_status(ws)
    # Under new design, non-text inputs do not passthrough on mainline; expect text after EOT.
    assert msg["type"] == "text"
    assert msg["text"].startswith("model:")


def test_ws_image_b64_data_url_and_whitespace():
  client = TestClient(app)
  with client.websocket_connect("/ws") as ws:
    # data: URL prefix and stray whitespace/newlines should be accepted
    raw = "iVBORw0KGgo"  # missing padding on purpose
    data_url = f"data:image/png;base64,  {raw}\n"
    ws.send_json({
      "type": "image",
      "mimetype": "image/png",
      "data_b64": data_url,
      "substream": "realtime",
    })
    ws.send_json({"type": "end_of_turn"})

    msg = _recv_first_non_status(ws)
    assert msg["type"] == "text"
    assert msg["text"].startswith("model:")


def test_ws_audio_b64_urlsafe_missing_padding():
  client = TestClient(app)
  with client.websocket_connect("/ws") as ws:
    # Generate a urlsafe base64 string and strip '=' padding
    raw_bytes = b"\x00\x01\x02\xfb\xef"
    b64 = base64.urlsafe_b64encode(raw_bytes).decode("ascii").rstrip("=")
    ws.send_json({
      "type": "audio",
      "mimetype": "audio/wav",
      "data_b64": b64,
      "substream": "realtime",
    })
    ws.send_json({"type": "end_of_turn"})

    msg = _recv_first_non_status(ws)
    assert msg["type"] == "text"
    assert msg["text"].startswith("model:")
