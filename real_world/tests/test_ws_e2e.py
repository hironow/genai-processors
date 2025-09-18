import base64
import pytest
from fastapi.testclient import TestClient

from genai_processors import content_api, processor
from real_world.app.main import app
from real_world.agent import models


@processor.processor_function
async def _stub_turn_model(content):
  # Buffer everything and emit a single text response.
  buf = content_api.ProcessorContent()
  async for p in content:
    buf += p
  yield content_api.ProcessorPart(f"model:{content_api.as_text(buf)}", role="model")


@processor.processor_function
async def _stub_caption_model(content):
  async for p in content:
    if content_api.is_image(p.mimetype):
      yield content_api.ProcessorPart("ALT(image facts)", role="model")
    elif content_api.is_audio(p.mimetype):
      yield content_api.ProcessorPart("ALT(audio facts)", role="model")
    elif content_api.is_video(p.mimetype):
      yield content_api.ProcessorPart("ALT(video facts)", role="model")


def _recv_non_reserved(ws, expected_min=1, max_msgs=20, reserved=("debug", "status", "caption")):
  out = []
  for _ in range(max_msgs):
    try:
      msg = ws.receive_json()
    except Exception:
      break
    if msg.get("substream") in reserved:
      continue
    out.append(msg)
    if len(out) >= expected_min:
      break
  return out


@pytest.fixture(autouse=True)
def patch_models(monkeypatch):
  monkeypatch.setattr(models, "build_turn_model", lambda *a, **k: _stub_turn_model)
  monkeypatch.setattr(models, "build_caption_model", lambda *a, **k: _stub_caption_model)


def test_ws_e2e_simple_text_roundtrip():
  client = TestClient(app)
  with client.websocket_connect("/ws") as ws:
    # Send a simple text and end the turn
    ws.send_json({"type": "text", "text": "hello"})
    ws.send_json({"type": "end_of_turn"})

    msgs = _recv_non_reserved(ws, expected_min=1)
    assert msgs, "no output"
    first = msgs[0]
    assert first["type"] == "text"
    assert first["text"].startswith("model:")
    assert "hello" in first["text"]


def test_ws_e2e_media_caption_and_log():
  client = TestClient(app)
  with client.websocket_connect("/ws") as ws:
    # Send an image (minimal valid header) and then end_of_turn
    ws.send_json({
      "type": "image",
      "mimetype": "image/png",
      "data_b64": "iVBORw0KGgo=",
      "substream": "realtime",
    })
    ws.send_json({"type": "end_of_turn"})

    # Expect a caption log (reserved) and a model text using the caption in prompt
    # First read a few messages to capture reserved caption logs
    logs = []
    non_reserved = []
    for _ in range(10):
      msg = ws.receive_json()
      if msg.get("substream") == "caption":
        logs.append(msg)
      elif msg.get("substream") in ("status", "debug"):
        continue
      else:
        non_reserved.append(msg)
        if len(non_reserved) >= 1:
          break

    assert logs, "no caption log received"
    assert logs[0]["type"] == "text"
    assert logs[0]["text"].startswith("[caption]")

    assert non_reserved, "no model output received"
    out = non_reserved[0]
    assert out["type"] == "text"
    assert out["text"].startswith("model:")
    # In the new split design, caption is observed on the monitor branch and may
    # be injected asynchronously; do not require immediate inclusion in first model output.


def test_ws_e2e_alpha_png_does_not_crash():
  # Create a small grayscale+alpha PNG and send via WS; server should process it
  # through MediaSizeLimiter (JPEG target) without crashing.
  import io
  from PIL import Image
  import base64 as _b64

  img = Image.new('LA', (16, 16), color=(128, 120))
  bio = io.BytesIO()
  img.save(bio, format='PNG')
  png_bytes = bio.getvalue()
  b64 = _b64.b64encode(png_bytes).decode('ascii')

  client = TestClient(app)
  with client.websocket_connect("/ws") as ws:
    ws.send_json({
      "type": "image",
      "mimetype": "image/png",
      "data_b64": b64,
      "substream": "realtime",
    })
    ws.send_json({"type": "end_of_turn"})

    msgs = _recv_non_reserved(ws, expected_min=1)
    assert msgs, "no output received"
    # Expect model text (stub model echoes buffered text) or binary; in either
    # case, the pipeline did not crash due to JPEG alpha mode issues.
    assert msgs[0]["type"] in ("text", "binary")
