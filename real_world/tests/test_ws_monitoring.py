import pytest
from fastapi.testclient import TestClient

from genai_processors import content_api, processor
from real_world.app.main import app
from real_world.agent import pipelines


@processor.processor_function
async def _stub_mainline(content):
  # Echo text; if EOT only, returns "model:" once.
  async for part in content:
    if content_api.is_text(part.mimetype):
      yield content_api.ProcessorPart(f"model:{part.text}", role="model")


@processor.processor_function
async def _stub_monitor_with_event(content):
  # When an image/audio/video arrives, emit a custom_event that injects
  # a user text into the mainline and requests an end_of_turn.
  async for part in content:
    if content_api.is_image(part.mimetype) or content_api.is_audio(part.mimetype) or content_api.is_video(part.mimetype):
      yield content_api.ProcessorPart(
        "caption-text",
        role="model",
        metadata={
          "custom_event": {
            "type": "caption",
            "inject": {"role": "user", "end_of_turn": True},
            "data": {"from": "stub_monitor"},
          }
        },
      )
    # also pass through (ignored by monitor loop for default stream)
    yield part


@pytest.fixture(autouse=True)
def patch_pipelines(monkeypatch):
  monkeypatch.setattr(pipelines, "build_mainline_pipeline", lambda: _stub_mainline)
  monkeypatch.setattr(pipelines, "build_monitor_pipeline", lambda: _stub_monitor_with_event)


def test_event_injection_delivered_to_mainline():
  client = TestClient(app)
  with client.websocket_connect("/ws") as ws:
    # Send an image; monitor should emit a custom_event that injects text and EOT.
    ws.send_json({
      "type": "image",
      "mimetype": "image/png",
      "data_b64": "iVBORw0KGgo=",
      "substream": "realtime",
    })

    # Collect non-reserved messages until we see the injected model text.
    got = []
    for _ in range(10):
      msg = ws.receive_json()
      if msg.get("substream") in ("status", "debug", "caption"):
        continue
      got.append(msg)
      if msg.get("type") == "text" and msg.get("text", "").startswith("model:caption-text"):
        break

    assert any(m.get("type") == "text" and m.get("text", "").startswith("model:caption-text") for m in got), "injected caption-text not observed on mainline"


def test_monitor_mirrors_model_out_as_status():
  client = TestClient(app)
  with client.websocket_connect("/ws") as ws:
    ws.send_json({"type": "text", "text": "hello"})
    ws.send_json({"type": "end_of_turn"})

    # Look for a status message mirroring the model output.
    mirrored = None
    for _ in range(20):
      msg = ws.receive_json()
      if msg.get("substream") == "status" and msg.get("type") == "text" and msg.get("text", "").startswith("[model_out]"):
        mirrored = msg
        break
    assert mirrored is not None, "no [model_out] mirror observed on monitor stream"

