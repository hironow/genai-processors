import json
import pytest
from fastapi.testclient import TestClient

from genai_processors import content_api, processor
from real_world.app.main import app
from real_world.agent import pipelines


@processor.processor_function
async def _stub_live(content):
  # For each incoming part, echo text, emit tool/binary on demand; pass through non-text
  async for part in content:
    # Handle tool response first: function_response MIME looks like text/plain
    if part.part.function_response is not None:
      yield content_api.ProcessorPart("model:tool_ok", role="model")
      continue
    if content_api.is_text(part.mimetype):
      t = part.text
      if t == "emit_tool":
        yield content_api.ProcessorPart.from_function_call(
          name="test_tool", args={"k": 2}, role="model"
        )
      elif t == "emit_binary":
        yield content_api.ProcessorPart(b"\x00\x01", mimetype="audio/wav", role="model")
      else:
        yield content_api.ProcessorPart(f"model:{t}", role="model")
    else:
      # Pass through images/audio etc.
      yield part


@processor.processor_function
async def _stub_monitor(content):
  # Lightweight monitor: pass through without modification.
  async for part in content:
    yield part


@pytest.fixture(autouse=True)
def patch_pipelines(monkeypatch):
  # Mainline uses the live stub; monitor is a no-op to avoid external calls.
  monkeypatch.setattr(pipelines, "build_mainline_pipeline", lambda: _stub_live)
  monkeypatch.setattr(pipelines, "build_monitor_pipeline", lambda: _stub_monitor)


def _recv_all(ws, expected_min=1, max_msgs=10, *, skip_status=True):
  out = []
  for _ in range(max_msgs):
    try:
      msg = ws.receive_json()
    except Exception:
      break
    if skip_status and msg.get("substream") == "status":
      continue
    out.append(msg)
    if len(out) >= expected_min:
      break
  return out


@pytest.mark.parametrize(
  "send_msgs, expects",
  [
    # Given text -> Then model echo
    ([{"type": "text", "text": "hello"}], [("text", "model:hello")]),
    # Given tool trigger -> Then tool_call envelope
    ([{"type": "text", "text": "emit_tool"}], [("tool_call", "test_tool")]),
    # Given binary trigger (server generated) -> Then binary payload
    ([{"type": "text", "text": "emit_binary"}], [("binary", None)]),
    # Given inbound image -> Then mainline emits text after EOT (no passthrough)
    ([{"type": "image", "mimetype": "image/png", "data_b64": "iVBORw0KGgo=", "substream": "realtime"}], [("text", "model:")]),
    # Given inbound audio -> Then mainline emits text after EOT
    ([{"type": "audio", "mimetype": "audio/wav", "data_b64": "AAE=", "substream": "realtime"}], [("text", "model:")]),
    # Given inbound video -> Then mainline emits text after EOT
    ([{"type": "video", "mimetype": "video/mp4", "data_b64": "AAE=", "substream": "realtime"}], [("text", "model:")]),
  ],
)
def test_websocket_stream(send_msgs, expects):
  # Given: client
  client = TestClient(app)
  with client.websocket_connect("/ws") as ws:
    # When: send messages (+ end_of_turn for good measure)
    for m in send_msgs:
      ws.send_json(m)
    ws.send_json({"type": "end_of_turn"})

    # Then: receive at least one message and validate shape
    recvd = _recv_all(ws, expected_min=1)
    assert len(recvd) >= 1
    # Validate first expected signature
    etype, maybe_val = expects[0]
    payload = recvd[0]
    assert payload["type"] == etype
    if etype == "text":
      assert payload["text"] == maybe_val
    if etype == "tool_call":
      assert payload["name"] == maybe_val
    if etype == "binary":
      assert "data_b64" in payload and isinstance(payload["data_b64"], str)


def test_websocket_tool_roundtrip():
  client = TestClient(app)
  with client.websocket_connect("/ws") as ws:
    # Trigger a tool call from the model
    ws.send_json({"type": "text", "text": "emit_tool"})
    call_msg = _recv_all(ws, expected_min=1)[0]
    assert call_msg["type"] == "tool_call"
    assert call_msg["name"] == "test_tool"

    # Send tool response (no need to send end_of_turn; server does it)
    ws.send_json({
      "type": "tool_response",
      "name": "test_tool",
      "response": {"result": 42},
    })

    # Expect the model follow-up (from stub)
    follow = _recv_all(ws, expected_min=1)[0]
    assert follow["type"] == "text"
    assert follow["text"] == "model:tool_ok"
