import pytest
from fastapi.testclient import TestClient

from genai_processors import content_api, processor
from real_world.app.main import app
from real_world.agent import pipelines


@processor.processor_function
async def _stub_turn_model(content):
  # Given: buffer input text
  buf = content_api.ProcessorContent()
  async for part in content:
    buf += part
  txt = content_api.as_text(buf)
  # When: produce a deterministic model response with optional extras
  yield content_api.ProcessorPart(f"stub:{txt}", role="model")
  if "emit_image" in txt:
    yield content_api.ProcessorPart(b"\x89PNG\r\n\x1a\n", mimetype="image/png", role="model")
  if "emit_tool" in txt:
    yield content_api.ProcessorPart.from_function_call(name="test_tool", args={"x": 1}, role="model")


@pytest.fixture(autouse=True)
def patch_pipeline(monkeypatch):
  monkeypatch.setattr(pipelines, "build_chat_pipeline", lambda: _stub_turn_model)


@pytest.mark.parametrize(
  "messages, expected",
  [
    # Given/When/Then: simple echo
    (
      [{"role": "user", "content": "hello"}],
      "stub:hello",
    ),
    # Given/When/Then: extras do not affect REST aggregated text
    (
      [{"role": "system", "content": "emit_image emit_tool"}, {"role": "user", "content": "ok"}],
      "stub:emit_image emit_toolok",
    ),
  ],
)
def test_chat_rest(messages, expected):
  # Given: FastAPI client
  client = TestClient(app)

  # When: call /chat with messages
  resp = client.post("/chat", json={"messages": messages})

  # Then: response text matches stubbed model output aggregation
  assert resp.status_code == 200
  body = resp.json()
  assert body["text"] == expected

