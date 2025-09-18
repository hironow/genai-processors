import asyncio
import pytest

from genai_processors import content_api, processor
from real_world.lib import streaming_agent
from real_world.agent import pipelines


@processor.processor_function
async def _stub_live(content):
  async for part in content:
    if content_api.is_text(part.mimetype):
      t = part.text
      if t == "emit_tool":
        yield content_api.ProcessorPart.from_function_call(name="t", args={"a": 1}, role="model")
      elif t == "emit_binary":
        yield content_api.ProcessorPart(b"\x00\x01\x02", mimetype="image/png", role="model")
      else:
        yield content_api.ProcessorPart(f"M:{t}", role="model")


@pytest.fixture(autouse=True)
def patch_live_pipeline(monkeypatch):
  monkeypatch.setattr(pipelines, "build_live_pipeline", lambda: _stub_live)


async def _aiter(seq):
  for x in seq:
    yield x


@pytest.mark.asyncio
@pytest.mark.parametrize(
  "events, expects",
  [
    # Given text -> Then text ModelEvent
    ([streaming_agent.UserEvent(type="text", text="hi")], [("text", "M:hi")]),
    # Given tool -> Then mapped as text with marker
    ([streaming_agent.UserEvent(type="text", text="emit_tool")], [("text", "[tool_call:t]")]),
    # Given binary -> Then binary ModelEvent
    ([streaming_agent.UserEvent(type="text", text="emit_binary")], [("binary", None)]),
  ],
)
async def test_library_streaming(events, expects):
  # Given: a user event stream
  async_events = _aiter(events)

  # When: process via library API (internally uses genai-processors)
  out_iter = await streaming_agent.process_user_event_stream(async_events)

  # Then: collect and validate
  results = []
  async for e in out_iter:
    results.append(e)
    break  # one output is enough for this test

  assert results, "no output events produced"
  etype, maybe_text = expects[0]
  evt = results[0]
  assert evt.type == etype
  if etype == "text":
    assert evt.text == maybe_text
  if etype == "binary":
    assert evt.mimetype in ("audio/wav", "image/png")

