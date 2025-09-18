from __future__ import annotations

"""Library-only streaming API that uses GenAI Processors internally.

Public API accepts/returns domain events without exposing ProcessorPart types.
"""

import dataclasses
from typing import AsyncIterable, Iterable

from genai_processors import content_api, processor
from genai_processors.core import realtime
from ..agent import pipelines


@dataclasses.dataclass(frozen=True)
class UserEvent:
    """Domain event coming from callers.

    type: "text" | "end_of_turn"
    role: "user" | "system" | "model" (typically "user")
    """

    type: str
    role: str = "user"
    text: str | None = None


@dataclasses.dataclass(frozen=True)
class ModelEvent:
    """Domain event produced by the agent."""

    type: str  # "text" | "tool_call" | "binary" | "other"
    role: str
    text: str | None = None
    mimetype: str | None = None
    metadata: dict | None = None


def _events_to_parts(events: AsyncIterable[UserEvent]) -> AsyncIterable[content_api.ProcessorPart]:
    async def gen():
        async for e in events:
            if e.type == "text" and e.text is not None:
                yield content_api.ProcessorPart(e.text, role=e.role)
            elif e.type == "end_of_turn":
                yield content_api.ProcessorPart.end_of_turn()
            else:
                # Ignore unrecognized events for simplicity.
                continue
    return gen()


def _parts_to_events(parts: Iterable[content_api.ProcessorPart]) -> Iterable[ModelEvent]:
    for p in parts:
        # Skip reserved substreams (debug/status logs).
        if p.substream_name in ("status", "debug"):
            continue
        # Important: check function_call first; function_call Parts have empty
        # mimetype and would otherwise be misclassified as text/plain.
        if p.part.function_call is not None:
            name = p.part.function_call.name
            yield ModelEvent(
                type="text", role=p.role, text=f"[tool_call:{name}]", mimetype=p.mimetype, metadata=p.metadata
            )
        elif p.part.inline_data is not None:
            yield ModelEvent(type="binary", role=p.role, text=None, mimetype=p.mimetype, metadata=p.metadata)
        elif content_api.is_text(p.mimetype):
            yield ModelEvent(type="text", role=p.role, text=p.text, mimetype=p.mimetype, metadata=p.metadata)
        else:
            yield ModelEvent(type="other", role=p.role, text=None, mimetype=p.mimetype, metadata=p.metadata)


async def process_user_event_stream(
    events: AsyncIterable[UserEvent],
) -> AsyncIterable[ModelEvent]:
    """Accepts a user event stream and yields model events.

    The public API surface doesn’t expose genai Processors; internally we
    construct a realtime pipeline (client-side) wrapping our turn-based model.
    """
    live = pipelines.build_live_pipeline()

    async def run():
        async for part in live(_events_to_parts(events)):
            # Yield ModelEvent(s) in the same order.
            for evt in _parts_to_events([part]):
                yield evt

    return run()
