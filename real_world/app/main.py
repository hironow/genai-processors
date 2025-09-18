from __future__ import annotations

import asyncio
import base64
import json
from typing import AsyncIterable, Optional, Literal

from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from genai_processors import content_api, processor, streams
from genai_processors import context as gp_context
from ..agent import pipelines
from loguru import logger
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env

# server: uv run fastapi dev real_world/app/main.py
# client: uv run python -m real_world.tools.ws_client --url ws://127.0.0.1:8000/ws
app = FastAPI(title="GenAI Processors – Real-World Samples")


# ======== REST (messages in, model out) ========


class ChatMessage(BaseModel):
    role: str = Field(..., description="user|model|system")
    content: str = Field(..., description="Plain text content")


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


class ChatResponse(BaseModel):
    text: str
    # Raw parts could also be returned if needed.
    # parts: list[dict] | None = None


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest = Body(...)) -> ChatResponse:
    """Accepts OpenAI-like messages and returns a single text response.

    Converts incoming messages into ProcessorParts (role preserved) and applies a
    turn-based pipeline. Aggregates the text output using content_api.as_text.
    """
    pipeline = pipelines.build_chat_pipeline()

    # Convert to ProcessorParts preserving role.
    parts: list[content_api.ProcessorPart] = [
        content_api.ProcessorPart(m.content, role=m.role) for m in req.messages
    ]

    # Apply the model and gather output parts.
    # Ensure dedicated substreams like 'caption' bypass the model and are
    # delivered directly to the client.
    try:
        # Keep caption as reserved on REST as well; monitoring logs bypass model.
        async with gp_context.context(reserved_substreams=("debug", "status", "caption")):
            output_parts = await processor.apply_async(pipeline, parts)
    except Exception:
        logger.exception("REST /chat: unhandled error while processing request")
        raise
    # Only aggregate default substream text (exclude status/debug logs).
    text = content_api.as_text(output_parts, substream_name="")
    return ChatResponse(text=text)


# ======== WebSocket (streaming both ways) ========


class WSInText(BaseModel):
    type: Literal["text"] = "text"
    role: str = Field("user")
    text: str


class WSInEnd(BaseModel):
    type: Literal["end_of_turn"] = "end_of_turn"


class WSInImage(BaseModel):
    type: Literal["image"] = "image"
    data_b64: str
    mimetype: str  # e.g., image/png, image/jpeg, image/webp
    role: str = Field("user")
    substream: str = Field("")
    metadata: dict | None = None


class WSInAudio(BaseModel):
    type: Literal["audio"] = "audio"
    data_b64: str
    mimetype: str  # e.g., audio/wav, audio/l16;rate=24000
    role: str = Field("user")
    substream: str = Field("realtime")
    metadata: dict | None = None
    # Optional flag for server-side Live API bridges.
    audio_stream_end: bool | None = None

class WSInVideo(BaseModel):
    type: Literal["video"] = "video"
    data_b64: str
    mimetype: str  # e.g., video/mp4, video/webm
    role: str = Field("user")
    substream: str = Field("realtime")
    metadata: dict | None = None

class WSInConfig(BaseModel):
    type: Literal["config"] = "config"
    output_mode: Literal["generic", "typed"] = "generic"


class WSInToolResponse(BaseModel):
    type: Literal["tool_response"] = "tool_response"
    name: str
    response: dict
    id: str | None = None


WSIncoming = WSInText | WSInEnd | WSInImage | WSInAudio | WSInVideo | WSInConfig | WSInToolResponse


def _b64decode_loose(s: str) -> bytes:
    if not s:
        return b""
    t = s.strip()
    if t.startswith("data:"):
        try:
            t = t.split(",", 1)[1]
        except Exception:
            pass
    t = "".join(t.split())
    if (len(t) % 4) != 0:
        t += "=" * (-len(t) % 4)
    try:
        if "-" in t or "_" in t:
            return base64.urlsafe_b64decode(t)
        return base64.b64decode(t)
    except Exception:
        logger.exception("Failed to base64-decode payload (len={})", len(t))
        raise


def _ws_in_to_part(msg: WSIncoming) -> content_api.ProcessorPart:
    if isinstance(msg, WSInText):
        return content_api.ProcessorPart(msg.text, role=msg.role)
    if isinstance(msg, WSInEnd):
        return content_api.ProcessorPart.end_of_turn()
    if isinstance(msg, WSInImage):
        data = _b64decode_loose(msg.data_b64)
        return content_api.ProcessorPart(
            data,
            mimetype=msg.mimetype,
            role=msg.role,
            substream_name=msg.substream,
            metadata=msg.metadata or {},
        )
    if isinstance(msg, WSInAudio):
        md = dict(msg.metadata or {})
        if msg.audio_stream_end:
            md["audio_stream_end"] = True
        data = _b64decode_loose(msg.data_b64)
        return content_api.ProcessorPart(
            data,
            mimetype=msg.mimetype,
            role=msg.role,
            substream_name=msg.substream,
            metadata=md,
        )
    if isinstance(msg, WSInVideo):
        data = _b64decode_loose(msg.data_b64)
        return content_api.ProcessorPart(
            data,
            mimetype=msg.mimetype,
            role=msg.role,
            substream_name=msg.substream,
            metadata=msg.metadata or {},
        )
    if isinstance(msg, WSInToolResponse):
        return content_api.ProcessorPart.from_function_response(
            name=msg.name, response=msg.response, function_call_id=msg.id
        )
    # Fallback, shouldn't happen
    return content_api.ProcessorPart.end_of_turn()


def _iter_to_async_iter(queue: "asyncio.Queue[Optional[content_api.ProcessorPart]]") -> AsyncIterable[content_api.ProcessorPart]:
    async def gen():
        while True:
            item = await queue.get()
            if item is None:
                return
            yield item
    return gen()


def _jsonify_metadata(obj):
    try:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {k: _jsonify_metadata(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonify_metadata(v) for v in obj]
        if hasattr(obj, 'to_json_dict'):
            return _jsonify_metadata(obj.to_json_dict())
        if hasattr(obj, 'model_dump'):
            return _jsonify_metadata(obj.model_dump())
        return str(obj)
    except Exception:
        return str(obj)


def _part_to_ws_payload(part: content_api.ProcessorPart, *, output_mode: str = "generic") -> dict:
    """Serialize ProcessorPart to a compact websocket JSON payload."""
    payload: dict = {
        "mimetype": part.mimetype,
        "role": part.role,
        "substream": part.substream_name,
        "metadata": _jsonify_metadata(part.metadata or {}),
    }
    # Important: detect function calls before text, otherwise they appear as
    # text/plain with empty string.
    if part.part.function_call is not None:
        payload.update({
            "type": "tool_call",
            "name": part.part.function_call.name,
            "args": part.part.function_call.args or {},
        })
    elif part.part.inline_data is not None:
        # Binary payload (image/audio/etc) base64-encode for simplicity.
        b = part.part.inline_data.data or b""
        media_kind = (
            "image"
            if content_api.is_image(part.mimetype)
            else "audio"
            if content_api.is_audio(part.mimetype)
            else "video"
            if content_api.is_video(part.mimetype)
            else "binary"
        )
        data_b64 = base64.b64encode(b).decode("ascii")
        if output_mode == "typed" and media_kind in ("image", "audio", "video"):
            # Emit specialized types when requested by client.
            payload.update({
                "type": media_kind,
                "data_b64": data_b64,
            })
        else:
            payload.update({
                "type": "binary",
                "data_b64": data_b64,
                "media_kind": media_kind,
            })
    elif content_api.is_text(part.mimetype):
        payload.update({"type": "text", "text": part.text})
    else:
        payload.update({"type": "other"})
    return payload


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("WS: connection open from {}", websocket.client)
    # Build mainline (simple) and monitoring pipelines.
    main_pipeline = pipelines.build_mainline_pipeline()
    monitor_pipeline = pipelines.build_monitor_pipeline()

    # Bridge: WS → AsyncIterable[ProcessorPart]
    in_q: asyncio.Queue[Optional[content_api.ProcessorPart]] = asyncio.Queue()

    # Client-configurable output mode (generic|typed). Defaults to generic for
    # backwards compatibility.
    output_mode = "generic"

    def _short(s: str, n: int = 64) -> str:
        return s if len(s) <= n else s[:n] + '...'

    async def consume_ws():
        try:
            while True:
                raw = await websocket.receive_json()
                # Validate/parse message into our Pydantic unions.
                msg: WSIncoming
                t = raw.get("type")
                if t == "end_of_turn":
                    logger.info("WS IN: end_of_turn")
                    msg = WSInEnd()
                elif t == "text":
                    logger.info("WS IN: text len={}", len(raw.get("text", "")))
                    msg = WSInText(**raw)
                elif t == "image":
                    b64 = raw.get("data_b64", "")
                    logger.info(
                        "WS IN: image mime={} b64_len={} sub={}",
                        raw.get("mimetype"), len(b64), raw.get("substream", ""),
                    )
                    msg = WSInImage(**raw)
                elif t == "audio":
                    b64 = raw.get("data_b64", "")
                    logger.info(
                        "WS IN: audio mime={} b64_len={} sub={}",
                        raw.get("mimetype"), len(b64), raw.get("substream", ""),
                    )
                    msg = WSInAudio(**raw)
                elif t == "video":
                    b64 = raw.get("data_b64", "")
                    logger.info(
                        "WS IN: video mime={} b64_len={} sub={}",
                        raw.get("mimetype"), len(b64), raw.get("substream", ""),
                    )
                    msg = WSInVideo(**raw)
                elif t == "config":
                    cfg = WSInConfig(**raw)
                    nonlocal output_mode
                    output_mode = cfg.output_mode
                    # Do not forward config as a part; continue to next message.
                    logger.info("WS IN: config output_mode={}", cfg.output_mode)
                    continue
                elif t == "tool_response":
                    msg = WSInToolResponse(**raw)
                    # Forward the tool response and immediately request a new
                    # turn to allow the model to continue using tool results.
                    logger.info(
                        "WS IN: tool_response name={} id={}", msg.name, msg.id
                    )
                    in_q.put_nowait(_ws_in_to_part(msg))
                    in_q.put_nowait(content_api.ProcessorPart.end_of_turn())
                    continue
                else:
                    # Unknown type → ignore
                    logger.warning("WS IN: unknown message: {}", _short(str(raw)))
                    continue
                in_q.put_nowait(_ws_in_to_part(msg))
        except WebSocketDisconnect:
            logger.info("WS: client disconnected")
        except Exception:
            logger.exception("WS IN: unhandled error in consumer loop")
        finally:
            in_q.put_nowait(None)

    # Start consumer (WS→queue) task.
    consumer_task = asyncio.create_task(consume_ws())

    # Helpers
    send_lock = asyncio.Lock()

    async def _send_part(part: content_api.ProcessorPart):
        payload = _part_to_ws_payload(part, output_mode=output_mode)
        # Avoid dumping binary; log type/substream/size info only.
        if payload.get("type") == "binary":
            logger.info(
                "WS OUT: {} kind={} b64_len={} sub={}",
                payload.get("type"),
                payload.get("media_kind"),
                len(payload.get("data_b64", "")),
                payload.get("substream", ""),
            )
        else:
            txt = payload.get("text")
            logger.info(
                "WS OUT: {} sub={} text={}",
                payload.get("type"),
                payload.get("substream", ""),
                _short(txt if isinstance(txt, str) else ""),
            )
        async with send_lock:
            await websocket.send_json(payload)

    # Split input stream for chat vs monitor, and create a control channel.
    input_stream = _iter_to_async_iter(in_q)
    chat_in, monitor_in = streams.split(input_stream, n=2, with_copy=True)
    control_q: asyncio.Queue[Optional[content_api.ProcessorPart]] = asyncio.Queue()
    control_stream = streams.dequeue(control_q)

    async def _mainline_allowed(src: AsyncIterable[content_api.ProcessorPart]):
        async for p in src:
            if (
                content_api.is_text(p.mimetype)
                or content_api.is_end_of_turn(p)
                or (p.part.function_response is not None)
            ):
                yield p

    # Mainline input is chat text/EOT plus control events from the monitor.
    main_in = streams.merge([_mainline_allowed(chat_in), control_stream])
    # Build mainline output once and tee: client vs monitor mirror.
    main_out = main_pipeline(main_in)
    main_to_client, main_to_monitor = streams.split(main_out, n=2, with_copy=True)

    async def run_monitor():
        try:
            async with gp_context.context(reserved_substreams=("debug", "status", "caption")):
                # Mirror mainline outputs to reserved 'status' logs for observers.
                async def _mirror_main_outputs(src: AsyncIterable[content_api.ProcessorPart]):
                    async for p in src:
                        try:
                            if p.part.function_call is not None:
                                txt = f"[model_out] tool_call {p.part.function_call.name}"
                            elif content_api.is_text(p.mimetype):
                                t = p.text
                                txt = f"[model_out] {t[:160]}{'...' if len(t)>160 else ''}"
                            elif content_api.is_image(p.mimetype):
                                txt = "[model_out] <image>"
                            elif content_api.is_audio(p.mimetype):
                                txt = "[model_out] <audio>"
                            elif content_api.is_video(p.mimetype):
                                txt = "[model_out] <video>"
                            else:
                                txt = "[model_out] <other>"
                            yield content_api.ProcessorPart(
                                txt,
                                role="model",
                                substream_name=processor.STATUS_STREAM,
                                metadata={"origin": "mainline_mirror"},
                            )
                        except Exception:
                            logger.debug("monitor: failed to mirror main output", exc_info=True)

                monitor_stream = streams.merge([
                    monitor_pipeline(monitor_in),
                    _mirror_main_outputs(main_to_monitor),
                ])

                async for out in monitor_stream:
                    # If monitor produced a custom_event on the default stream, inject into the mainline.
                    try:
                        if content_api.is_text(out.mimetype) and not gp_context.is_reserved_substream(out.substream_name):
                            ce = out.metadata.get("custom_event")
                            # Backward-compat: accept legacy caption_event=True as a caption custom_event.
                            if ce is None and out.metadata.get("caption_event"):
                                ce = {"type": "caption", "data": {"original_mimetype": out.metadata.get("original_mimetype")}}
                            if ce is not None:
                                inject = ce.get("inject", {}) if isinstance(ce, dict) else {}
                                role = inject.get("role", "user")
                                text_override = inject.get("text")
                                end_of_turn = bool(inject.get("end_of_turn", False))
                                evt_text = text_override if isinstance(text_override, str) else out.text
                                evt = content_api.ProcessorPart(
                                    evt_text,
                                    role=role,
                                    metadata={
                                        "origin": "custom_event",
                                        "event_type": (ce.get("type") if isinstance(ce, dict) else "custom"),
                                        **(ce.get("data", {}) if isinstance(ce, dict) else {}),
                                    },
                                )
                                control_q.put_nowait(evt)
                                if end_of_turn:
                                    control_q.put_nowait(content_api.ProcessorPart.end_of_turn())
                                # Do not forward default-stream event text to client from monitor branch.
                                continue
                    except Exception:
                        logger.debug("monitor: failed to inspect custom_event metadata", exc_info=True)

                    # Forward only reserved-substream logs (status/debug/caption) to client.
                    if gp_context.is_reserved_substream(out.substream_name):
                        await _send_part(out)
                    # Ignore non-reserved outputs from the monitor branch.
        except Exception:
            logger.exception("WS monitor loop error")
        finally:
            control_q.put_nowait(None)

    async def run_mainline():
        try:
            async with gp_context.context(reserved_substreams=("debug", "status", "caption")):
                async for out in main_to_client:
                    await _send_part(out)
        except Exception:
            logger.exception("WS mainline loop error")

    # Run monitor and mainline concurrently.
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(run_monitor())
            tg.create_task(run_mainline())
    except WebSocketDisconnect:
        logger.info("WS: client disconnected")
    except Exception:
        logger.exception("WS: unhandled error in concurrent loops")
    finally:
        consumer_task.cancel()
        logger.info("WS: connection closed")


# ========== Root ==========


@app.get("/")
async def root():
    return {"ok": True, "endpoints": ["POST /chat", "WS /ws"]}
