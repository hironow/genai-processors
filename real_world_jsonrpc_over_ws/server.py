from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import asdict, dataclass
from typing import Any, AsyncIterable

import websockets
from loguru import logger
from dotenv import load_dotenv

from genai_processors import content_api
from genai_processors import processor
from genai_processors import context as gp_context
from real_world.agent import pipelines


# ===== JSON-RPC helpers =====


def _rpc_response_ok(id_: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _rpc_response_error(id_: Any, code: int, message: str, data: Any | None = None) -> dict:
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": id_, "error": err}


def _rpc_notification(method: str, params: Any) -> dict:
    return {"jsonrpc": "2.0", "method": method, "params": params}


def _is_jsonrpc_request(obj: dict) -> bool:
    return isinstance(obj, dict) and obj.get("jsonrpc") == "2.0" and "method" in obj


# ===== Domain payloads (client-visible) =====


@dataclass(frozen=True)
class Message:
    role: str
    content: str


def _short(s: str, n: int = 200) -> str:
    return s if len(s) <= n else s[:n] + "..."


def _part_to_event(part: content_api.ProcessorPart) -> dict | None:
    """Convert ProcessorPart to a compact event dict for client.

    Skips reserved substreams (status/debug/caption). Returns None when filtered.
    """
    if gp_context.is_reserved_substream(part.substream_name):
        return None

    # Function calls must be detected first (mimetype is empty)
    if part.part.function_call is not None:
        name = part.part.function_call.name
        return {
            "type": "tool_call",
            "role": part.role,
            "name": name,
            "args": part.part.function_call.args or {},
            "metadata": _jsonify_metadata(part.metadata or {}),
        }
    if part.part.inline_data is not None:
        media_kind = (
            "image"
            if content_api.is_image(part.mimetype)
            else "audio"
            if content_api.is_audio(part.mimetype)
            else "video"
            if content_api.is_video(part.mimetype)
            else "binary"
        )
        data_b64 = base64.b64encode(part.part.inline_data.data or b"").decode("ascii")
        return {
            "type": media_kind,
            "role": part.role,
            "mimetype": part.mimetype,
            "data_b64": data_b64,
            "metadata": _jsonify_metadata(part.metadata or {}),
        }
    if content_api.is_text(part.mimetype):
        return {
            "type": "text",
            "role": part.role,
            "text": part.text,
            "mimetype": part.mimetype,
            "metadata": _jsonify_metadata(part.metadata or {}),
        }
    return {
        "type": "other",
        "role": part.role,
        "mimetype": part.mimetype,
        "metadata": _jsonify_metadata(part.metadata or {}),
    }


def _jsonify_metadata(obj):
    try:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {k: _jsonify_metadata(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonify_metadata(v) for v in obj]
        if hasattr(obj, "to_json_dict"):
            return _jsonify_metadata(obj.to_json_dict())
        if hasattr(obj, "model_dump"):
            return _jsonify_metadata(obj.model_dump())
        # dataclasses or pydantic-like objects sometimes support asdict
        try:
            return _jsonify_metadata(asdict(obj))
        except Exception:
            pass
        return str(obj)
    except Exception:
        return str(obj)


def _process_messages_stream(
    messages: list[Message],
) -> AsyncIterable[content_api.ProcessorPart]:
    """Run the live pipeline and yield parts as they are produced."""
    live = pipelines.build_live_pipeline()

    async def _in_stream():
        for m in messages:
            yield content_api.ProcessorPart(m.content, role=m.role)
        yield content_api.ProcessorPart.end_of_turn()

    async def _run():
        async with gp_context.context(reserved_substreams=("debug", "status", "caption")):
            async for p in live(_in_stream()):
                yield p

    return _run()


async def _handle_chat_process(
    request_id: Any,
    params: dict,
    websocket,
    send_lock: asyncio.Lock,
):
    """Handle chat.process RPC: stream chunks via notifications then return final result."""
    try:
        raw_msgs = params.get("messages")
        if not isinstance(raw_msgs, list):
            raise ValueError("params.messages must be a list of {role, content}")
        msgs: list[Message] = []
        for i, m in enumerate(raw_msgs):
            if not isinstance(m, dict) or "role" not in m or "content" not in m:
                raise ValueError("each message requires role and content")
            msgs.append(Message(role=str(m["role"]), content=str(m["content"])) )

        logger.info("RPC chat.process: messages={} first='{}'", len(msgs), _short(msgs[0].content) if msgs else "")

        # Aggregate plain-text on default stream for convenience result.text
        agg_text: list[str] = []

        async for part in _process_messages_stream(msgs):
            evt = _part_to_event(part)
            if evt is None:
                # Reserved stream (status/debug/caption)
                continue
            # Accumulate only default stream text
            try:
                if evt.get("type") == "text" and (part.substream_name or "") == "":
                    agg_text.append(evt.get("text", ""))
            except Exception:
                pass

            note = _rpc_notification(
                "chat.chunk",
                {"request_id": request_id, "event": evt},
            )
            async with send_lock:
                await websocket.send(json.dumps(note, ensure_ascii=False))

        # Final result
        result = {"text": "".join(agg_text)}
        resp = _rpc_response_ok(request_id, result)
        async with send_lock:
            await websocket.send(json.dumps(resp, ensure_ascii=False))
    except asyncio.CancelledError:
        # Respond to original request with a cancellation error and exit cleanly.
        try:
            resp = _rpc_response_error(request_id, -32800, "Request cancelled")
            async with send_lock:
                await websocket.send(json.dumps(resp, ensure_ascii=False))
        except Exception:
            pass
        return
    except Exception as e:
        logger.exception("chat.process failed")
        err = _rpc_response_error(request_id, -32000, "chat.process failed", {"detail": str(e)})
        async with send_lock:
            await websocket.send(json.dumps(err, ensure_ascii=False))


async def _connection_handler(websocket):
    logger.info("WS JSON-RPC: client connected: {}", getattr(websocket, "remote_address", None))
    send_lock = asyncio.Lock()
    inflight: dict[Any, asyncio.Task] = {}
    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except Exception:
                logger.warning("non-JSON message ignored")
                continue
            if not _is_jsonrpc_request(msg):
                logger.warning("not a JSON-RPC 2.0 request: {}", msg)
                continue
            method = msg.get("method")
            req_id = msg.get("id")
            params = msg.get("params", {}) or {}

            if method == "ping":
                async with send_lock:
                    await websocket.send(json.dumps(_rpc_response_ok(req_id, {"pong": True})))
                continue

            if method == "chat.process":
                task = asyncio.create_task(_handle_chat_process(req_id, params, websocket, send_lock))
                if req_id is not None:
                    inflight[req_id] = task
                    # Cleanup mapping on finish
                    def _done_cb(t: asyncio.Task, rid=req_id):
                        inflight.pop(rid, None)
                    task.add_done_callback(_done_cb)
                # For None id, we still run the task without mapping
                continue

            if method == "chat.cancel":
                target_id = params.get("request_id") if isinstance(params, dict) else None
                if target_id is None:
                    async with send_lock:
                        await websocket.send(json.dumps(_rpc_response_error(req_id, -32602, "Invalid params: request_id required")))
                    continue
                task = inflight.get(target_id)
                had = task is not None
                if had:
                    task.cancel()
                async with send_lock:
                    await websocket.send(json.dumps(_rpc_response_ok(req_id, {"cancelled": bool(had), "request_id": target_id})))
                continue

            # Unknown method
            async with send_lock:
                await websocket.send(json.dumps(_rpc_response_error(req_id, -32601, f"Method not found: {method}")))

    except websockets.exceptions.ConnectionClosed:
        logger.info("WS JSON-RPC: connection closed")
    finally:
        # Cancel any remaining tasks
        for t in list(inflight.values()):
            t.cancel()
        if inflight:
            await asyncio.gather(*inflight.values(), return_exceptions=True)


async def main(host: str = "127.0.0.1", port: int = 8765):
    load_dotenv()
    logger.info("Starting JSON-RPC over WS server on ws://{}:{}", host, port)
    async with websockets.serve(_connection_handler, host, port):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
