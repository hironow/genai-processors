from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import asdict, dataclass
import contextlib
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
    """Retained for compatibility; delegates to a queue-backed generator."""
    q: asyncio.Queue[content_api.ProcessorPart | None] = asyncio.Queue()
    for m in messages:
        q.put_nowait(content_api.ProcessorPart(m.content, role=m.role))
    q.put_nowait(content_api.ProcessorPart.end_of_turn())
    return _process_stream_from_queue(q)


def _process_stream_from_queue(
    in_q: "asyncio.Queue[content_api.ProcessorPart | None]",
) -> AsyncIterable[content_api.ProcessorPart]:
    """Run the live pipeline, sourcing inputs from a queue so we can inject tool responses."""
    live = pipelines.build_live_pipeline()

    async def _in_stream():
        while True:
            item = await in_q.get()
            if item is None:
                break
            yield item

    async def _run():
        async with gp_context.context(reserved_substreams=("debug", "status", "caption")):
            async for p in live(_in_stream()):
                yield p

    return _run()


@dataclass
class _InflightReq:
    task: asyncio.Task
    in_q: "asyncio.Queue[content_api.ProcessorPart | None]"


async def _handle_chat_process(
    request_id: Any,
    params: dict,
    websocket,
    send_lock: asyncio.Lock,
    in_q: "asyncio.Queue[content_api.ProcessorPart | None]",
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

        # Optional stream options
        opts = params.get("options") or {}
        suppress_ws = bool(opts.get("suppress_whitespace", False))
        try:
            coalesce_chars = int(opts.get("coalesce_chars", 0) or 0)
        except Exception:
            coalesce_chars = 0
        try:
            coalesce_time_ms = int(opts.get("coalesce_time_ms", 0) or 0)
        except Exception:
            coalesce_time_ms = 0

        # Aggregate plain-text on default stream for convenience result.text
        agg_text: list[str] = []
        coalesce_enabled = (coalesce_chars > 0) or (coalesce_time_ms > 0)
        pending_text: list[str] = [] if coalesce_enabled else None  # type: ignore[assignment]
        _buf_lock = asyncio.Lock()
        _stop_timer = asyncio.Event()

        async def _flush_pending():
            if not coalesce_enabled:
                return
            async with _buf_lock:
                if pending_text and len(pending_text) > 0:
                    combined = "".join(pending_text)
                    pending_text.clear()
                else:
                    return
            # Update aggregate and notify (outside lock for send)
            agg_text.append(combined)
            note = _rpc_notification(
                "chat.chunk",
                {
                    "request_id": request_id,
                    "event": {
                        "type": "text",
                        "role": "model",
                        "text": combined,
                        "mimetype": "text/plain",
                        "metadata": {},
                    },
                },
            )
            async with send_lock:
                await websocket.send(json.dumps(note, ensure_ascii=False))

        # Periodic flush if time-based coalescing is enabled
        timer_task: asyncio.Task | None = None
        if coalesce_enabled and coalesce_time_ms > 0:
            interval = max(1, coalesce_time_ms) / 1000.0

            async def _periodic():
                try:
                    while not _stop_timer.is_set():
                        await asyncio.sleep(interval)
                        if _stop_timer.is_set():
                            break
                        await _flush_pending()
                except asyncio.CancelledError:
                    pass

            timer_task = asyncio.create_task(_periodic())

        # Prime the input queue with initial messages + end_of_turn
        for m in msgs:
            in_q.put_nowait(content_api.ProcessorPart(m.content, role=m.role))
        in_q.put_nowait(content_api.ProcessorPart.end_of_turn())

        async for part in _process_stream_from_queue(in_q):
            evt = _part_to_event(part)
            if evt is None:
                # Reserved stream (status/debug/caption)
                continue
            # Accumulate only default stream text
            if evt.get("type") == "text" and (part.substream_name or "") == "":
                t = evt.get("text", "") or ""
                if suppress_ws and t.strip() == "":
                    # skip whitespace-only chunk
                    continue
                if coalesce_enabled:
                    async with _buf_lock:
                        pending_text.append(t)
                        size = sum(len(s) for s in pending_text)
                    if (coalesce_chars > 0 and size >= coalesce_chars) or ("\n" in t):
                        await _flush_pending()
                else:
                    agg_text.append(t)
                    note = _rpc_notification(
                        "chat.chunk",
                        {"request_id": request_id, "event": evt},
                    )
                    async with send_lock:
                        await websocket.send(json.dumps(note, ensure_ascii=False))
            else:
                # Non-text event: flush pending coalesced text first
                if coalesce_enabled:
                    await _flush_pending()
                note = _rpc_notification(
                    "chat.chunk",
                    {"request_id": request_id, "event": evt},
                )
                async with send_lock:
                    await websocket.send(json.dumps(note, ensure_ascii=False))

        # Flush any pending coalesced text, then final result
        if coalesce_enabled:
            # stop timer first to avoid race
            _stop_timer.set()
            try:
                if timer_task:
                    timer_task.cancel()
                    with contextlib.suppress(Exception):
                        await timer_task
            except Exception:
                pass
            await _flush_pending()
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
        # Ensure timer is stopped
        try:
            _stop_timer.set()
            if 'timer_task' in locals() and timer_task:
                timer_task.cancel()
                with contextlib.suppress(Exception):
                    await timer_task
        except Exception:
            pass
        return
    except Exception as e:
        logger.exception("chat.process failed")
        err = _rpc_response_error(request_id, -32000, "chat.process failed", {"detail": str(e)})
        async with send_lock:
            await websocket.send(json.dumps(err, ensure_ascii=False))
        # Ensure timer is stopped on error
        try:
            _stop_timer.set()
            if 'timer_task' in locals() and timer_task:
                timer_task.cancel()
                with contextlib.suppress(Exception):
                    await timer_task
        except Exception:
            pass


async def _connection_handler(websocket):
    logger.info("WS JSON-RPC: client connected: {}", getattr(websocket, "remote_address", None))
    send_lock = asyncio.Lock()
    inflight: dict[Any, _InflightReq] = {}
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
                in_q: asyncio.Queue[content_api.ProcessorPart | None] = asyncio.Queue()
                task = asyncio.create_task(_handle_chat_process(req_id, params, websocket, send_lock, in_q))
                if req_id is not None:
                    inflight[req_id] = _InflightReq(task=task, in_q=in_q)
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
                inf = inflight.get(target_id)
                had = inf is not None
                if had and inf.task:
                    inf.task.cancel()
                async with send_lock:
                    await websocket.send(json.dumps(_rpc_response_ok(req_id, {"cancelled": bool(had), "request_id": target_id})))
                continue

            if method == "chat.tool_response":
                # params: {request_id, name, response, id?}
                if not isinstance(params, dict):
                    async with send_lock:
                        await websocket.send(json.dumps(_rpc_response_error(req_id, -32602, "Invalid params")))
                    continue
                target_id = params.get("request_id")
                name = params.get("name")
                response = params.get("response")
                fr_id = params.get("id")
                if target_id is None or not isinstance(name, str) or not isinstance(response, dict):
                    async with send_lock:
                        await websocket.send(json.dumps(_rpc_response_error(req_id, -32602, "Invalid params: request_id,name,response required")))
                    continue
                inf = inflight.get(target_id)
                accepted = False
                if inf is not None:
                    try:
                        part = content_api.ProcessorPart.from_function_response(name=name, response=response, function_call_id=fr_id)
                        inf.in_q.put_nowait(part)
                        # allow model to continue
                        inf.in_q.put_nowait(content_api.ProcessorPart.end_of_turn())
                        accepted = True
                    except Exception:
                        logger.exception("failed to queue tool_response")
                        accepted = False
                async with send_lock:
                    await websocket.send(json.dumps(_rpc_response_ok(req_id, {"accepted": accepted, "request_id": target_id})))
                continue

            # Unknown method
            async with send_lock:
                await websocket.send(json.dumps(_rpc_response_error(req_id, -32601, f"Method not found: {method}")))

    except websockets.exceptions.ConnectionClosed:
        logger.info("WS JSON-RPC: connection closed")
    finally:
        # Cancel any remaining tasks
        for inf in list(inflight.values()):
            try:
                inf.task.cancel()
                # Unblock input generator
                inf.in_q.put_nowait(None)
            except Exception:
                pass
        if inflight:
            await asyncio.gather(*[inf.task for inf in inflight.values()], return_exceptions=True)


async def main(host: str = "127.0.0.1", port: int = 8765):
    load_dotenv()
    logger.info("Starting JSON-RPC over WS server on ws://{}:{}", host, port)
    async with websockets.serve(_connection_handler, host, port):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
