from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any
import argparse

import websockets
from loguru import logger


def _rpc_request(id_: Any, method: str, params: dict | None = None) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "method": method, "params": params or {}}


async def run_demo(
    url: str = "ws://127.0.0.1:8765",
    *,
    text: str = "Generative AI is transforming the world.",
    request_id: int = 1,
    suppress_whitespace: bool = False,
    coalesce_chars: int = 0,
    coalesce_time_ms: int = 0,
):
    logger.info(
        "Client connecting to {} (opts: suppress_ws={} coalesce_chars={} coalesce_time_ms={})",
        url,
        suppress_whitespace,
        coalesce_chars,
        coalesce_time_ms,
    )
    async with websockets.connect(url) as ws:
        logger.info("Client connected to {}", url)
        req = _rpc_request(
            request_id,
            "chat.process",
            {
                "messages": [
                    {"role": "user", "content": text}
                ],
                "options": {
                    "suppress_whitespace": bool(suppress_whitespace),
                    "coalesce_chars": int(coalesce_chars or 0),
                    "coalesce_time_ms": int(coalesce_time_ms or 0),
                },
            },
        )
        await ws.send(json.dumps(req, ensure_ascii=False))
        logger.info("-> chat.process sent id={} text_len={} ", request_id, len(text))

        # Receive notifications (chat.chunk) until final response with id==req_id
        while True:
            raw = await ws.recv()
            try:
                msg = json.loads(raw)
            except Exception:
                print(raw)
                continue
            if "id" in msg and msg.get("id") == request_id:
                if "error" in msg:
                    logger.error("<- error id={} {}", request_id, json.dumps(msg.get("error"), ensure_ascii=False))
                else:
                    logger.info("<- result id={} {}", request_id, json.dumps(msg.get("result"), ensure_ascii=False))
                break
            if msg.get("method") == "chat.chunk":
                evt = msg.get("params", {}).get("event", {})
                t = evt.get("type")
                if t == "text":
                    logger.debug("<- chunk(text) len={}", len(evt.get("text") or ""))
                else:
                    logger.debug("<- chunk({})", t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSON-RPC over WS demo client")
    parser.add_argument("--url", default="ws://127.0.0.1:8765", help="WebSocket URL")
    parser.add_argument("--text", default="Generative AI is transforming the world.", help="User message text")
    parser.add_argument("--request-id", type=int, default=1, help="JSON-RPC request id")
    parser.add_argument("--suppress-whitespace", action="store_true", help="Suppress whitespace-only text chunks")
    parser.add_argument("--coalesce-chars", type=int, default=0, help="Coalesce text chunks by this many characters (0=off)")
    parser.add_argument("--coalesce-time-ms", type=int, default=0, help="Coalesce text chunks by time in milliseconds (0=off)")
    args = parser.parse_args()
    asyncio.run(
        run_demo(
            args.url,
            text=args.text,
            request_id=args.request_id,
            suppress_whitespace=args.suppress_whitespace,
            coalesce_chars=args.coalesce_chars,
            coalesce_time_ms=args.coalesce_time_ms,
        )
    )


# Test-friendly helper that performs one chat.process and returns all chunks and final message.
async def rpc_chat(url: str, messages: list[dict[str, Any]], *, request_id: int = 1, timeout: float = 5.0):
    notes: list[dict[str, Any]] = []
    logger.debug("rpc_chat: connect {} id={}", url, request_id)
    async with websockets.connect(url) as ws:
        req = _rpc_request(request_id, "chat.process", {"messages": messages})
        await ws.send(json.dumps(req, ensure_ascii=False))
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError("timeout waiting for final JSON-RPC response")
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            if "id" in msg and msg.get("id") == request_id:
                return notes, msg
            if msg.get("method") == "chat.chunk":
                notes.append(msg)


async def rpc_send_ws(ws, id_: int, method: str, params: dict | None = None):
    req = _rpc_request(id_, method, params or {})
    await ws.send(json.dumps(req, ensure_ascii=False))


async def rpc_cancel_ws(ws, *, cancel_rpc_id: int, request_id: int):
    """Send chat.cancel on an existing WebSocket connection.

    Args:
      ws: an open websockets connection to the same server handling chat.process.
      cancel_rpc_id: JSON-RPC id for the cancellation request.
      request_id: the original chat.process request id to cancel.
    """
    logger.debug("rpc_cancel_ws: request cancel target_id={} id={}", request_id, cancel_rpc_id)
    await rpc_send_ws(ws, cancel_rpc_id, "chat.cancel", {"request_id": request_id})


async def rpc_tool_response_ws(ws, *, ack_id: int, request_id: int, name: str, response: dict, id: str | None = None):
    params = {"request_id": request_id, "name": name, "response": response}
    if id is not None:
        params["id"] = id
    logger.debug("rpc_tool_response_ws: send tool_response name={} target_id={} id={}", name, request_id, ack_id)
    await rpc_send_ws(ws, ack_id, "chat.tool_response", params)
