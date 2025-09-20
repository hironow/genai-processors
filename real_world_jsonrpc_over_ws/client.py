from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any

import websockets


def _rpc_request(id_: Any, method: str, params: dict | None = None) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "method": method, "params": params or {}}


async def run_demo(url: str = "ws://127.0.0.1:8765"):
    async with websockets.connect(url) as ws:
        req_id = 1
        req = _rpc_request(
            req_id,
            "chat.process",
            {
                "messages": [
                    {"role": "user", "content": "Summarize: Generative AI is transforming the world."}
                ]
            },
        )
        await ws.send(json.dumps(req, ensure_ascii=False))
        print(f"-> {json.dumps(req, ensure_ascii=False)}")

        # Receive notifications (chat.chunk) until final response with id==req_id
        while True:
            raw = await ws.recv()
            try:
                msg = json.loads(raw)
            except Exception:
                print(raw)
                continue
            if "id" in msg and msg.get("id") == req_id:
                if "error" in msg:
                    print("<- error:", json.dumps(msg.get("error"), ensure_ascii=False, indent=2))
                else:
                    print("<- result:", json.dumps(msg.get("result"), ensure_ascii=False, indent=2))
                break
            if msg.get("method") == "chat.chunk":
                evt = msg.get("params", {}).get("event", {})
                t = evt.get("type")
                if t == "text":
                    print(f"<- chunk(text): {evt.get('text')}")
                else:
                    print(f"<- chunk({t}): {json.dumps(evt)[:200]}")


if __name__ == "__main__":
    asyncio.run(run_demo())


# Test-friendly helper that performs one chat.process and returns all chunks and final message.
async def rpc_chat(url: str, messages: list[dict[str, Any]], *, request_id: int = 1, timeout: float = 5.0):
    notes: list[dict[str, Any]] = []
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
    await rpc_send_ws(ws, cancel_rpc_id, "chat.cancel", {"request_id": request_id})


async def rpc_tool_response_ws(ws, *, ack_id: int, request_id: int, name: str, response: dict, id: str | None = None):
    params = {"request_id": request_id, "name": name, "response": response}
    if id is not None:
        params["id"] = id
    await rpc_send_ws(ws, ack_id, "chat.tool_response", params)
