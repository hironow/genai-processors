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
