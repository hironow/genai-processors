from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
import websockets

from genai_processors import content_api, processor
from real_world_jsonrpc_over_ws import server as rpc_server
from real_world_jsonrpc_over_ws import client as rpc_client
from real_world.agent import pipelines


async def _start_server(monkeypatch, *, stub_kind: str = "text", seq: list[str] | None = None):
    if stub_kind == "text":
        chunks = list(seq or ["X"])  # default single chunk

        @processor.processor_function
        async def _stub_live(content):
            async for p in content:
                if p.metadata.get("turn_complete"):
                    break
            for s in chunks:
                yield content_api.ProcessorPart(s, role="model")

        monkeypatch.setattr(pipelines, "build_live_pipeline", lambda *a, **k: _stub_live)

    elif stub_kind == "error":

        @processor.processor_function
        async def _stub_live(content):
            async for p in content:
                if p.metadata.get("turn_complete"):
                    break
            if False:  # keep async-gen shape
                yield content_api.ProcessorPart("unreached", role="model")
            raise RuntimeError("boom")

        monkeypatch.setattr(pipelines, "build_live_pipeline", lambda *a, **k: _stub_live)
    else:
        raise ValueError("unsupported stub_kind")

    srv = await websockets.serve(rpc_server._connection_handler, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]
    url = f"ws://127.0.0.1:{port}"
    return srv, url


@pytest.mark.asyncio
@pytest.mark.parametrize("seq", [["hello"], ["foo", "bar"], ["A", "", "B"]])
async def test_client_e2e_text_stream(monkeypatch, seq):
    # given: text stub and client using rpc_chat helper
    srv, url = await _start_server(monkeypatch, stub_kind="text", seq=seq)
    try:
        # when
        notes, final = await rpc_client.rpc_chat(url, messages=[{"role": "user", "content": "hi"}], request_id=7)
        # then
        texts = [n.get("params", {}).get("event", {}).get("text") for n in notes]
        assert texts == seq
        assert final.get("id") == 7 and "result" in final
        assert final["result"].get("text") == "".join(seq)
    finally:
        srv.close()
        await srv.wait_closed()


@pytest.mark.asyncio
async def test_client_e2e_server_error(monkeypatch):
    # given: erroring stub and client using rpc_chat helper
    srv, url = await _start_server(monkeypatch, stub_kind="error")
    try:
        # when
        notes, final = await rpc_client.rpc_chat(url, messages=[{"role": "user", "content": "boom"}], request_id=8)
        # then: no result, but error present
        assert final.get("id") == 8 and "error" in final
        assert final["error"].get("code") == -32000
    finally:
        srv.close()
        await srv.wait_closed()


@pytest.mark.asyncio
async def test_client_e2e_cancel_via_client_helper(monkeypatch):
    # given: slow streaming stub
    @processor.processor_function
    async def _stub_live(content):
        # stop consuming at end_of_turn to avoid hanging
        async for p in content:
            if p.metadata.get("turn_complete"):
                break
        for i in range(100):
            await asyncio.sleep(0.01)
            yield content_api.ProcessorPart(f"{i}", role="model")

    monkeypatch.setattr(pipelines, "build_live_pipeline", lambda *a, **k: _stub_live)

    srv = await websockets.serve(rpc_server._connection_handler, "127.0.0.1", 0)
    try:
        port = srv.sockets[0].getsockname()[1]
        url = f"ws://127.0.0.1:{port}"
        async with websockets.connect(url) as ws:
            chat_id = 501
            await rpc_client.rpc_send_ws(ws, chat_id, "chat.process", {"messages": [{"role": "user", "content": "stream"}]})
            # when: wait first chunk then cancel using client helper
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                if msg.get("method") == "chat.chunk":
                    break
            await rpc_client.rpc_cancel_ws(ws, cancel_rpc_id=777, request_id=chat_id)

            # then: expect cancel ack and original error
            got_ack = False
            got_cancel_error = False
            for _ in range(200):
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                if msg.get("id") == 777 and msg.get("result", {}).get("cancelled") is not None:
                    got_ack = True
                if msg.get("id") == chat_id and "error" in msg:
                    assert msg["error"].get("code") == -32800
                    got_cancel_error = True
                    break
            assert got_ack and got_cancel_error
    finally:
        srv.close()
        await srv.wait_closed()


@pytest.mark.asyncio
async def test_client_e2e_tool_roundtrip_via_client_helper(monkeypatch):
    # given: stub emits tool_call then waits for tool_response and replies with ok:value
    @processor.processor_function
    async def _stub_live(content):
        yield content_api.ProcessorPart.from_function_call(name="sum", args={"a": 1, "b": 2}, role="model")
        async for p in content:
            if p.part.function_response is not None and p.part.function_response.name == "sum":
                v = p.part.function_response.response.get("value")
                yield content_api.ProcessorPart(f"ok:{v}", role="model")
                break

    monkeypatch.setattr(pipelines, "build_live_pipeline", lambda *a, **k: _stub_live)

    srv = await websockets.serve(rpc_server._connection_handler, "127.0.0.1", 0)
    try:
        port = srv.sockets[0].getsockname()[1]
        url = f"ws://127.0.0.1:{port}"
        async with websockets.connect(url) as ws:
            chat_id = 601
            await rpc_client.rpc_send_ws(ws, chat_id, "chat.process", {"messages": [{"role": "user", "content": "use tool"}]})
            # wait for tool_call
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                if msg.get("method") == "chat.chunk":
                    evt = msg.get("params", {}).get("event", {})
                    if evt.get("type") == "tool_call" and evt.get("name") == "sum":
                        break
            # when: reply with tool_response via helper
            await rpc_client.rpc_tool_response_ws(ws, ack_id=888, request_id=chat_id, name="sum", response={"value": 3})

            # then: expect ack and final result ok:3
            got_ack = False
            final = None
            for _ in range(50):
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                if msg.get("id") == 888:
                    assert msg.get("result", {}).get("accepted") is True
                    got_ack = True
                if msg.get("id") == chat_id:
                    final = msg
                    break
            assert got_ack and final and final.get("result", {}).get("text") == "ok:3"
    finally:
        srv.close()
        await srv.wait_closed()
