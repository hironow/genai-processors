from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
import websockets

from genai_processors import content_api, processor
from real_world_jsonrpc_over_ws import server as rpc_server
from real_world.agent import pipelines


# ---------- Test helpers ----------


async def _start_server(monkeypatch, stub_kind: str = "text", seq: list[str] | None = None, delay: float = 0.0):
    """Start a WS server with a stubbed live pipeline.

    stub_kind: 'text' | 'binary' | 'tool_call' | 'error'
    seq: for 'text', a list of text chunks to stream
    """

    # Build a stub live pipeline.
    if stub_kind == "text":
        chunks = list(seq or ["hello", " ", "world"])

        @processor.processor_function
        async def _stub_live(content):
            # Consume until end_of_turn, then emit chunks
            async for p in content:
                if p.metadata.get("turn_complete"):
                    break
            for s in chunks:
                if delay > 0:
                    await asyncio.sleep(delay)
                yield content_api.ProcessorPart(s, role="model")

        monkeypatch.setattr(pipelines, "build_live_pipeline", lambda *a, **k: _stub_live)

    elif stub_kind == "binary":

        @processor.processor_function
        async def _stub_live(content):
            async for p in content:
                if p.metadata.get("turn_complete"):
                    break
            yield content_api.ProcessorPart(b"\x89PNGtest", mimetype="image/png", role="model")

        monkeypatch.setattr(pipelines, "build_live_pipeline", lambda *a, **k: _stub_live)

    elif stub_kind == "tool_call":

        @processor.processor_function
        async def _stub_live(content):
            # Emit a tool_call first, then wait for a tool_response in input and respond with text.
            yield content_api.ProcessorPart.from_function_call(name="sum", args={"a": 1, "b": 2}, role="model")
            async for p in content:
                if p.part.function_response is not None and p.part.function_response.name == "sum":
                    v = p.part.function_response.response.get("value")
                    yield content_api.ProcessorPart(f"ok:{v}", role="model")
                    break

        monkeypatch.setattr(pipelines, "build_live_pipeline", lambda *a, **k: _stub_live)

    elif stub_kind == "error":

        @processor.processor_function
        async def _stub_live(content):
            async for p in content:
                if p.metadata.get("turn_complete"):
                    break
            # Force async-generator function shape while raising an error.
            if False:  # pragma: no cover
                yield content_api.ProcessorPart("unreached", role="model")
            raise RuntimeError("boom")

        monkeypatch.setattr(pipelines, "build_live_pipeline", lambda *a, **k: _stub_live)

    else:
        raise ValueError(f"unknown stub_kind: {stub_kind}")

    # Start server on ephemeral port.
    srv = await websockets.serve(rpc_server._connection_handler, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]
    url = f"ws://127.0.0.1:{port}"
    return srv, url


async def _rpc_roundtrip(url: str, req: dict[str, Any], *, timeout: float = 5.0):
    """Send a JSON-RPC request and collect notifications until final response.

    Returns: (notifications, final_message)
    """
    notes = []
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps(req))
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError("timeout waiting for final JSON-RPC response")
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=remaining))
            if "id" in msg and msg.get("id") == req.get("id"):
                return notes, msg
            notes.append(msg)


# ---------- Tests ----------


@pytest.mark.asyncio
async def test_rpc_ping_and_unknown_method(monkeypatch):
    # given: a server with a stub (unused)
    srv, url = await _start_server(monkeypatch, stub_kind="text", seq=["x"])
    try:
        # when: ping
        notes, resp = await _rpc_roundtrip(url, {"jsonrpc": "2.0", "id": 100, "method": "ping"})

        # then: no notifications, pong True
        assert notes == []
        assert resp.get("result", {}).get("pong") is True

        # when: unknown method
        notes2, resp2 = await _rpc_roundtrip(url, {"jsonrpc": "2.0", "id": 101, "method": "foo.bar"})

        # then: error -32601
        assert notes2 == []
        assert "error" in resp2
        assert resp2["error"].get("code") == -32601
    finally:
        srv.close()
        await srv.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("seq", [
    ["hello"],
    ["foo", "bar"],
    ["A", "B", "C"],
    ["", "x"],
])
async def test_chat_process_streaming_text(monkeypatch, seq):
    # given: server with text stub streaming given sequence
    srv, url = await _start_server(monkeypatch, stub_kind="text", seq=seq)
    try:
        req_id = 1
        req = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "chat.process",
            "params": {"messages": [{"role": "user", "content": "hi"}]},
        }
        # when: send request and collect notifications
        notes, resp = await _rpc_roundtrip(url, req)

        # then: received as many chunks as in seq, followed by final result
        #       each note is a JSON-RPC notification: method chat.chunk
        texts = [n.get("params", {}).get("event", {}).get("text") for n in notes if n.get("method") == "chat.chunk"]
        assert texts == seq
        assert "result" in resp and resp.get("id") == req_id
        assert resp["result"].get("text") == "".join(seq)
    finally:
        srv.close()
        await srv.wait_closed()


@pytest.mark.asyncio
async def test_chat_process_suppress_whitespace(monkeypatch):
    # given: server with whitespace and text mixed
    srv, url = await _start_server(monkeypatch, stub_kind="text", seq=["", "A", "  ", "B", "\n", " "])
    try:
        req_id = 41
        req = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "chat.process",
            "params": {
                "messages": [{"role": "user", "content": "hi"}],
                "options": {"suppress_whitespace": True},
            },
        }
        notes, resp = await _rpc_roundtrip(url, req)
        texts = [n.get("params", {}).get("event", {}).get("text") for n in notes if n.get("method") == "chat.chunk"]
        assert texts == ["A", "B"]
        assert resp.get("result", {}).get("text") == "AB"
    finally:
        srv.close()
        await srv.wait_closed()


@pytest.mark.asyncio
async def test_chat_process_coalesce_chars(monkeypatch):
    # given: small tokens, coalesce by 3 chars
    srv, url = await _start_server(monkeypatch, stub_kind="text", seq=["A", "B", "C", "D"])
    try:
        req_id = 42
        req = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "chat.process",
            "params": {
                "messages": [{"role": "user", "content": "hi"}],
                "options": {"coalesce_chars": 3},
            },
        }
        notes, resp = await _rpc_roundtrip(url, req)
        texts = [n.get("params", {}).get("event", {}).get("text") for n in notes if n.get("method") == "chat.chunk"]
        assert texts == ["ABC", "D"]
        assert resp.get("result", {}).get("text") == "ABCD"
    finally:
        srv.close()
        await srv.wait_closed()


@pytest.mark.asyncio
async def test_chat_process_coalesce_time_combine(monkeypatch):
    # given: small delay; time-based coalesce larger than delay, so combined
    srv, url = await _start_server(monkeypatch, stub_kind="text", seq=["A", "B", "C"], delay=0.01)
    try:
        req_id = 43
        req = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "chat.process",
            "params": {
                "messages": [{"role": "user", "content": "hi"}],
                "options": {"coalesce_time_ms": 50},
            },
        }
        notes, resp = await _rpc_roundtrip(url, req)
        texts = [n.get("params", {}).get("event", {}).get("text") for n in notes if n.get("method") == "chat.chunk"]
        # Expect a single combined chunk 'ABC' (timer fires once)
        assert texts in (["ABC"], ["AB", "C"])  # allow slight timing variance
        assert resp.get("result", {}).get("text") == "ABC"
    finally:
        srv.close()
        await srv.wait_closed()


@pytest.mark.asyncio
async def test_chat_process_coalesce_time_split(monkeypatch):
    # given: larger delay; time-based coalesce smaller than delay, so split each
    srv, url = await _start_server(monkeypatch, stub_kind="text", seq=["A", "B", "C"], delay=0.06)
    try:
        req_id = 44
        req = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "chat.process",
            "params": {
                "messages": [{"role": "user", "content": "hi"}],
                "options": {"coalesce_time_ms": 30},
            },
        }
        notes, resp = await _rpc_roundtrip(url, req)
        texts = [n.get("params", {}).get("event", {}).get("text") for n in notes if n.get("method") == "chat.chunk"]
        assert texts == ["A", "B", "C"]
        assert resp.get("result", {}).get("text") == "ABC"
    finally:
        srv.close()
        await srv.wait_closed()


@pytest.mark.asyncio
async def test_chat_process_streaming_binary(monkeypatch):
    # given: server with binary stub
    srv, url = await _start_server(monkeypatch, stub_kind="binary")
    try:
        req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "chat.process",
            "params": {"messages": [{"role": "user", "content": "image please"}]},
        }
        # when: call and collect
        notes, resp = await _rpc_roundtrip(url, req)

        # then: at least one chunk of type image with b64 data; final text empty
        evts = [n.get("params", {}).get("event", {}) for n in notes if n.get("method") == "chat.chunk"]
        assert evts and evts[0].get("type") == "image"
        assert isinstance(evts[0].get("data_b64"), str) and evts[0]["data_b64"]
        assert resp.get("result", {}).get("text") in ("", None)
    finally:
        srv.close()
        await srv.wait_closed()


@pytest.mark.asyncio
async def test_chat_process_streaming_tool_call_and_text(monkeypatch):
    # given: server with tool_call then waits for tool_response
    srv, url = await _start_server(monkeypatch, stub_kind="tool_call")
    try:
        req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "chat.process",
            "params": {"messages": [{"role": "user", "content": "calc"}]},
        }
        # when: connect manually to issue a tool_response mid-stream
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps(req))
            # Wait for tool_call notification
            tool_seen = False
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                if msg.get("method") == "chat.chunk":
                    evt = msg.get("params", {}).get("event", {})
                    if evt.get("type") == "tool_call" and evt.get("name") == "sum":
                        tool_seen = True
                        break
            assert tool_seen
            # send tool_response
            await ws.send(json.dumps({
                "jsonrpc": "2.0",
                "id": 333,
                "method": "chat.tool_response",
                "params": {"request_id": 3, "name": "sum", "response": {"value": 3}},
            }))
            # then: expect ack and final result ok:3
            got_ack = False
            final = None
            for _ in range(50):
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                if msg.get("id") == 333:
                    assert msg.get("result", {}).get("accepted") is True
                    got_ack = True
                if msg.get("id") == 3:
                    final = msg
                    break
            assert got_ack and final and final.get("result", {}).get("text") == "ok:3"
    finally:
        srv.close()
        await srv.wait_closed()


@pytest.mark.asyncio
async def test_chat_process_error_bubbles_as_rpc_error(monkeypatch):
    # given: server with erroring stub
    srv, url = await _start_server(monkeypatch, stub_kind="error")
    try:
        req = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "chat.process",
            "params": {"messages": [{"role": "user", "content": "boom"}]},
        }
        # when
        notes, resp = await _rpc_roundtrip(url, req)

        # then: no final result; JSON-RPC error with generic server code
        assert "error" in resp
        assert resp["error"].get("code") == -32000
        assert "chat.process failed" in resp["error"].get("message", "")
    finally:
        srv.close()
        await srv.wait_closed()


# ---------- Additional exception/edge-case coverage ----------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_params",
    [
        None,
        {},
        {"messages": "not-a-list"},
        {"messages": [1, 2]},
        {"messages": [{"role": "user"}]},
        {"messages": [{"content": "hi"}]},
        {"messages": [{}]},
    ],
)
async def test_chat_process_invalid_params_return_rpc_error(monkeypatch, bad_params):
    # given: server up with any stub
    srv, url = await _start_server(monkeypatch, stub_kind="text", seq=["ok"])
    try:
        req = {"jsonrpc": "2.0", "id": 10, "method": "chat.process"}
        if bad_params is not None:
            req["params"] = bad_params
        # when: call with invalid params
        notes, resp = await _rpc_roundtrip(url, req)
        # then: error -32000
        assert notes == [] or all(n.get("method") == "chat.chunk" for n in notes)  # no stray responses
        assert "error" in resp and resp["error"]["code"] == -32000
    finally:
        srv.close()
        await srv.wait_closed()


@pytest.mark.asyncio
async def test_invalid_frames_then_valid_request(monkeypatch):
    # given: server up
    srv, url = await _start_server(monkeypatch, stub_kind="text", seq=["x"])
    try:
        # when: send non-JSON and wrong jsonrpc version, then a valid ping
        async with websockets.connect(url) as ws:
            await ws.send("not-json")
            await ws.send(json.dumps({"jsonrpc": "1.0", "id": 11, "method": "ping"}))
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": 12, "method": "ping"}))
            msg = json.loads(await ws.recv())
        # then: server should ignore first two and answer the valid one
        assert msg.get("id") == 12 and msg.get("result", {}).get("pong") is True
    finally:
        srv.close()
        await srv.wait_closed()


@pytest.mark.asyncio
async def test_reserved_substreams_are_filtered(monkeypatch):
    # given: stub that emits a reserved-status part and a default text
    @processor.processor_function
    async def _stub_live(content):
        async for p in content:
            if p.metadata.get("turn_complete"):
                break
        yield content_api.ProcessorPart("status log", role="model", substream_name=processor.STATUS_STREAM)
        yield content_api.ProcessorPart("ok", role="model")

    monkeypatch.setattr(pipelines, "build_live_pipeline", lambda *a, **k: _stub_live)

    srv = await websockets.serve(rpc_server._connection_handler, "127.0.0.1", 0)
    try:
        port = srv.sockets[0].getsockname()[1]
        url = f"ws://127.0.0.1:{port}"
        req = {"jsonrpc": "2.0", "id": 13, "method": "chat.process", "params": {"messages": [{"role": "user", "content": "hi"}]}}
        notes, resp = await _rpc_roundtrip(url, req)
        # then: only the default text made it to notifications; no status chunk
        evts = [n.get("params", {}).get("event", {}) for n in notes if n.get("method") == "chat.chunk"]
        assert any(e.get("type") == "text" and e.get("text") == "ok" for e in evts)
        assert not any(e.get("type") == "text" and e.get("text") == "status log" for e in evts)
    finally:
        srv.close()
        await srv.wait_closed()


@pytest.mark.asyncio
async def test_metadata_is_json_sanitized(monkeypatch):
    # given: stub that emits text with non-serializable metadata (dataclass)
    import dataclasses

    @dataclasses.dataclass
    class Meta:
        a: int

    @processor.processor_function
    async def _stub_live(content):
        async for p in content:
            if p.metadata.get("turn_complete"):
                break
        yield content_api.ProcessorPart("x", role="model", metadata={"meta": Meta(1)})

    monkeypatch.setattr(pipelines, "build_live_pipeline", lambda *a, **k: _stub_live)

    srv = await websockets.serve(rpc_server._connection_handler, "127.0.0.1", 0)
    try:
        port = srv.sockets[0].getsockname()[1]
        url = f"ws://127.0.0.1:{port}"
        req = {"jsonrpc": "2.0", "id": 14, "method": "chat.process", "params": {"messages": [{"role": "user", "content": "hi"}]}}
        notes, resp = await _rpc_roundtrip(url, req)
        # then: the metadata in the chunk is JSON-friendly (dict with 'meta': {'a': 1})
        evts = [n.get("params", {}).get("event", {}) for n in notes if n.get("method") == "chat.chunk"]
        assert evts and isinstance(evts[0].get("metadata", {}), dict)
        assert evts[0]["metadata"].get("meta") in ({"a": 1}, "Meta(a=1)")
    finally:
        srv.close()
        await srv.wait_closed()


@pytest.mark.asyncio
async def test_concurrent_requests_same_connection(monkeypatch):
    # given: stub that emits two short chunks per request
    @processor.processor_function
    async def _stub_live(content):
        async for p in content:
            if p.metadata.get("turn_complete"):
                break
        yield content_api.ProcessorPart("A", role="model")
        yield content_api.ProcessorPart("B", role="model")

    monkeypatch.setattr(pipelines, "build_live_pipeline", lambda *a, **k: _stub_live)

    srv = await websockets.serve(rpc_server._connection_handler, "127.0.0.1", 0)
    try:
        port = srv.sockets[0].getsockname()[1]
        url = f"ws://127.0.0.1:{port}"
        async with websockets.connect(url) as ws:
            # when: send two requests quickly
            req1 = {"jsonrpc": "2.0", "id": 21, "method": "chat.process", "params": {"messages": [{"role": "user", "content": "hi1"}]}}
            req2 = {"jsonrpc": "2.0", "id": 22, "method": "chat.process", "params": {"messages": [{"role": "user", "content": "hi2"}]}}
            await ws.send(json.dumps(req1))
            await ws.send(json.dumps(req2))

            chunks: dict[int, list[str]] = {21: [], 22: []}
            results: dict[int, str] = {}
            # Collect until both results arrive
            for _ in range(20):
                msg = json.loads(await ws.recv())
                if msg.get("method") == "chat.chunk":
                    rid = msg.get("params", {}).get("request_id")
                    evt = msg.get("params", {}).get("event", {})
                    if rid in chunks and evt.get("type") == "text":
                        chunks[rid].append(evt.get("text"))
                elif "id" in msg:
                    results[msg["id"]] = (msg.get("result", {}) or {}).get("text", "")
                if 21 in results and 22 in results:
                    break

        # then: both requests completed with concatenated text
        assert results.get(21) == "AB"
        assert results.get(22) == "AB"
        assert chunks[21] and chunks[22]
    finally:
        srv.close()
        await srv.wait_closed()


@pytest.mark.asyncio
async def test_missing_id_returns_null_id(monkeypatch):
    # given
    srv, url = await _start_server(monkeypatch, stub_kind="text", seq=["x"])
    try:
        # when: send request without id
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps({
                "jsonrpc": "2.0",
                "method": "chat.process",
                "params": {"messages": [{"role": "user", "content": "hi"}]},
            }))
            # gather a couple of frames, final one should have id: null
            final = None
            for _ in range(10):
                msg = json.loads(await ws.recv())
                if "id" in msg:
                    final = msg
                    break
        # then
        assert final is not None and final.get("id") is None
    finally:
        srv.close()
        await srv.wait_closed()


@pytest.mark.asyncio
async def test_chat_cancel_stops_stream_and_errors_original(monkeypatch):
    # given: long-running stub that yields chunks with delay
    @processor.processor_function
    async def _stub_live(content):
        async for p in content:
            if p.metadata.get("turn_complete"):
                break
        for i in range(50):
            await asyncio.sleep(0.02)
            yield content_api.ProcessorPart(f"{i}", role="model")

    monkeypatch.setattr(pipelines, "build_live_pipeline", lambda *a, **k: _stub_live)

    srv = await websockets.serve(rpc_server._connection_handler, "127.0.0.1", 0)
    try:
        port = srv.sockets[0].getsockname()[1]
        url = f"ws://127.0.0.1:{port}"
        async with websockets.connect(url) as ws:
            # when: send a chat request
            chat_id = 31
            await ws.send(json.dumps({
                "jsonrpc": "2.0",
                "id": chat_id,
                "method": "chat.process",
                "params": {"messages": [{"role": "user", "content": "stream"}]},
            }))
            # Receive first chunk, then cancel
            seen_chunks = 0
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("method") == "chat.chunk":
                    seen_chunks += 1
                    break
            await ws.send(json.dumps({
                "jsonrpc": "2.0",
                "id": 991,
                "method": "chat.cancel",
                "params": {"request_id": chat_id},
            }))

            got_cancel_ack = False
            got_cancel_error = False
            # then: receive cancel ack and an error for the original request
            for _ in range(100):
                msg = json.loads(await ws.recv())
                if msg.get("id") == 991:
                    assert msg.get("result", {}).get("cancelled") is True
                    got_cancel_ack = True
                elif msg.get("id") == chat_id and "error" in msg:
                    assert msg["error"].get("code") == -32800
                    got_cancel_error = True
                    break
            assert got_cancel_ack and got_cancel_error
            assert seen_chunks >= 1
    finally:
        srv.close()
        await srv.wait_closed()


@pytest.mark.asyncio
async def test_chat_cancel_unknown_id_is_ok_false(monkeypatch):
    # given
    srv, url = await _start_server(monkeypatch, stub_kind="text", seq=["x"])
    try:
        # when: cancel non-existent
        notes, resp = await _rpc_roundtrip(url, {
            "jsonrpc": "2.0", "id": 992, "method": "chat.cancel", "params": {"request_id": 4242}
        })
        # then: ok result with cancelled False
        assert notes == []
        assert resp.get("result", {}).get("cancelled") is False
        assert resp.get("result", {}).get("request_id") == 4242
    finally:
        srv.close()
        await srv.wait_closed()


@pytest.mark.asyncio
async def test_chat_cancel_invalid_params_error(monkeypatch):
    # given
    srv, url = await _start_server(monkeypatch, stub_kind="text", seq=["x"])
    try:
        # when: cancel without request_id
        notes, resp = await _rpc_roundtrip(url, {
            "jsonrpc": "2.0", "id": 993, "method": "chat.cancel", "params": {}
        })
        # then: JSON-RPC invalid params
        assert notes == []
        assert "error" in resp and resp["error"].get("code") == -32602
    finally:
        srv.close()
        await srv.wait_closed()
