JSON-RPC over WebSocket (real_world_jsonrpc_over_ws)

This example exposes a minimal JSON-RPC 2.0 API over WebSocket that drives the existing real_world live pipeline from this repo.

API
- Methods (allowed)
  - `ping`
    - Params: none
    - Result: `{ "pong": true }`
  - `chat.process`
    - Params: `{ "messages": [{"role": "user|system|model", "content": "..."}, ...], "options"?: { "suppress_whitespace"?: bool, "coalesce_chars"?: number, "coalesce_time_ms"?: number } }`
    - Notifications (0..N): `{"jsonrpc":"2.0","method":"chat.chunk","params":{"request_id":<id>,"event":{...}}}`
    - Final response: `{"jsonrpc":"2.0","id":<id>,"result":{"text":"..."}}`
  - `chat.cancel`
    - Params: `{ "request_id": <id of chat.process> }`
    - Result: `{ "cancelled": true|false, "request_id": <id> }`
    - Side effect: The original `chat.process` ends with error `{"code": -32800, "message": "Request cancelled"}`.
  - `chat.tool_response`
    - Params: `{ "request_id": <id>, "name": <str>, "response": <object>, "id"?: <string> }`
    - Result: `{ "accepted": true|false, "request_id": <id> }`
    - Effect: Injects a tool response into the running `chat.process` stream and continues the turn.

Examples
- Tool call → tool response roundtrip
  1) Client sends chat.process:
     `{ "jsonrpc":"2.0","id":1,"method":"chat.process","params":{"messages":[{"role":"user","content":"calc"}]}}`
  2) Server notifies tool call:
     `{ "jsonrpc":"2.0","method":"chat.chunk","params":{"request_id":1,"event":{"type":"tool_call","name":"sum","args":{"a":1,"b":2}}}}`
  3) Client responds with tool output:
     `{ "jsonrpc":"2.0","id":2,"method":"chat.tool_response","params":{"request_id":1,"name":"sum","response":{"value":3}} }`
  4) Server acks and then completes original request:
     `{"jsonrpc":"2.0","id":2,"result":{"accepted":true,"request_id":1}}`
     `{"jsonrpc":"2.0","id":1,"result":{"text":"ok:3"}}`


Run
- Server: `python -m real_world_jsonrpc_over_ws.server`
- Client demo: `python -m real_world_jsonrpc_over_ws.client`
  - Flags (coalescer control):
    - `--suppress-whitespace` — drop whitespace-only text chunks
    - `--coalesce-chars N` — coalesce default-stream text roughly every N chars
    - `--coalesce-time-ms N` — coalesce text by time (ms)
  - Example:
    - `python -m real_world_jsonrpc_over_ws.client --coalesce-chars 32`
    - `python -m real_world_jsonrpc_over_ws.client --suppress-whitespace --coalesce-time-ms 50`

Notes
- Uses the same live pipeline as real_world/app, and filters reserved substreams (status/debug/caption).
- Text is aggregated from the default stream into `result.text` as a convenience.
- Stream options:
  - `suppress_whitespace` (bool, default false): ignore whitespace-only text chunks.
  - `coalesce_chars` (int, default 0): when >0, coalesces consecutive default-stream text chunks and sends combined chunks roughly that size (also flushes on newline/end).
  - `coalesce_time_ms` (int, default 0): when >0, flushes the coalescer periodically by time.

Coalescer (unique behavior)
- Goal: reduce chatty token-by-token updates while keeping low perceived latency.
- Inputs affected: only default-stream text (non-reserved, `substream==""`).
- Flush triggers:
  - Size: total buffered text length reaches `coalesce_chars`.
  - Time: `coalesce_time_ms` elapses since the last flush (if >0).
  - Newline: any incoming chunk contains a newline (`\n`).
  - Event boundary: a non-text event (e.g., image/tool_call) arrives.
  - Completion: request ends (success/cancel/error) — pending buffer is flushed once.
- What it emits: one `chat.chunk` with the combined text. Aggregation (`result.text`) uses the same combined pieces, so `result.text` is the exact concatenation of what was sent after coalescing.
- Tradeoffs:
  - Larger `coalesce_chars` or `coalesce_time_ms` lowers message rate but increases burst size/latency.
  - If both are set, either trigger can flush (whichever happens first).
  - Set both to 0 to preserve original token-by-token behavior.

Other notable behaviors
- JSON-safe metadata: any rich objects attached by the pipeline are sanitized into JSON-friendly forms before sending.
- Missing id: requests without `id` are accepted; final response uses `"id": null` (permissible in JSON-RPC 2.0 servers).
- Concurrency: multiple `chat.process` calls can run concurrently per connection; `chat.cancel` targets one by `request_id`.
- Tool bridging: `chat.tool_response` injects a tool function response and advances the turn, enabling iterative tool use workflows.
