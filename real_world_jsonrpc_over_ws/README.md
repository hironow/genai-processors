JSON-RPC over WebSocket (real_world_jsonrpc_over_ws)

This example exposes a minimal JSON-RPC 2.0 API over WebSocket that drives the existing real_world live pipeline from this repo.

API
- Method `chat.process`: params `{ "messages": [{"role": "user|system|model", "content": "..."}, ...] }`
  - Streams notifications: `{"jsonrpc":"2.0","method":"chat.chunk","params":{"request_id":<id>,"event":{...}}}`
  - Final response: `{"jsonrpc":"2.0","id":<id>,"result":{"text":"..."}}`

Run
- Server: `python -m real_world_jsonrpc_over_ws.server`
- Client demo: `python -m real_world_jsonrpc_over_ws.client`

Notes
- Uses the same live pipeline as real_world/app, and filters reserved substreams (status/debug/caption).
- Text is aggregated from the default stream into `result.text` as a convenience.

