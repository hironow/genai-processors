"""JSON-RPC over WebSocket example that drives GenAI Processors.

This package exposes a small server and client to demonstrate how to wrap the
existing "real_world" live pipeline behind a JSON-RPC 2.0 protocol over a
WebSocket transport. The server accepts a `chat.process` method with
OpenAI-like messages and streams model outputs back as JSON-RPC notifications
(`chat.chunk`), followed by a final result.
"""

