# Streaming Architecture (WS)

This document describes the WebSocket streaming architecture after the split into a mainline (human↔AI conversation) and a monitor (observer) stream, and outlines future extensions.

## High‑Level Flow (ASCII)

```
Client (WS)
  │  JSON msgs (text/image/audio/video/tool_response/config/end_of_turn)
  ▼
Server: real_world/app/main.py
  ┌──────────────────────────────────────────────────────────────────────┐
  │                        Input Ingest (consume_ws)                     │
  │  WS → in_q (asyncio.Queue[ProcessorPart])                            │
  └──────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
                     input_stream = _iter_to_async_iter(in_q)
                             │
              streams.split(n=2, with_copy=True)
             ┌───────────────────────┴───────────────────────┐
             ▼                                               ▼
        chat_in (mainline)                             monitor_in (observer)

  ┌─────────────────────────┐          ┌─────────────────────────────────────┐
  │ Control Channel         │          │ Monitor Pipeline                    │
  │ control_q (Queue)       │          │ metrics → dedupe → size_limit →     │
  │ control_stream=dequeue  │          │   caption (MediaCaptioner)          │
  └──────────┬──────────────┘          │                                     │
             │                         │  - Emits reserved logs:             │
             │                         │      status/debug/caption           │
             │                         │  - Emits default-stream text w/     │
             │                         │    metadata.custom_event (e.g.      │
             │                         │    type="caption")                  │
             │                         └───────────┬─────────────────────────┘
             │                                     │
             │                         custom_event on default stream? ── Yes ─►
             │                                     │   inject into control_q:
             │                                     │   - text (overrideable)
             │                                     │   - role (default user)
             │                                     │   - optional EOT
             │                                     ▼
             │                         control_q.put_nowait(event / EOT)
             │
             ▼
  main_in = merge([ filter(chat_in), control_stream ])
    where filter(chat_in) ≈ allow { text, end_of_turn, tool_response }

  ┌──────────────────────────────────────┐
  │ Mainline Pipeline (conversation)     │
  │  LiveProcessor(turn_model)           │
  └──────────────────────────────────────┘
                             │
                             ▼
                      main_out (AsyncIterable)
                             │
             streams.split(n=2, with_copy=True)
             ┌───────────────────────┴─────────────────────────┐
             ▼                                                 ▼
      main_to_client                                     main_to_monitor

  ┌─────────────────────────┐                 ┌───────────────────────────────┐
  │ Client Sender           │                 │ Monitor Mirror                │
  │ send_json(payload)      │◄──── reserved ─►│ [model_out] … as status logs  │
  └─────────────────────────┘                 └───────┬───────────────────────┘
                                                      │
                      monitor_stream = merge([ monitor_pipeline(monitor_in),
                                                mirror(main_to_monitor) ])
                                                      │
                                                      ▼
                                   forward only reserved substreams to client
                                   (status / debug / caption)
```

Notes
- Reserved substreams (`debug`, `status`, `caption`) are forwarded to the client immediately and never enter the model.
- Default substream (`""`) is the mainline. Monitor’s default-stream text with `metadata.custom_event` is not sent to the client; it is injected into the mainline via `control_q`.
- Outbound mainline parts are mirrored to the monitor as status text `[model_out] …` for observability.

## Event Injection (custom_event)

Monitor-generated default-stream text can carry:

```
metadata = {
  "custom_event": {
    "type": "caption" | "keyword" | "anomaly" | ...,   # event category
    "inject": {
      "role": "user" | "system",                       # default: user
      "end_of_turn": true|false,                         # default: false
      "text": "override text"                           # optional; else part.text
    },
    "data": { ... arbitrary payload ... }
  }
}
```

Behavior
- If `custom_event` is present on a default-stream text from the monitor, the server enqueues an injected part into `control_q` for the mainline.
- If `inject.end_of_turn` is true, an end-of-turn marker is also enqueued.
- Back-compat: `metadata["caption_event"]==True` is treated as `custom_event.type=="caption"`.

## Substreams & Routing

- Reserved: `status`, `debug`, `caption` (bypass model; forwarded immediately).
- Default: `""` (goes to mainline). Only text/EOT/tool_response from the client and injected events are allowed into mainline.
- Monitor mirror: mainline outputs summarized as status text `[model_out] …` to avoid binary spam while preserving observability.

## Concurrency & Safety

- WS writes are serialized with an `asyncio.Lock` to avoid interleaved frames.
- `streams.split(..., with_copy=True)` prevents in-place mutations in one branch from affecting the other.
- Control channel (`control_q`) is independent, so a slow monitor cannot back-pressure the mainline.

## REST vs WS

- REST (`POST /chat`) remains single-shot. Reserved substreams are filtered so they don’t enter the model; only default substream text is aggregated for the response.

## Test Coverage (summary)

- Mainline isolation: non-text inputs don’t pass through directly; mainline emits text after EOT.
- Event injection: a monitor `custom_event` injects text/EOT; mainline produces the corresponding model output.
- Monitor mirroring: status messages `[model_out] …` appear for mainline outputs.
- Timing: reserved streams (status/caption) can arrive before model outputs.

## Anticipated Extensions

- Multi-observer support
  - Add `/ws/monitor` endpoint, broadcast monitor_stream to multiple observers.
  - Session scoping (e.g., `session_id`) to correlate events and outputs.

- Substream policy
  - Introduce a dedicated reserved substream `model_out` to distinguish from `status`.
  - Configurable reserved sets per endpoint or per session.

- Event policy & gating
  - Pluggable rules for which `custom_event.type` is allowed to inject, their target `role`, and auto‑EOT behavior.
  - Priority/ordering (e.g., drain `control_q` first to preempt user input when required).

- Backpressure & resource control
  - Bounded queues with drop policies for monitor mirror, or sampling of `[model_out]` logs.
  - Size limits for mirrored summaries (truncate long texts; redact PII).

- Rich mirrors
  - Option to mirror structured envelopes (tool_call/tool_response metadata) or limited binary thumbnails/previews.

- Tooling integration
  - `custom_event` types that request tool execution or tool cancellation (maps to GenAI function calls/responses).

- Audio/voice routing
  - Endpointing events (StartOfSpeech/EndOfSpeech) as `custom_event` to trigger model turns for audio UX.

- Observability hooks
  - Metrics export (Prometheus/OpenTelemetry) sourced from monitor stream counters.

## File Pointers

- Server entry: `real_world/app/main.py`
- Pipelines: `real_world/agent/pipelines.py`
- Captioner (emits custom_event): `real_world/agent/media_captioner.py`
- Streams utilities: `genai_processors/streams.py`

