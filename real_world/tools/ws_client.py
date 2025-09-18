#!/usr/bin/env python3
from __future__ import annotations

"""Interactive WebSocket client for debugging the running server.

Usage:
  python -m real_world.tools.ws_client --url ws://127.0.0.1:8000/ws

Commands (type in the prompt):
  /text <message>                      send text
  /eot                                 end of turn
  /image <path> [mimetype]             send image file as base64 (default mimetype=image/png)
  /audio <path> [mimetype] [sub]       send audio file (default mimetype=audio/wav, sub=realtime)
  /video <path> [mimetype] [sub]       send video file (default mimetype=video/mp4, sub=realtime)
  /tool_response <name> <json> [id]    send tool response (json object as string)
  /config <typed|generic>              switch server output mode
  /await [timeout_sec]                 wait for next non-reserved message and print it
  /quit                                exit client
  /help                                show this help

Incoming messages are printed as JSON. For binary, only metadata is shown.
"""

import argparse
import asyncio
import base64
import json
import os
import contextlib
from typing import Any

import websockets


def b64_file(path: str) -> str:
  with open(path, 'rb') as f:
    return base64.b64encode(f.read()).decode('ascii')


def fmt(obj: Any) -> str:
  try:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)
  except Exception:
    return str(obj)


HELP = __doc__.split("Usage:")[1].strip() if __doc__ else ""


async def sender(ws, alive: asyncio.Event, q: asyncio.Queue):
  print("Type /help for commands. Ctrl+C to exit.")
  while True:
    try:
      line = await asyncio.to_thread(input, "> ")
    except (EOFError, KeyboardInterrupt):
      break
    line = line.strip()
    if not line:
      continue
    if line.startswith('/quit'):
      break
    if line.startswith('/help'):
      print(HELP)
      continue
    try:
      await handle_command(ws, line, q)
    except Exception as e:
      print(f"! error: {e}")
    if not alive.is_set():
      print("! connection is closed")
      break


def _send_json(ws, obj):
  data = json.dumps(obj, ensure_ascii=False)
  print(f"-> {data}")
  return ws.send(data)


def _flush_queue(q: asyncio.Queue):
  try:
    while True:
      q.get_nowait()
  except asyncio.QueueEmpty:
    return


async def handle_command(ws, line: str, q: asyncio.Queue):
  toks = line.split()
  cmd = toks[0]
  if cmd == '/await':
    timeout = float(toks[1]) if len(toks) > 1 else 5.0
    print(f"# awaiting next non-reserved message (timeout={timeout}s)...")
    # Discard any backlog so that we wait for the next messages after this point.
    _flush_queue(q)
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
      remaining = deadline - asyncio.get_running_loop().time()
      if remaining <= 0:
        print("! timeout waiting for server message")
        break
      try:
        msg = await asyncio.wait_for(q.get(), timeout=remaining)
      except asyncio.TimeoutError:
        print("! timeout waiting for server message")
        break
      sub = msg.get('substream', '')
      if sub in ('status','debug','caption'):
        # Already printed by receiver; continue waiting for non-reserved
        continue
      print(f"<- {msg.get('type')}: {fmt(msg)}")
      break
    return
  if cmd == '/text':
    msg = line[len('/text'):].strip()
    await _send_json(ws, {"type": "text", "text": msg})
    return
  if cmd == '/turn':
    # Convenience: send text and immediately end the turn.
    msg = line[len('/turn'):].strip()
    await _send_json(ws, {"type": "text", "text": msg})
    await _send_json(ws, {"type": "end_of_turn"})
    return
  if cmd == '/eot':
    await _send_json(ws, {"type": "end_of_turn"})
    return
  if cmd == '/config':
    if len(toks) < 2 or toks[1] not in ('typed', 'generic'):
      raise ValueError('/config <typed|generic>')
    await _send_json(ws, {"type": "config", "output_mode": toks[1]})
    return
  if cmd == '/image':
    if len(toks) < 2:
      raise ValueError('/image <path> [mimetype]')
    path = toks[1]
    mime = toks[2] if len(toks) >= 3 else 'image/png'
    await _send_json(ws, {
      "type": "image",
      "mimetype": mime,
      "data_b64": b64_file(path),
      "substream": "realtime",
    })
    return
  if cmd == '/audio':
    if len(toks) < 2:
      raise ValueError('/audio <path> [mimetype] [sub]')
    path = toks[1]
    mime = toks[2] if len(toks) >= 3 else 'audio/wav'
    sub = toks[3] if len(toks) >= 4 else 'realtime'
    await _send_json(ws, {
      "type": "audio",
      "mimetype": mime,
      "data_b64": b64_file(path),
      "substream": sub,
    })
    return
  if cmd == '/video':
    if len(toks) < 2:
      raise ValueError('/video <path> [mimetype] [sub]')
    path = toks[1]
    mime = toks[2] if len(toks) >= 3 else 'video/mp4'
    sub = toks[3] if len(toks) >= 4 else 'realtime'
    await _send_json(ws, {
      "type": "video",
      "mimetype": mime,
      "data_b64": b64_file(path),
      "substream": sub,
    })
    return
  if cmd == '/tool_response':
    # /tool_response <name> <json> [id]
    if len(toks) < 3:
      raise ValueError('/tool_response <name> <json> [id]')
    name = toks[1]
    json_str = line.split(' ', 2)[2]
    try:
      payload = json.loads(json_str)
      tool_id = None
    except json.JSONDecodeError:
      # With id at the end: split json and id
      # find last space to separate id
      pos = json_str.rfind(' ')
      if pos <= 0:
        raise
      payload = json.loads(json_str[:pos])
      tool_id = json_str[pos+1:]
    msg = {"type": "tool_response", "name": name, "response": payload}
    if tool_id:
      msg["id"] = tool_id
    await _send_json(ws, msg)
    return
  raise ValueError('unknown command, type /help')


async def receiver(ws, alive: asyncio.Event, q: asyncio.Queue):
  alive.set()
  while True:
    try:
      raw = await ws.recv()
    except (asyncio.CancelledError, websockets.exceptions.ConnectionClosedOK, websockets.exceptions.ConnectionClosedError):
      # Graceful shutdown or remote close
      alive.clear()
      return
    except Exception as e:
      print(f'! disconnected: {e}')
      alive.clear()
      return
    try:
      msg = json.loads(raw)
    except Exception:
      print(raw)
      continue
    # Fan-out to a queue for /await command without concurrent recv
    try:
      q.put_nowait(msg)
    except Exception:
      pass
    t = msg.get('type')
    sub = msg.get('substream', '')
    if t == 'binary':
      size = len(msg.get('data_b64', ''))
      print(f"<- {t}({msg.get('media_kind','?')}), bytes(b64)={size}, sub={sub}")
    else:
      print(f"<- {t}: {fmt(msg)}")


async def _bootstrap(ws):
  """Send a short sample sequence to demonstrate server behavior."""
  print("# sending bootstrap sample: text → image(1x1) → end_of_turn")
  await _send_json(ws, {"type": "config", "output_mode": "typed"})
  await _send_json(ws, {"type": "text", "text": "hello from ws_client"})
  # 1x1 PNG pixel
  tiny_png_b64 = (
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVQ"
      "ImWNgYAAAAAMAASsJTYQAAAAAElFTkSuQmCC"
  )
  await _send_json(ws, {
      "type": "image",
      "mimetype": "image/png",
      "data_b64": tiny_png_b64,
      "substream": "realtime",
  })
  await _send_json(ws, {"type": "end_of_turn"})


async def main(url: str, *, bootstrap: bool = True):
  async with websockets.connect(url) as ws:
    alive = asyncio.Event()
    q: asyncio.Queue = asyncio.Queue()
    recv_task = asyncio.create_task(receiver(ws, alive, q))
    try:
      if bootstrap:
        await _bootstrap(ws)
      await sender(ws, alive, q)
    finally:
      recv_task.cancel()
      with contextlib.suppress(asyncio.CancelledError, Exception):
        await recv_task


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--url', default='ws://127.0.0.1:8000/ws')
  parser.add_argument('--no-bootstrap', action='store_true', help='Disable sending a sample sequence on connect')
  args = parser.parse_args()
  asyncio.run(main(args.url, bootstrap=not args.no_bootstrap))
