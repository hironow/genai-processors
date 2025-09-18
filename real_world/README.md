# Real-World サンプル（日本語版）

GenAI Processors を使った「現場でそのまま使える」サンプル実装です。REST / WebSocket / ライブラリ API の3パターンに加え、実運用を見据えた前処理（重複排除・サイズ制限）とマルチモーダル LLM によるキャプション化を組み合わせ、どんな入力ストリームでも最終的にテキスト中心へ正規化できます。

■ 目的
- 任意モダリティ（テキスト/画像/音声/動画）を安全に受け取り、後段処理が扱いやすいサイズ・形式へ整える
- マルチモーダル LLM を活用して事実ベースの代替テキスト（alt-text）を生成し、ストリームをテキスト化
- 逐次ログ（status/caption）を予約サブストリームで即時配信（モデル処理を待たない）

■ 実装の全体像
- REST: `POST /chat`（テキストのみ）
- WebSocket: `GET /ws`（テキスト/画像/音声/動画/ツール応答、双方向ストリーミング）
- ライブラリ: `real_world/lib/streaming_agent.py`（外部 API はストリームだけ公開、内部で GenAI Processors を使用）

■ パイプライン（REST/WS 共通の中核）
`ModalityMetricsLogger` → `ContentDeduper` → `MediaSizeLimiter` → `MediaCaptioner(build_caption_model)` → `TurnModel / LiveProcessor`

- ModalityMetricsLogger: 各パート到着時に件数をカウントし、同モダリティのログパートを逐次出力（status サブストリーム）
- ContentDeduper: 内容ハッシュで重複をドロップ（status に重複ログ）
- MediaSizeLimiter: 画像縮小/再圧縮、音声短縮、動画はポリシー（許可/ドロップ/バイト切詰）
- MediaCaptioner: マルチモーダル LLM（GenaiModel）で事実のみの alt-text を生成し TEXT 化。キャプション生成時のみ caption サブストリームへログ
- TurnModel/LiveProcessor: 既定は Gemini（GOOGLE_API_KEY 必須）。WS では LiveProcessor が双方向リアルタイムを提供

■ 予約サブストリーム（即時配信）
- `debug` / `status` / `caption` は「予約サブストリーム」です。モデル入力へ回さず、即座にクライアントへ送出されます。
- WebSocket/REST ともに、サーバ側で `reserved_substreams=("debug","status","caption")` を設定済み。

■ WebSocket JSON スキーマ（要約）
- クライアント→サーバ
  - Text: `{ "type":"text", "text":"..." }`
  - End of turn: `{ "type":"end_of_turn" }`
  - Image: `{ "type":"image", "mimetype":"image/png", "data_b64":"...", "substream":"realtime" }`
  - Audio: `{ "type":"audio", "mimetype":"audio/wav|audio/l16;rate=24000", "data_b64":"...", "substream":"realtime", "audio_stream_end":false }`
  - Video: `{ "type":"video", "mimetype":"video/mp4", "data_b64":"...", "substream":"realtime" }`
  - Tool response: `{ "type":"tool_response", "name":"fn", "response":{...}, "id":"optional" }`
- サーバ→クライアント
  - Text: `{ "type":"text", "text":"...", "substream":"" }`
  - Tool call: `{ "type":"tool_call", "name":"fn", "args":{...} }`
  - Binary: 既定（generic）`{ "type":"binary", "data_b64":"...", "mimetype":"...", "media_kind":"image|audio|video" }`
    - Typed モード（クライアントが `{"type":"config","output_mode":"typed"}` を接続直後に送信）では `type:image|audio|video` を返却
  - 予約サブストリーム（例）
    - status: `text_count=...` などの逐次メトリクス
    - caption: キャプション生成時だけのログ `[caption] ...`

■ セットアップ / 実行
- 依存: Python 3.10+, `pip install fastapi uvicorn[standard] pydantic genai-processors google-genai`
- 環境変数: `export GOOGLE_API_KEY=...`
- 実行（開発）
  - `uvicorn real_world.app.main:app --reload`
  - REST 例: `curl -X POST http://127.0.0.1:8000/chat -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"hello"}]}'`
  - WS 例: 仕様に沿って JSON を送受信（上記スキーマ参照）
  - 手動デバッグ用 WS クライアント:
    - 依存: `pip install websockets`
    - 起動: `python -m real_world.tools.ws_client --url ws://127.0.0.1:8000/ws`
      - 接続直後にサンプル（text→1x1画像→end_of_turn）を自動送信します。無効化したい場合は `--no-bootstrap` を付けてください。
    - プロンプトで `/help` を入力すると利用可能なコマンドが表示されます（/text, /eot, /image, /audio, /video, /tool_response, /config など）

■ テスト
- `pytest real_world/tests` で E2E/ユニット混在の一連の検証を実行
  - 逐次ログの即時性（status/caption がモデル出力より先に届く）
  - ツール呼び出しの往復（tool_call → tool_response → 次ターン出力）
  - 画像/音声/動画のサイズ制限・ドロップ/切詰の挙動
  - キャプション生成時だけ caption サブストリームにログが1件出ること

■ Parallel と Concurrent の違い（やさしく）
- Concurrent（並行）: 「同時期に進む」こと。GenAI Processors のチェーンは非同期で動き、I/O待ち（LLM呼び出し・音声再生・WS送受信）と他の処理が重なって進みます。予約サブストリームはモデル処理を待たずに先にクライアントへ送られる＝良い並行性の例。
- Parallel（並列）: 「同時に実行する」こと。CPUコア等を使って同時実行。GenAI Processors では PartProcessor を複数本で //（平行：parallel）に流す設計も可能（map_processor の parallel 関数や PartProcessor の `__floordiv__` 演算子）。今回のサンプルでは、1本のチェーン内で各パート処理を同時進行（並行）させています。複数の PartProcessor を // で束ねると、同一パートに対する複数のサブルーチンを並列に実行できます。
- 実務上のコツ
  - ブロッキング処理（画像エンコード、長い I/O）はできるだけ小さなクリティカルセクションにし、await の外側でもたせる（例: Deduper はハッシュ計算はロック外/登録だけロック）
  - 予約サブストリームの活用で、UI が欲しい「即時性のある進捗」を先に届ける
  - LiveProcessor は「入力の蓄積」「モデル呼び出し」「出力配信」を別タスクで進め、TTFT を下げる工夫を持つ

■ 補足
- 例は Gemini（GOOGLE_API_KEY）で動きます。Vertex AI のモデル名に差し替える場合は `agent/models.py` を調整してください。
- よりリッチな音声体験は `core/audio_io.py` / `core/speech_to_text.py` / `core/rate_limit_audio.py` との併用を推奨します。
