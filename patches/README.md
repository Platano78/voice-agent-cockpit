# Persona / brain-selector patch pack

Task #12: in-UI persona + brain-selector panel for the voice agent.

Task #8 (2026-07-03): reinstalled from `github.com/huggingface/speech-to-speech`
main into a fresh venv, pinned at commit `1e63f7e9343e491809d0d60e64f7ea551dbe845a`
(2026-07-01), and ported this pack onto it. The live service now runs the
`chat-completions` LLM backend (`speech_to_speech/LLM/chat_completions_language_model.py::ChatCompletionsApiModelHandler`,
subclass of the new `LLM/base_openai_compatible_language_model.py::BaseOpenAICompatibleHandler`
shared by both OpenAI-style handlers) instead of `responses-api`. The old
0.2.10 install at `$HOME/speech-to-speech/.venv` is untouched and
kept as an instant rollback (see below).

## Install

- Repo: `$INSTALL_DIR` (default `$HOME/speech-to-speech-main`) — editable install,
  `.venv` there, python3.10, CPU-only torch/torchaudio 2.11.0 — no CUDA on this
  box outside the external :8084 model.
- `faster-qwen3-tts[ggml]` is a base (non-Darwin) dependency and its
  `qwentts-cpp-python` PyPI wheel is CUDA-only; the CPU wheel had to be
  fetched by hand from
  `https://huggingface.co/datasets/andito/qwentts-cpp-python-wheels/resolve/main/whl/cpu/qwentts_cpp_python-0.3.0+cpu-py3-none-manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl`
  and installed before `pip install -e ".[kokoro]"` would resolve. We don't
  use qwen3 TTS (we run `pocket`), but the base dependency still has to
  import cleanly.
- Extras installed: `.[kokoro]` (base install command) + `.[pocket,websocket]`
  (pocket TTS + the websocket-mode extra) — `nano-parakeet` (STT) was
  already a base dependency.

## Files

- `brain_control.py` — `speech_to_speech/brain_control.py`. Defines
  `BrainControl`, which handles `config_get`/`config_set` control messages:
  swapping the LLM backend (`client`/`model_name`/`_extra_body` — now on
  `BaseOpenAICompatibleHandler`, shared by both OpenAI-style handlers) live,
  updating the persona (`runtime_config.session.instructions`), and resetting
  chat history. Loads brain definitions from
  `$HOME/speech-to-speech/brains.json`. Ported for #8: `_extra_body`
  is now recomputed via `BaseOpenAICompatibleHandler._build_extra_body(base_url,
  True, None)` (mirrors the base class's own `setup()` rule instead of
  reimplementing it) and a brain's API key can come from a literal `api_key`
  or be read lazily from `api_key_file` (env-style `VAR=value` lines) +
  `api_key_var` — used for the Hermes shim's bearer token
  (`~/.hermes/shim.env`, `HERMES_SHIM_TOKEN`). The token is never
  logged; `_resolve_model`'s `GET {base_url}/models` sends it as
  `Authorization: Bearer <key>` when present.

- `websocket_streamer.py` — `speech_to_speech/connections/websocket_streamer.py`.
  Adds a `control_callback` constructor kwarg and a text-frame branch in
  `_handle_client`'s message loop: JSON text frames with
  `type in ("config_get", "config_set")` are dispatched to the callback via
  `asyncio.to_thread` (keeps network I/O off the audio loop) and the reply is
  sent back to the requesting client only. Binary (audio) frame handling is
  unchanged. Unmodified on main — diffed byte-for-byte before reapplying,
  zero drift.

- `s2s_pipeline.py` — `speech_to_speech/s2s_pipeline.py`. In `build_pipeline`,
  the `RuntimeConfig` built for the transcription notifier is hoisted into a
  local variable (`runtime_config`) so it can be reused, and
  `websocket_streamer_ref` is captured when `mode == "websocket"` (main
  doesn't track this itself). After the pipeline handlers are built, for
  websocket mode with either OpenAI-style LLM backend
  (`llm_backend in ("responses-api", "chat-completions")` — main already
  guards `_lm_vars` selection the same way), the handler instance is located
  by `isinstance(h, BaseOpenAICompatibleHandler)` (was
  `isinstance(h, ResponsesApiModelHandler)`) and a `BrainControl` is wired
  onto `websocket_streamer.control_callback`. #15: `LMOutputProcessor`'s
  `setup_kwargs` now also passes `text_prompt_queue` (the queue feeding the
  LLM handler) so it can push a follow-up generation request after a tool
  call resolves.

- `voice_tools.py` (#15, new) — `speech_to_speech/voice_tools.py`. Defines
  `TOOL_DEFS` (weather / web search / QMD knowledge-base lookup, as
  Realtime-style function tool dicts) and `execute(name, kwargs) -> str`,
  which dispatches to the matching implementation under a hard per-tool
  timeout and always returns a short, TTS-safe plain-text string (never
  raises, never returns raw JSON). `BrainControl.__init__` arms these onto
  `runtime_config.session.tools`. Args are extracted by name (`kwargs.get`),
  not positionally, with a clean spoken refusal on a missing/empty required
  arg; failures are logged with the exception repr (`logger.warning`) and the
  spoken string differentiates timeout ("... timed out.") from other failures
  ("... failed."). `knowledge_lookup` sends QMD both a `lex` and a `vec`
  sub-query in one `searches` call — `lex` alone misses approximate/misheard
  queries (STT can turn "pi harness" into "pie harness"); QMD itself
  merges/ranks/dedupes both sub-queries into one result list server-side, so
  no client-side merge step is needed.

- `lm_output_processor.py` (#15) — `speech_to_speech/LLM/lm_output_processor.py`.
  `process()` gained a `_run_tool_calls` step: when an `LLMResponseChunk`
  carries `tools`, each call is executed via `voice_tools.execute`, its
  output is recorded with `chat.append_tool_output`, and (if at least one
  call resolved) a follow-up `GenerateResponseRequest` is pushed onto
  `text_prompt_queue` so the model can speak the result. Only the native
  OpenAI-style `tool_calls` channel is live for this deployment's
  `chat-completions` LLM backend (`BaseOpenAICompatibleHandler` /
  `ChatCompletionsApiModelHandler`) — the tag-based `<code>func()</code>`
  prompting in `LLM/language_model.py` belongs exclusively to the
  `transformers`/`mlx-lm` backend class hierarchy and is never invoked here,
  so there is no double-channel risk to guard against. Because a tool-call
  turn's `content` is empty on the native channel, `process()` also yields a
  short filler `TTSInput` (only when `response_wants_audio`) immediately
  before `_run_tool_calls` blocks the thread on the executor — otherwise the
  user hears silence for the tool's timeout budget. The filler is picked
  from a rotating pool (`_pick_filler`, default seven phrases such as "Let
  me check." and "One sec.") that avoids repeating the previous phrase
  back-to-back; the pool is overridable via the `VOICE_TOOL_FILLERS` env var
  (pipe-`|`-separated phrases, since a phrase may itself contain a comma),
  and the special value `off` disables the filler entirely.

- `wakeword_gate.py` (new) — `speech_to_speech/wakeword_gate.py`. Defines
  `WakewordGate`, an optional join-deaf gate (borrowed from
  `ShayneP/local-voice-ai`, see
  `knowledge-base/research-notes/shaynep-local-voice-ai-2026-07-12.md`).
  Env-driven, off by default: `VOICE_WAKE_WORD=1` (truthy: `1`/`true`/`yes`/`on`)
  arms it; `VOICE_WAKE_WORD_MODEL` (default `hey_jarvis`, any openWakeWord
  pretrained model name or a path to a custom `.onnx`) selects the model;
  `VOICE_WAKE_WORD_THRESHOLD` (default `0.5`) sets the score gate; scores at
  or above `0.25` that don't cross it log a "near miss" at INFO (visible at
  the live log level) so a real attempt that fell short still leaves journal
  evidence — anything above `0.05` still also logs at DEBUG for finer
  calibration. While
  armed and not yet awake, `websocket_streamer.py`'s binary branch routes
  mic audio through `WakewordGate.feed()` instead of `input_queue` — the
  pipeline stays deaf until the wake phrase scores above threshold, then
  wakes once for the rest of that session (`WakewordGate.reset()` re-arms it
  on the next session, called next to the `SESSION_END` put on last-client
  disconnect). Fail-open by design: any exception loading the model or
  scoring a frame is logged once and treated as an immediate, permanent
  wake — a broken detector must never brick the assistant. The module
  itself never imports `openwakeword` at module scope (lazy import inside
  `_load_model`), so it stays importable/testable with the dependency
  absent.

  Deploy prerequisite (only if `VOICE_WAKE_WORD=1` is actually used):
  `pip install openwakeword onnxruntime` into `$INSTALL_DIR/.venv`, then
  one-time model download:
  `$INSTALL_DIR/.venv/bin/python3 -c "import openwakeword.utils; openwakeword.utils.download_models()"`
  (openwakeword >=0.5 ships no models in the wheel; older 0.4.x bundles the
  pretrained onnx models already and this step is a no-op). Client-side,
  the webclient shows a `zzz — say "<phrase>" to wake` status while asleep
  and a status-bar flip + short two-tone chime on `wakeword_state: awake`.

  Runtime control (settings panel, added after the initial deploy):
  `VOICE_WAKE_WORD`/`VOICE_WAKE_WORD_MODEL` are boot defaults only — the
  settings panel's "Wake word" block (checkbox + phrase dropdown, hidden
  entirely when `config_state.wake_word` is `null`, i.e. no websocket
  streamer wired) lets a session flip the gate on/off and pick a different
  pretrained phrase without a restart. The checkbox sends
  `config_set {wake_word: bool}`; `true` also calls `WakewordGate.rearm()`
  so a previously fail-open detector gets one fresh load attempt. The
  dropdown sends `config_set {wake_word_model: name}`, validated by
  `WakewordGate.set_model()` against `available_models()` (or an existing
  `.onnx`/`.tflite` path) before swapping and re-arming. Either change
  broadcasts a fresh `wakeword_state` (now also emitted with `state: "off"`
  when disabled) to every connected client, so a toggle from one device is
  reflected on all of them. `WakewordGate.available_models()` lists
  openWakeWord's `resources/models` directory (same layout on 0.4.x and
  >=0.5), strips the `_vX.Y`/extension suffix, and excludes the non-wake-phrase
  files (`melspectrogram`, `embedding_model`, `silero_vad`, `timer`,
  `weather`) — this is the custom-model seam: drop a trained `.onnx` into
  that directory and it auto-appears in the dropdown next time the panel
  refreshes, no code change needed.

- `think_filter.py` (new) — `speech_to_speech/think_filter.py`. Defines
  `ThinkTagFilter`, a dependency-free (stdlib only) stateful streaming
  suppressor for literal `<think>...</think>` reasoning spans. Some local
  reasoning models (e.g. `reasoning-qwen36-27b-mtp` behind llama.cpp) emit
  their reasoning as `<think>...</think>` in the regular chat-completions
  `content` field even with `chat_template_kwargs.enable_thinking=false` —
  probed live: `content` was `"<think>\n\n</think>\n\nHi there friend"`.
  Without filtering, this reasoning text is spoken by TTS. `feed(text)`
  streams a chunk through and returns the safe-to-emit portion (a tag can
  straddle a chunk boundary, so a partial-tag prefix is buffered across
  calls); `flush()` is called once at stream end and returns any buffered
  prefix that turned out to be innocent text (an unclosed `<think>` is
  suppressed instead, since it was reasoning that never finished). Also
  swallows the leading newline run right after a closed think span so TTS
  doesn't see a leading blank chunk. No `speech_to_speech` import, so it (and
  `test_think_filter.py`) is importable/testable standalone.

- `chat_completions_language_model.py` (new) —
  `speech_to_speech/LLM/chat_completions_language_model.py`. Wires
  `ThinkTagFilter` into `ChatCompletionsApiModelHandler`: `_iter_stream_events`
  runs one filter instance per streamed response, feeding each delta's raw
  text through it and only accumulating the *filtered* piece into `raw_text`
  (the flush remnant is folded in the same way) before yielding a
  `TextDelta` — so `raw_text`, which becomes the `AssistantMessage` written
  back to conversation history, never contains a think span either; the
  model would otherwise re-read its own suppressed reasoning as context on
  every later turn, and an all-thinking response would silently store
  thinking while speaking nothing. The existing `if raw_text.strip():`
  guard then naturally skips emitting an `AssistantMessage` at all for a
  response that was 100% reasoning. `_iter_response_events` (non-streaming)
  runs the whole response's text through a one-shot `feed()` + `flush()`
  before yielding it, so both the `AssistantMessage` and `TextDelta` there
  are filtered identically.

## Files OUTSIDE the package (survive reinstall — not part of this pack)

- `$HOME/speech-to-speech/brains.json` — brain registry (label,
  base_url, model, availability, notes, optional `api_key`/`api_key_file`+
  `api_key_var`). `hermes` flipped to `available: true` for #8.
- `<cockpit-repo>/webclient/index.html` — settings panel UI
  (gear button, brain radio list, persona textarea, reset-chat button). No
  protocol change was needed for #8 — the ws control-message shape is
  identical on both backends.

## Remote access / HTTPS (mic requires a secure context)

Browsers only allow `getUserMedia` (microphone access) on a "secure context" —
`https://` or `http://localhost`. Loading the cockpit over plain `http://` from
another device on your LAN (by IP or hostname) will silently fail to get mic
permission. Both halves of the stack — the static webclient and the pipeline's
WebSocket — need to run over TLS to fix this.

1. **Generate a certificate.** [mkcert](https://github.com/FiloSottile/mkcert)
   is the easiest path and avoids browser warnings on any device that trusts
   its local CA:

   ```bash
   mkcert -install
   mkcert <lan-ip-or-hostname>   # e.g. mkcert 192.168.1.50
   ```

   This produces a `<name>.pem` (cert) and `<name>-key.pem` (key) in the
   current directory. If you'd rather not install mkcert, a self-signed cert
   with `openssl` works too (browsers will show a one-time warning to click
   through, on each device):

   ```bash
   openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
     -keyout key.pem -out cert.pem -subj "/CN=<lan-ip-or-hostname>"
   ```

2. **Serve the webclient over HTTPS** using the cert pair:

   ```bash
   python3 webclient/serve.py --certfile cert.pem --keyfile key.pem
   ```

3. **Point the pipeline's WebSocket at the same cert pair** via env vars
   before starting the `speech-to-speech` service:

   ```bash
   export VOICE_WS_CERTFILE=/path/to/cert.pem
   export VOICE_WS_KEYFILE=/path/to/key.pem
   # export VOICE_WSS_PORT=8443   # optional, this is the default
   ```

   When both are set, `websocket_streamer.py` starts a second, TLS-wrapped
   listener on `VOICE_WSS_PORT` alongside the existing plain-`ws` listener
   (which keeps running unchanged for `localhost`/http use) — the same cert
   pair from step 1 works for both the webclient and the WebSocket. If the
   cert fails to load, the service logs the error and continues serving
   plain `ws` only; a bad cert never takes down the voice pipeline.

4. **`localhost` needs none of this** — plain `http`/`ws` on `localhost` is
   already a secure context, so the mic works there with no cert setup at
   all.

## Re-applying

```bash
bash <cockpit-repo>/patches/apply.sh
sudo systemctl restart voice-agent
```

`apply.sh` locates the current `speech_to_speech` package directory via the
active venv's Python (`import speech_to_speech; ... __file__`) — since
`$INSTALL_DIR` is an editable install, this resolves
straight to the repo's `src/speech_to_speech`, so a `pip install` of new deps
never wipes these edits (only a `git checkout` of those three files would).
Safe to re-run any number of times.

## Rollback (0.2.10 / responses-api)

The pre-#8 systemd unit is saved at
`<cockpit-repo>/patches/voice-agent-0.2.10.service.bak`
(`ExecStart` uses `$HOME/speech-to-speech/.venv` and
`--llm_backend responses-api`, unchanged from before #8). To roll back:

```bash
sudo cp <cockpit-repo>/patches/voice-agent-0.2.10.service.bak /etc/systemd/system/voice-agent.service && sudo systemctl daemon-reload && sudo systemctl restart voice-agent
```

The old venv (`$HOME/speech-to-speech/.venv`) was never touched by
the #8 work — `$HOME/speech-to-speech/.venv/bin/speech-to-speech
--help` still runs — so this restores the exact pre-#8 pipeline.
