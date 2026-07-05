# Voice Agent Cockpit

A self-hostable local voice-agent cockpit: the Hugging Face
[`speech-to-speech`](https://github.com/huggingface/speech-to-speech) framework
paired with a static web cockpit (`webclient/`) that includes an avatar pane
with an **avatar selector dropdown** to pick among bundled
[TalkingHead](https://github.com/met4citizen/TalkingHead) 3D heads (GLB) or a
2D still-image avatar (mouth lip-synced from the same audio) — the head choice
is user-owned and persisted, independent of the active theme — a
brain/persona selector panel, and a settings UI for switching LLM backends
live over the WebSocket control channel.

This repo is a **skeleton**: the custom cockpit UI and the patch pack that
wires persona/brain switching into `speech-to-speech` are here, but the
framework itself, your model endpoints, and any avatar assets are supplied by
you at install time.

## Architecture

```
 browser (webclient/index.html)
        │  WebSocket  ws://<host>:8765
        ▼
 speech-to-speech pipeline (patched)
   ├─ STT   — parakeet-tdt
   ├─ LLM   — OpenAI-compatible chat-completions endpoint ("brain")
   └─ TTS   — pocket
        ▲
        │  HTTP :8770 (static files)
 webclient/serve.py
```

- The browser cockpit connects to the pipeline over a single WebSocket
  (`ws://<hostname>:8765`) for audio in/out plus JSON control messages
  (`config_get`/`config_set` — brain selection, persona text, chat reset).
- `webclient/serve.py` serves the cockpit's own static files on `:8770`
  (nothing else — no other client-side endpoints beyond the WS and a relative
  `fetch("themes/themes.json")`).
- The LLM ("brain") is any OpenAI-compatible `chat-completions` endpoint —
  local (llama.cpp, vLLM, Ollama, etc.) or hosted. `patches/brain_control.py`
  lets you register several brains in `brains.json` and hot-swap between them
  from the cockpit UI without restarting the service.

## Prerequisites

- An **OpenAI-compatible chat-completions LLM endpoint** (local or remote).
- Enough CPU/GPU for **STT** (`parakeet-tdt`) and **TTS** (`pocket`) — both
  run fine CPU-only; a GPU only helps the LLM.
- Python 3.10+, `git`.

## Self-host setup

```bash
# 1. Clone and pin the framework at the commit this pack was built against
git clone https://github.com/huggingface/speech-to-speech.git speech-to-speech-main
cd speech-to-speech-main
git checkout 1e63f7e9343e491809d0d60e64f7ea551dbe845a

# 2. Create a venv and install (CPU-only torch/torchaudio is fine)
python3 -m venv .venv
.venv/bin/pip install -e ".[kokoro]"
.venv/bin/pip install -e ".[pocket,websocket]"

# 3. Apply this cockpit's patch pack on top of the editable install
bash /path/to/voice-agent-cockpit/patches/apply.sh

# 4. Configure your brain(s)
cp /path/to/voice-agent-cockpit/brains.json.example /path/to/voice-agent-cockpit/brains.json
# edit brains.json: base_url / model / api_key(_file) for each backend you want.
# The pipeline locates this file via the BRAINS_JSON env var (default
# ~/speech-to-speech/brains.json) — export it to point at the file you just
# created (done in step 5 below), or move the file to that default path.

# 5. Run the pipeline, pointing --responses_api_base_url / --model_name at
#    your own endpoint (these are the flags patches/apply.sh's target reads
#    at startup; brains.json lets you add more brains to hot-swap between
#    afterwards):
BRAINS_JSON=/path/to/voice-agent-cockpit/brains.json \
.venv/bin/speech-to-speech \
  --mode websocket --ws_host 0.0.0.0 --ws_port 8765 \
  --stt parakeet-tdt --parakeet_tdt_device cpu \
  --tts pocket --pocket_tts_voice jean --pocket_tts_device cpu \
  --llm_backend chat-completions \
  --responses_api_base_url http://localhost:8084/v1 \
  --responses_api_api_key dummy \
  --model_name <your-model-name> \
  --responses_api_stream

# 6. Serve the cockpit
python3 /path/to/voice-agent-cockpit/webclient/serve.py --port 8770
```

Then open `http://<host>:8770` in a browser.

Key flags to point at your own infrastructure:

| Flag | Purpose |
|---|---|
| `--responses_api_base_url` | your OpenAI-compatible LLM endpoint |
| `--model_name` | model id served at that endpoint |
| `--stt` | STT backend (`parakeet-tdt` by default) |
| `--tts` | TTS backend (`pocket` by default) |
| `--ws_port` | WebSocket port the cockpit connects to |

See `patches/README.md` for what the patch pack changes and why, and
`systemd/*.template` for a reference of running both processes as systemd
services.

## Assets NOT bundled

Avatar model files (`webclient/avatar/model/*.glb`), vendor binary assets
(`webclient/avatar/vendor/**/*.bin`), and any theme reference images are
**not included** in this repo (see `.gitignore` and `NOTICE`) — licensing on
those is separate from this project's code. Supply your own avatar GLB into
`webclient/avatar/model/` (compatible with
[TalkingHead](https://github.com/met4citizen/TalkingHead)) and your own theme
reference images if you use the theming tools under `webclient/themes/`. If
you want the 2D still-image avatar option, also supply a still image at
`webclient/avatar/refs/<name>.png` (gitignored, same as the GLBs) — it's
lip-synced by `webclient/avatar/avatar2d.mjs`.

## Optional integrations

The patch pack reads a few env vars for optional local-service integrations.
All are optional with localhost defaults — ignore them if you don't run those
services; the core voice agent (LLM brain + STT + TTS) works without any of
them.

| Env var | Default | Purpose |
|---|---|---|
| `BRAINS_JSON` | `~/speech-to-speech/brains.json` | path to your brain registry |
| `HERMES_SHIM_URL` | `http://localhost:8087/v1/chat/completions` | optional Hermes "shim" brain endpoint |
| `HERMES_SHIM_TOKEN_FILE` | `~/.hermes/shim.env` | optional Hermes shim token file |
| `HERMES_MCP_URL` | `http://localhost:8088/mcp` | optional Hermes MCP endpoint (cockpit brain) |
| `QMD_MCP_URL` | `http://localhost:8070/mcp` | optional QMD knowledge endpoint (voice tools) |

## Attribution

- [`huggingface/speech-to-speech`](https://github.com/huggingface/speech-to-speech) — Apache-2.0
- [`met4citizen/TalkingHead`](https://github.com/met4citizen/TalkingHead) — MIT

See `NOTICE` for full attribution and licensing notes.
