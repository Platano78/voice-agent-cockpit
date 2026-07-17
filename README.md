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

### One conversation, many screens

Every connected browser is a window onto the **same** session: one chat
history, one brain, one voice. Start a conversation at the desk, continue it
from the phone — that continuity is deliberate. Events are broadcast live to
whoever is connected at that moment (there is no history replay on join, so a
device that reconnects only shows what happened after it joined).

If you want genuinely separate conversations per device, don't look for a
toggle — run a **second pipeline instance** on another port (`--ws_port 8766`
plus a second systemd unit) and point the other device at it. Both instances
can share the same LLM endpoint, which is stateless per request; the cost is a
second copy of STT+TTS on CPU. Per-client sessions inside one instance would
require restructuring the upstream framework's single-conversation design and
is not planned.

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

## Bundled assets & what you supply

**Bundled:** six ready avatar heads in `webclient/avatar/model/` (mirrored from
the [TalkingHead](https://github.com/met4citizen/TalkingHead) repo's own public
distribution) — per-file licensing in `webclient/avatar/model/LICENSE-NOTE.txt`
(`mpfb.glb` is CC0; the rest are non-commercial/personal-use). The HeadAudio
viseme model (`model-en-mixed.bin`) and vendored three.js/TalkingHead/HeadAudio
libraries are included (MIT).

**You supply:** a still image at `webclient/avatar/refs/<name>.png` (gitignored)
if you want the 2D still-image avatar — it's lip-synced by
`webclient/avatar/avatar2d.mjs`; and your own theme reference images if you use
the theming tools under `webclient/themes/`. You can also drop in any extra
TalkingHead-compatible GLB and add one line to `AVATAR_REGISTRY` in
`webclient/index.html`.

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
| `HERMES_TARGET` | *(unset — required for `send_to_hermes`)* | Hermes message target, `platform:chat_id` per Hermes `channels_list` |
| `QMD_MCP_URL` | `http://localhost:8070/mcp` | optional QMD knowledge endpoint (voice tools) |
| `VOICE_TOOLS` | *(unset)* | pin the armed voice-tool set to a comma-separated list |
| `VOICE_TOOLS_DIR` | *(unset)* | directory of drop-in local voice tools (one `.py` per tool) |
| `VOICE_CLONE_DIR` | `~/speech-to-speech/voices` | where custom (cloned) voice states are stored |
| `VOICE_AUDITION_TEXT` | `Hi, I'm {name}. This is how I sound.` | sample spoken after a voice switch; `off` disables |
| `VOICE_PHONE_CONTEXT` | *(unset)* | set to `off` to disable phone context server-side, even if a client has it toggled on |

## Voice tools

The voice agent's LLM can call a small set of server-side tools (defined in
`patches/voice_tools.py`). Their spoken results are kept short and TTS-friendly.

| Tool | What it does | Backing service |
|---|---|---|
| `get_weather` | Current conditions for a place | Open-Meteo (public web API) |
| `web_search` | Search the public web; results also appear as clickable links on screen | DuckDuckGo (`ddgs`) |
| `knowledge_lookup` | Search your own notes, projects, and research | QMD MCP (`QMD_MCP_URL`) |
| `set_mood` | Set the interface/avatar mood | none — client-side visual only |
| `delegate_to_hermes` | Hand off a long-running / multi-step task to Hermes | Hermes shim (`HERMES_SHIM_URL`) |
| `hermes_status` | Report Hermes' status (summary, last result, recent steps, or pending approvals) | Hermes MCP (`HERMES_MCP_URL`) |
| `respond_permission` | Approve or deny the oldest pending Hermes approval | Hermes MCP (`HERMES_MCP_URL`) |
| `send_to_hermes` | Send a quick free-form message / follow-up to Hermes | Hermes MCP (`messages_send` → `HERMES_TARGET`) |

Arming is **availability-probed once at pipeline start**: the weather, search,
and mood tools are always armed; `knowledge_lookup` is armed only if the QMD MCP
endpoint answers a probe; the four Hermes tools are armed only if the Hermes MCP
endpoint answers a probe. This keeps a self-hoster without those services from
arming dead tools the LLM would call and then narrate as failures. Set
`VOICE_TOOLS=<comma-list>` to pin the set explicitly (no probing — only listed
names that exist are kept). Probing happens at startup only, so **restart the
pipeline to re-arm** after bringing a service up or down. The settings UI's
"N armed" count shows the result. `web_search` results also surface as a
clickable **Links card** in the cockpit.

### Add your own tools

Set `VOICE_TOOLS_DIR=/path/to/dir` to drop in extra tools without editing repo
files — one `.py` per tool exposing a `TOOL_DEF` dict and a `run()` callable
(see `examples/tools/` for five ready-to-copy tools (clock, model-server status, service health, GPU status, news headlines) — edit each file's EDIT-ME constants for your own endpoints). This **executes your Python on your
box**, so treat the directory with the same trust as editing config. Drop-ins
arm unconditionally when the directory is set (unless `VOICE_TOOLS` pins the
list); a broken file is skipped with a logged warning rather than crashing the
pipeline. No restart needed to pick up changes — the settings panel's
"Reload tools" button re-probes and re-arms live (also
`{"type":"config_set", "reload_tools":true}` over the WS).

## Custom voices (voice cloning)

The settings panel's **Advanced — custom voice** section lets you add your own
voices to the dropdown: record 10–30 seconds of speech in the browser, or
upload a clip (`.wav`, `.aiff`, `.flac`, `.ogg`, `.mp3` — not `.webm`/`.m4a`;
convert those first). The server builds a pocket-tts voice state from it once
(a few seconds of CPU), stores it under `VOICE_CLONE_DIR`
(default `~/speech-to-speech/voices/`, safe across framework reinstalls), and
the voice appears in the dropdown on every connected client. Switching voices
speaks a short audition sample so you hear the result immediately
(`VOICE_AUDITION_TEXT`, set to `off` to disable). Clips longer than 30 seconds
are truncated to the first 30. Clean audio matters: background noise, echo,
and compression artifacts become part of the cloned voice, so record somewhere
quiet (Kyutai recommends cleaning the sample first — e.g. Adobe's free
[Enhance Speech](https://podcast.adobe.com/en/enhance)).

**One-time setup** — pocket-tts ships its clone-capable weights behind a
Hugging Face terms gate. Until you complete this, the cockpit shows a
friendly error instead of building voices (already-built custom voices keep
working regardless):

1. Accept the terms at <https://huggingface.co/kyutai/pocket-tts>
   (any HF account; approval is automatic).
2. On the box that runs the pipeline, log that account in as the service
   user: `hf auth login` (or place a token at `~/.cache/huggingface/token`).
3. Restart the pipeline service — it re-downloads the model weights with
   cloning enabled (~220 MB, one time).

`soundfile` must be installed in the pipeline's venv for non-WAV uploads and
recording normalization: `pip install soundfile`.

**Consent notice**: pocket-tts's license prohibits "voice impersonation or
cloning without explicit and lawful consent." Clone only voices you have the
right to clone — your own, or a consenting speaker's.

## Phone context (optional)

The settings panel has a **Phone context** toggle, **off by default**: "Share
location & phone state". When you turn it on, the browser streams your
approximate location (via the W3C Geolocation API), timezone, and battery
level/charging state to your own voice server over the same WebSocket the
rest of the cockpit already uses — no new endpoint, no third-party service.
This lets tools like `get_weather` and the LLM's sense of "now"/"here" work
without you naming a place every time.

It's sent only to your voice server, but the ambient line it produces reaches
whichever brain is currently active — including a remote/cloud brain, if
that's what you've selected in the Brain panel. If that matters to you,
switch to a local brain before enabling it, or leave it off.

Location updates are throttled client-side (moved >100m or 5+ minutes since
the last send) and expire server-side after 30 minutes of staleness. Denying
the browser's location permission prompt turns the toggle back off. Set
`VOICE_PHONE_CONTEXT=off` on the server to disable the feature entirely,
regardless of what any client has toggled.

Any browser works — a phone gives GPS-grade accuracy, a desktop typically
falls back to coarser IP/network-based geolocation, which is still useful for
timezone/weather purposes.

## Attribution & acknowledgments

This project borrows ideas as well as code, and gladly says so:

- [`huggingface/speech-to-speech`](https://github.com/huggingface/speech-to-speech) (Apache-2.0) — the STT/LLM/TTS pipeline this cockpit drives.
- [`met4citizen/TalkingHead`](https://github.com/met4citizen/TalkingHead) + HeadAudio by Mika Suominen (MIT) — the 3D avatar + audio-driven lip-sync approach, the ready-head roster, and the Blender avatar pipelines that shaped this project's avatar architecture.
- Ready Player Me, AvatarSDK, Avaturn, VRoid Studio, and the MakeHuman/MPFB community — creators of the bundled example heads.
- Classic visual-novel / Live2D-style talking portraits — the inspiration for the 2D still-image avatar path.

See `NOTICE` for full attribution and per-asset licensing.
