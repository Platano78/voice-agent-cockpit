# Camera-vision lane — mid-conversation "what do you see" (spec)

**Goal:** the user asks "what do you see / read this / what am I holding", the LLM calls a `look`
tool, the local Gemma-12B (:8084, multimodal) describes the live camera frame, spoken back. Borrowed
from hf-realtime-voice's camera-snapshot pattern; adapted to this stack's server-side tool model.

## Status of the pieces (2026-07-10)
- ✅ **Vision backend** — Gemma-12B at `:8084` is multimodal; proven on OCR ("HELLO MONARCH 42") and
  scene description (red circle / blue rect / yellow triangle, positions + colors correct) via the
  standard OpenAI `image_url` content part. No model work needed.
- ✅ **Server tool** — `/home/platano/voice-tools/_look.py` (staged **dormant**, `_`-prefixed so the
  drop-in loader skips it). Reads `$VOICE_CAMERA_FRAME` (default `/dev/shm/voice_camera_frame.jpg`,
  must be <10 s fresh) → POST to `:8084` vision → TTS-safe ≤600-char reply. Tested 3 ways: live frame
  → correct description; stale frame → "camera view is stale"; no frame → "camera isn't on". Never
  raises. **Activate: `mv _look.py look.py` + click Reload tools (or `config_set{reload_tools:true}`).**
- ⬜ **Client capture** (`webclient/index.html`) — NOT built (no getUserMedia anywhere today).
- ⬜ **Streamer frame-write** (`websocket_streamer.py`) — NOT built.

## Remaining work (two files + activation)

### Slice A — client camera capture (`webclient/index.html`)
- Add a **camera toggle** button near the PTT controls (OFF by default — privacy).
- On enable: `getUserMedia({video:{width:640,facingMode:"environment"}})` into a hidden `<video>`; a
  hidden `<canvas>` grabs a frame. On disable: stop all tracks, stop the interval.
- While enabled, every **~1.5 s**: draw the video frame to canvas at ~512px long-edge,
  `canvas.toDataURL("image/jpeg", 0.8)`, strip the `data:` prefix, and send over the existing ws as a
  **text control frame**: `ws.send(JSON.stringify({ type: "camera_frame", data: "<b64>" }))`. Reuse the
  same text-frame path the `config_set` messages already use — do NOT touch the binary/audio path.
- Immersive-theme note: the camera toggle must not disturb the `.avatar-immersive` layout; place it in
  the PTT control cluster, scoped like the other controls.

### Slice B — streamer frame-write (`websocket_streamer.py`)
- The text-frame branch in `_handle_client` already dispatches `config_get`/`config_set`. Add a
  `camera_frame` type → base64-decode `data` → **atomically** write to `$VOICE_CAMERA_FRAME`
  (`/dev/shm/...` tmpfs — no disk wear, auto-clears on reboot): write a temp file + `os.replace`. Do the
  write via `asyncio.to_thread` so it stays off the audio loop. Do NOT reply/echo. Cap payload size
  (e.g. reject >2 MB) and rate (drop if <1 s since last write) to protect the loop.

### Slice C — activate + verify
- `mv /home/platano/voice-tools/_look.py look.py`; Reload tools; confirm `look` shows in `tools_armed`.
- End-to-end (hands-on, needs camera+mic on the HTTPS cockpit): enable camera, hold up an object, ask
  "what do you see?" → the `look` tool fires and the Monarch speaks the description. Camera off →
  "can't see anything" (graceful). Verify the audio path is unregressed and no frames are sent while
  the toggle is off (privacy gate — check the ws frames in devtools).

## Design rulings (already decided)
- Camera **off by default**, explicit toggle, frames sent **only** while on (privacy).
- ~1.5 s cadence + ~512px JPEG q0.8 — enough for "what do you see", cheap on the vision call.
- Frame lives in `/dev/shm` (tmpfs); the tool's 10 s freshness check means "camera off" degrades on its
  own with no extra teardown signalling.
- Server-side tool (not client-side execution) — matches the existing `voice_tools.py` model; the only
  new client responsibility is *uploading frames*, not executing the tool.
