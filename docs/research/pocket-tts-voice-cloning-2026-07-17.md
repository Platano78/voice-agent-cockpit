# pocket-tts voice cloning — research note (Phase 0, custom-voice feature)

Date: 2026-07-17. Sources: **installed source on the box** (`pocket-tts 2.1.0`,
`~/speech-to-speech-main/.venv/.../pocket_tts/`, read first-hand), **upstream repo**
(github.com/kyutai-labs/pocket-tts @ `d108410`, README + pyproject), and **live box
probes** (ai-utility, 2026-07-17). Line numbers refer to the installed 2.1.0 files.
Every design ruling in the custom-voice spec must cite a section of this note.

## 1. The cloning API (verified in installed source)

`TTSModel.get_state_for_audio_prompt(audio_conditioning, truncate=False)`
(`models/tts_model.py:788`) builds a **model state** (FlowLM KV-cache dict) that
captures the speaker. Input branches, in order:

1. **`.safetensors` path/URL** (`tts_model.py:845-851`) → `_import_model_state()`
   direct load. Runs **before** the cloning gate — loading precomputed states is
   never gated. Docstring: "much faster than extracting from an audio file".
2. **Predefined voice name** (`tts_model.py:853-869`) → downloads a precomputed
   embedding from HF (§3).
3. **Audio file path/URL** — gated (§4): raises `VOICE_CLONING_UNSUPPORTED` if
   `has_voice_cloning` is False (`tts_model.py:871-872`). Otherwise:
   `audio_read()` → optional truncate → `convert_audio(...)` → mimi-encode →
   flow-LM prompt → state dict.
4. **`torch.Tensor` `[channels, samples]`** — bypasses the file gate but is useless
   without the cloning weights (§4).

**Save/load round-trip is first-class**: `export_model_state(model_state, dest)`
(`tts_model.py:1047-1052`) writes the state as `.safetensors` (keys flattened to
`"{module}/{key}"`); branch 1 loads it back. The official CLI has the same flow:
`pocket-tts export-voice <audio> <out>` calls `get_state_for_audio_prompt(...,
truncate=True)` then `export_model_state` (`main.py:370-372`). **Compute-once,
cache-as-safetensors is the upstream-blessed persistence pattern.**

`@lru_cache(maxsize=2)` exists only on `_cached_get_state_for_audio_prompt`
(`tts_model.py:781`) — our `brain_control._set_voice` calls the uncached method;
in-process caching is not something we can lean on beyond 2 entries.

## 2. Reference-clip methodology

- **Truncation: `truncate=True` caps at the FIRST 30 s** (`tts_model.py:880-884`,
  `int(30 * sample_rate)` samples). Default is `False` = unbounded (memory risk).
  Kyutai's own server endpoint and `export-voice` CLI both pass `truncate=True`.
- **No documented minimum clip length** (upstream README/docs/code — nothing).
  UNCONFIRMED territory; pick a UX minimum empirically by ear.
- **Sample rate/channels: anything in, 24 kHz mono internally.** Multi-channel is
  downmixed by channel-averaging at load (`data/audio.py:34-35, 51`), then
  `convert_audio(audio, in_rate, 24000, 1)` resamples (`tts_model.py:886-888`).
  Higher-rate cleaner input = strictly more information for the clone; there is no
  benefit to pre-downsampling client-side.
- **Quality guidance (README verbatim):** "We recommend cleaning the sample before
  using it with Pocket TTS, because the audio quality of the sample is also
  reproduced." Mic noise, room echo, and compression artifacts become part of the
  voice.
- State build time: model inference over the clip (the slow path pocket's docstring
  warns about); measured MB-scale states — the 19 English embeddings on the box are
  **5.3–7.8 MB each** (HF cache, `ls -lL`). Budget ~6-8 MB disk per custom voice.

## 3. How the 26 predefined voices are packaged

- Runtime load path: `get_predefined_voice(language, name)` → precomputed
  **`.safetensors` embedding** from the ungated HF repo
  `kyutai/pocket-tts-without-voice-cloning/languages/{language}/embeddings/{name}.safetensors`
  (pinned revision, `utils/utils.py:45-46`). NOT raw audio at runtime.
- `_ORIGINS_OF_PREDEFINED_VOICES` (`utils/utils.py:15-42`, 26 names) maps names to
  their source clips on `kyutai/tts-voices` / `kyutai/pocket-tts` — provenance and
  attribution, not the load path. Per-voice licenses:
  https://huggingface.co/kyutai/tts-voices
- Predefined names only resolve when the model was loaded from a language config
  (`tts_model.py:858-863`) — ours is (default `english`).
- **Implication: our custom voices should mirror this exact pattern** — a directory
  of `{name}.safetensors` states, loaded via branch 1, which works even without the
  gated weights.

## 4. The cloning gate (has_voice_cloning) — box is currently GATED OFF

- Config `english.yaml` names two weight sets: `weights_path` →
  **gated `kyutai/pocket-tts`** (accept-terms repo) and
  `weights_path_without_voice_cloning` → ungated
  `kyutai/pocket-tts-without-voice-cloning`.
- Load logic (`tts_model.py:201-210`): try gated download; **any failure silently
  falls back** to the no-cloning weights and sets `has_voice_cloning = False`.
- Error text when cloning is then attempted (`tts_model.py:52-59`): "go to
  https://huggingface.co/kyutai/pocket-tts and accept the terms, then make sure
  you're logged in locally with `uvx hf auth login`."
- **Live box probe (2026-07-17): NO HF token** (`~/.cache/huggingface/token`
  absent) and only `models--kyutai--pocket-tts-without-voice-cloning` in the HF
  cache → the running service has `has_voice_cloning = False`.
- **Prerequisite for the feature on any install** (README-worthy, applies to
  downstream users too): (1) accept terms at huggingface.co/kyutai/pocket-tts with
  an HF account, (2) `hf auth login` (or `HF_TOKEN`) as the service user,
  (3) restart the service so `load_model()` fetches the gated weights.
- Consent clause (upstream README, must surface in our README): "Prohibited uses
  include, without limitation, voice impersonation or cloning without explicit and
  lawful consent."
- Loading **precomputed** `.safetensors` states (predefined AND our custom ones,
  once built) needs no gate (§1 branch order) — only **building** a state from
  audio does.

## 5. Audio decode reality (formats)

`data/audio.py::audio_read` (module docstring claims av/torchaudio — **stale, the
code uses neither**):

- **`.wav`** → stdlib `wave` + numpy, `dtype=np.int16` hard-coded
  (`audio.py:27-36`). **Assumes 16-bit PCM**: a float32/24-bit/8-bit WAV will be
  misdecoded to garbage or crash — a real footgun for uploads. Non-16-bit WAVs
  must be routed to the soundfile path or converted.
- **Everything else** → `soundfile` (optional dep; **installed on the box:
  soundfile 0.14.0, libsndfile 1.2.2** — probed).
- Box-probed `sf.available_formats()`: **WAV, AIFF, FLAC, OGG (Vorbis), MP3, CAF,
  W64 = YES. MP4/AAC (m4a) = NO. WEBM = NO.** (Opus: no standalone major format;
  Opus-in-Ogg exists as a libsndfile subtype but was not probed — treat as
  unverified until tested with a real file.)
- **Implication for browser recording:** MediaRecorder emits `audio/webm` (Chrome)
  or `audio/mp4` (Safari) — **both undecodable** by the box. Raw-PCM capture
  (AudioWorklet, like the cockpit's existing mic path) wrapped into a 16-bit WAV
  is the only container-safe browser route, and the cockpit already owns that code
  path. Upload of user files: accept wav/aiff/flac/ogg/mp3; reject m4a/webm with a
  clear message.

## 6. Kyutai's own upload reference implementation

`main.py:122-181` (`POST /text_to_speech` on their FastAPI server): mutually
exclusive `voice_url` (name/URL) vs `voice_wav` (multipart upload). Upload path:
write to `NamedTemporaryFile(suffix=<original extension>)` — **extension drives
format detection** — then `get_state_for_audio_prompt(Path(tmp), truncate=True)`,
`finally: os.unlink(tmp)`. No size cap enforced by them; no minimum length check.

## 7. Cockpit-side seams this feature touches (verified in repo/box)

- `PocketTTSHandler` (s2s `TTS/pocket_tts_handler.py`): holds `self.model`,
  `self.voice`, `self.voice_state`; speaks via
  `generate_audio_stream(self.voice_state, text, copy_state=True)`. Default voice
  `alba`, `load_model()` at setup.
- `brain_control._set_voice(name)` (patches pack): validates against
  `_predefined_voices()` (= sorted `_ORIGINS_OF_PREDEFINED_VOICES`), builds the new
  state before swapping (fail-safe order), sets `handler.voice_state` +
  `handler.voice`. `config_state` broadcasts `voice` + `voices[]`;
  webclient dropdown sends `config_set {voice}` (`index.html:2095-2106, 2177-2180`).
- WS transport: JSON text frames; `camera_frame` precedent for base64 payloads
  (`patches/websocket_streamer.py:24-27, 233-234`). **No `max_size` passed to
  `websockets.serve()`** → library default **1 MiB per frame** applies; a 30 s
  16-bit mono WAV at 48 kHz is ~2.9 MB → single-frame upload does NOT fit.
  Chunked frames or an HTTP lane are required (spec decision).
- `webclient/serve.py`: static file server only, no POST surface today.

## 8. Design-relevant conclusions (each cites above)

1. **Persistence**: compute state once at upload → `export_model_state` →
   `<name>.safetensors` in a sidecar dir outside the package (mirrors §3 packaging,
   §1 round-trip; survives reinstall like `brains.json`). Voice switching loads it
   via branch 1 — fast, ungated.
2. **Gating UX**: feature must detect `has_voice_cloning == False` and return a
   friendly "accept terms + hf auth login" message (§4) rather than a stack trace.
   Existing custom voices (already-built states) must keep working regardless (§1).
3. **Formats**: accept wav/aiff/flac/ogg/mp3 uploads (§5); validate WAV bit-depth
   (16-bit stdlib path; route others through soundfile). Reject webm/m4a with
   guidance. Browser recording = raw PCM → WAV, never MediaRecorder containers (§5).
4. **Clip policy**: always pass `truncate=True` (upstream's own choice, §2/§6);
   surface "~10-30 s of clean speech" guidance in UI (§2 quality note); no hard
   minimum — audition (§2: no documented minimum) is the arbiter.
5. **Transport**: 1 MiB WS frame ceiling (§7) means chunked base64 control frames
   or an HTTP upload lane — decide in spec, both are viable; Kyutai's temp-file +
   extension pattern (§6) is the server-side shape either way.
6. **README obligations**: HF terms/auth prerequisite (§4), consent clause (§4),
   format list (§5), clean-audio guidance (§2) — downstream users get the generic
   story.

## 9. Addendum — live smoke test (2026-07-17, box unblocked)

The gate was lifted the same day: HF token installed on the box
(`~/.cache/huggingface/token`, account Platano78), terms accepted for
`kyutai/pocket-tts`, gated english weights predownloaded (219 MB), and a fresh
`load_model()` confirmed **`has_voice_cloning = True`** (live service flips at its
next restart). End-to-end probe of the exact feature mechanism, on the box CPU:

- state build from a 7.8 s 16-bit WAV (`truncate=True`): **1.7 s** → a 30 s clip
  extrapolates to ~6-7 s ("building…" UI phase is seconds, not tens of seconds)
- exported state: 4.8 MB; reload from `.safetensors`: **~0 s**; generation from the
  reloaded state: OK.
- Bonus footgun confirmation for §5: the first probe wrote the reference clip as a
  float32 WAV (scipy default for float arrays) and pocket's stdlib reader failed
  with `wave.Error: unknown format: 3` — non-16-bit-PCM WAVs don't just misdecode,
  float WAVs hard-crash. Ruling 5 (normalize all uploads via soundfile) is
  empirically mandatory.
