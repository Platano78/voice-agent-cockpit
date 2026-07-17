# Custom voice cloning — v1.3.0 plan

<!-- plan-artifact: append-only metadata header -->
- Status: ACTIVE
- Created: 2026-07-17 (orchestrator session, crew discipline)
- Research basis: `docs/research/pocket-tts-voice-cloning-2026-07-17.md` (cited as §N below — every ruling traces there)
- Repo: voice-agent-cockpit @ main (HEAD 2c1277d at spec time)
- Scope: `patches/` + `webclient/index.html` + `README.md` only. Everything else FROZEN.
- Slices: A = server (patch pack), B = webclient UI. One implementing agent per slice.

## Goal (user's words, hard requirements)

Record your own voice in-browser OR upload an audio file in the EXISTING webclient
settings (advanced area) → server builds a pocket_tts voice state → voice appears in the
existing dropdown, persists across reinstalls, auditions on switch. User never touches the
installed folder. Public-repo generic + README'd.

## Design rulings (with WHY)

1. **Persistence = sidecar safetensors dir**: `~/speech-to-speech/voices/{name}.safetensors`,
   overridable via `VOICE_CLONE_DIR`. WHY: mirrors upstream's own packaging of predefined
   voices as precomputed states (§3), uses the blessed `export_model_state` round-trip
   (§1), lives next to `brains.json` which already survives reinstalls. States are
   ~6–8 MB each (§2).
2. **Compute once, load forever**: state is built ONCE at upload (`get_state_for_audio_prompt`
   on the audio, `truncate=True`), exported to safetensors; every later switch loads the
   file via the ungated fast path (§1 branch 1, §2). WHY: build is slow model inference;
   load is fast; loading needs no gated weights (§4) so existing custom voices keep
   working even if HF auth breaks later.
3. **Always `truncate=True` (30 s cap)**: upstream's own server endpoint and export-voice
   CLI both do this (§2, §6). UI copy tells users "10–30 seconds of clean speech" (§2
   quality guidance; no documented minimum exists — the audition is the arbiter).
4. **Upload transport = chunked WS control frames** on the existing control channel:
   `voice_clone_begin` / `voice_clone_chunk` / `voice_clone_end`. WHY: the WS default
   frame cap is 1 MiB (§7) so single-frame won't fit; the control channel already has the
   JSON-frame + `camera_frame` base64 precedent (§7); serve.py is static-only (§7) and a
   separate process that can't reach the model — an HTTP lane would need new IPC.
5. **Normalize every upload through soundfile** to 16-bit mono WAV before handing to
   pocket. WHY: pocket's stdlib WAV path hard-codes int16 and would misdecode float/24-bit
   WAVs (§5); soundfile 0.14.0 is on the box and decodes wav/aiff/flac/ogg/mp3 (§5).
   soundfile missing ⇒ friendly error naming `pip install soundfile`. Accept extensions:
   `.wav .aiff .aif .flac .ogg .mp3`; reject `.webm .m4a .mp4` with "convert to WAV/MP3
   first" (§5 — libsndfile cannot decode them).
6. **Browser recording = raw PCM → WAV client-side, never MediaRecorder**. WHY:
   MediaRecorder emits webm/mp4, both undecodable on the box (§5). Capture Float32 PCM at
   the AudioContext's native rate (do NOT reuse the mic path's 16 kHz decimation — model
   resamples to 24 kHz internally and higher-rate input is strictly more information, §2),
   cap at 30 s (§2), encode 16-bit mono WAV in JS, then the same chunk-upload path.
7. **Cloning gate UX**: at `voice_clone_begin`, if `tts_handler.model.has_voice_cloning`
   is falsy → immediate error: "This install can't build new voices yet: accept the terms
   at https://huggingface.co/kyutai/pocket-tts, run `hf auth login` as the service user,
   then restart the service." (§4 — the live box is currently in this state: no HF token,
   only no-cloning weights cached). Loading already-built custom voices stays available
   regardless (§1/§4).
8. **Audition on switch**: after ANY successful voice change (predefined or custom, switch
   or fresh clone), BrainControl puts
   `TTSInput(text=<audition text>, turn_id=None, turn_revision=None)` onto
   `lm_processed_queue`. WHY: `SpeculativeTurnTracker` treats untracked/None turn ids as
   latest (verified: `_latest_revision.get(turn_id, revision) == revision` with None
   never observed ⇒ True), so the injected sample passes the pocket handler's staleness
   gates; this speaks through the normal TTS→WS audio path with zero new plumbing.
   Text: env `VOICE_AUDITION_TEXT`, default `Hi, I'm {name}. This is how I sound.`
   (`{name}` substituted); `off` disables auditioning. BrainControl gains an optional
   `tts_queue` constructor kwarg (None keeps all existing call sites/tests valid — same
   pattern as `streamer`).
9. **Names**: lowercase `[a-z0-9_-]`, 1–32 chars, must NOT collide with the 26 predefined
   names (§3) — reject with "name taken by a built-in voice". Re-uploading an existing
   custom name overwrites it (explicit UX: client confirms).
10. **Deletion**: `config_set {voice_delete: name}` removes the safetensors. Deleting the
    ACTIVE voice is rejected ("switch to another voice first") — keeps `_set_voice`'s
    fail-safe invariant (§7: state swap only after successful build) trivially intact.
11. **Dropdown**: `config_state` gains `custom_voices: [...]` alongside the existing
    `voices` (predefined) list. Client renders one dropdown (predefined + custom, custom
    suffixed "· custom") and delete affordance only for custom entries. `_set_voice`
    resolves custom names to `Path(VOICE_CLONE_DIR)/{name}.safetensors` and keeps the
    existing build-before-swap order (§7).
12. **README obligations** (§8): new "Custom voices (voice cloning)" section — HF
    terms/auth prerequisite + consent clause verbatim pointer ("voice impersonation or
    cloning without explicit and lawful consent" is prohibited by upstream), accepted
    formats, clean-audio guidance, `VOICE_CLONE_DIR`/`VOICE_AUDITION_TEXT` envs,
    reinstall-safety note. Generic wording — downstream users (§ MEMORY: public repo).

## Protocol (server ⇄ client, JSON text frames on the existing WS)

Client → server:
- `{"type":"voice_clone_begin","name":str,"ext":str,"size":int}` — size = raw bytes,
  cap 25 MB. Single in-flight upload per client; a new begin aborts the previous.
- `{"type":"voice_clone_chunk","data":<b64>}` — ≤512 KiB raw per chunk (b64 ~683 KiB,
  under the 1 MiB frame cap §7). Sequential.
- `{"type":"voice_clone_end"}` — server assembles temp file (original ext, upstream's
  extension-driven detection pattern §6), normalizes (ruling 5), builds state
  (`truncate=True`), `export_model_state` → sidecar dir, switches to it, auditions,
  broadcasts fresh `config_state`.
- `{"type":"config_set","voice_delete":name}` — ruling 10.

Server → requesting client:
- `{"type":"voice_clone_progress","stage":"receiving"|"building"}` (building sent once,
  before the state build starts — builds take seconds-to-tens-of-seconds of CPU).
- `{"type":"voice_clone_result","ok":bool,"name":str,"error":str|null}`.

Reuses the existing `config_get`/`config_set` dispatch seam in `websocket_streamer.py`
(`asyncio.to_thread`, reply-to-sender, §7); chunk frames are buffered on the streamer side
(bytes only, no model work) and the END frame triggers the BrainControl call.

## Slice A — server (patches pack)

Files:
- NEW `patches/voice_clone.py` — dependency-light module (no `speech_to_speech` import;
  lazy `soundfile`/`pocket_tts` imports, like `think_filter.py`/`voice_rules.py`
  conventions): name validation, upload session state machine (begin/chunk/end, size
  caps, abort), soundfile → 16-bit mono WAV normalization, sidecar-dir listing/delete,
  audition-text resolution. Pure logic testable without pocket_tts.
- `patches/brain_control.py` — `_set_voice` extended to custom names (resolve to sidecar
  path, same fail-safe order); `voice_clone_*` handling (gate check ruling 7, build,
  export, switch, audition); `custom_voices` in `config_state`; `voice_delete`;
  new optional `tts_queue` kwarg.
- `patches/websocket_streamer.py` — route `voice_clone_*` frame types to the control
  callback (chunks buffered per-client; binary audio branch untouched).
- `patches/s2s_pipeline.py` — pass `lm_processed_queue` into BrainControl construction.
- NEW `patches/test_voice_clone.py` — pytest, importable standalone (repo convention):
  name rules (valid/invalid/predefined-collision), state machine (happy path, oversize,
  out-of-order chunk, abort-on-new-begin), normalization (skip if soundfile absent),
  delete rules (active-voice rejection), audition text ({name} substitution, off).
- Existing tests in `patches/` must stay green (84 baseline).

HARD GATES (run from repo root, paste outputs verbatim, any failure = fix before report):
- `python3 -m pytest patches/ -q` → ALL pass, ZERO fail (baseline 84 + new).
- `python3 -m py_compile patches/voice_clone.py patches/brain_control.py patches/websocket_streamer.py patches/s2s_pipeline.py` → clean.

## Slice B — webclient (index.html)

Settings gear panel gains an "Advanced — custom voice" collapsible section:
record button (hold-to-record or start/stop, 30 s auto-stop, live seconds counter),
file picker (accept list from ruling 5), name field (validation mirroring ruling 9),
upload progress + "building voice…" state (from `voice_clone_progress`), result toast,
delete button per custom voice. Dropdown merge per ruling 11. Recording per ruling 6.
Mobile-friendly (the phone is a primary client). No framework — vanilla JS matching the
existing file's style.

GATE: manual — but the agent must verify JS syntax (node --check on extracted script or
equivalent) and report exactly what was wired to which protocol frame.

## Checklist (append-only)

- [x] Phase 0 research note written + haiku-verified (PASS, 0 defects, 21/21 citations exact)
- [x] Box unblocked: HF terms accepted, gated weights cached, has_voice_cloning=True proven; E2E smoke: build 1.7s/7.8s clip, state 4.8MB, reload ~0s
- [x] Slice A dispatched / gates green / verified 3-way (haiku found 1 MAJOR: partial-safetensors leak → fixed via atomic_export_state temp+os.replace; +2 orchestrator rulings: voice_delete broadcast, begin-disagreement downgrade; final 147 passed/3 skipped)
- [x] Slice B dispatched / gates green / verified 3-way (haiku live-probed the WAV encoder ±1 LSB + fact-checked README; found stale-recording bug → clear-at-start fix; orchestrator found the never-yielding chunk loop → yield+backpressure fix; ctx-ref cleanup)
- [x] README section (Custom voices: formats, HF gate setup, envs, consent clause — haiku fact-checked against research note)
- [x] Commits: 99c5041 (slice A), e95a371 (slice B + README), d6d840f (apply.sh deploy fix)
- [x] Box deploy: pulled to e95a371+, apply.sh gap caught pre-restart (voice_clone.py missing
      from its cp list — would have crashed brain_control import; fixed as d6d840f), service
      restarted healthy; live WS probe: config_state carries custom_voices, voice_clone_begin
      → progress{receiving} (has_voice_cloning=True live)
- [ ] User ear-test (record → clone → audition on phone) → then tag v1.3.0 + GitHub release
