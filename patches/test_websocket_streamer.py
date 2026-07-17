"""Unit tests for websocket_streamer.py's voice-clone begin/chunk hardening
(fix #3: don't leave a client stuck on "receiving" if the streamer-side
structural check ever disagrees with BrainControl's).

Run from repo root: python3 -m pytest patches/test_websocket_streamer.py -v

`websockets` is installed in this dev venv, but the `speech_to_speech`
package is not -- its handful of pipeline symbols websocket_streamer.py
imports are stubbed into `sys.modules` before the import below, same
hermetic pattern as `test_reflex_lane.py`/`test_brain_control.py`.
`speech_to_speech.voice_clone` is aliased to the REAL `patches.voice_clone`
module so `UploadManager` here is the actual deployed logic.

Only `_resolve_voice_clone_begin_result` and `_voice_clone_chunk` are
exercised -- both are plain synchronous methods, so no asyncio harness is
needed to cover this fix in isolation.
"""

from __future__ import annotations

import sys
import threading
import types
from queue import Queue


def _install_stubs():
    def mod(name, **attrs):
        # Merge onto an existing sys.modules entry rather than replacing it:
        # multiple test files stub the same speech_to_speech.* submodules,
        # and pytest imports every test module before any test function
        # runs -- a later file's stub would otherwise silently erase
        # attributes an earlier file's LAZY (call-time) import still needs.
        m = sys.modules.get(name)
        if m is None:
            m = types.ModuleType(name)
            sys.modules[name] = m
        for k, v in attrs.items():
            setattr(m, k, v)
        return m

    from patches import voice_clone as real_voice_clone

    pkg = mod("speech_to_speech")
    mod("speech_to_speech.pipeline")
    mod(
        "speech_to_speech.pipeline.control",
        SESSION_END=object(),
        PipelineControlMessage=type("_PipelineControlMessage", (), {}),
        is_control_message=lambda item, kind: False,
    )
    mod("speech_to_speech.pipeline.events", PipelineEvent=type("_PipelineEvent", (), {}))
    mod("speech_to_speech.pipeline.messages", AUDIO_RESPONSE_DONE=object(), PIPELINE_END=object())
    mod(
        "speech_to_speech.pipeline.queue_types",
        AudioInItem=object,
        AudioOutItem=object,
        TextEventItem=object,
    )
    mod("speech_to_speech.turn_stats", turn_stats=types.SimpleNamespace(mark=lambda *a, **kw: None))
    sys.modules["speech_to_speech.voice_clone"] = real_voice_clone
    pkg.voice_clone = real_voice_clone

    class _StubWakewordGate:
        def __init__(self):
            self.enabled = False
            self.awake = False
            self.phrase = "hey_jarvis"

        def reset(self):
            pass

    mod("speech_to_speech.wakeword_gate", WakewordGate=_StubWakewordGate)


_install_stubs()

from patches.websocket_streamer import WebSocketStreamer  # noqa: E402


def _make_streamer(control_callback=None):
    return WebSocketStreamer(
        stop_event=threading.Event(),
        input_queue=Queue(),
        output_queue=Queue(),
        should_listen=threading.Event(),
        control_callback=control_callback,
    )


# ── _resolve_voice_clone_begin_result (fix #3) ──────────────────────────


def test_begin_result_passthrough_on_brain_control_error():
    streamer = _make_streamer()
    msg = {"type": "voice_clone_begin", "name": "my_voice", "ext": ".wav", "size": 100}
    brain_control_result = {"type": "voice_clone_result", "ok": False, "name": "my_voice", "error": "name taken"}

    result = streamer._resolve_voice_clone_begin_result("client-1", msg, brain_control_result)

    assert result is brain_control_result
    # No session should have been opened.
    ok, error, name = streamer._voice_uploads.chunk("client-1", b"x")
    assert not ok
    assert error == "no upload in progress"


def test_begin_result_opens_session_on_success():
    streamer = _make_streamer()
    msg = {"type": "voice_clone_begin", "name": "my_voice", "ext": ".wav", "size": 100}
    brain_control_result = {"type": "voice_clone_progress", "stage": "receiving"}

    result = streamer._resolve_voice_clone_begin_result("client-1", msg, brain_control_result)

    assert result == brain_control_result
    ok, error, name = streamer._voice_uploads.chunk("client-1", b"hello")
    assert (ok, error, name) == (True, "", "my_voice")


def test_begin_result_downgrades_when_local_check_disagrees():
    streamer = _make_streamer()
    # Simulate the local structural check disagreeing with BrainControl's
    # (the exact scenario fix #3 guards against): size is invalid locally.
    msg = {"type": "voice_clone_begin", "name": "my_voice", "ext": ".wav", "size": "not-a-number"}
    brain_control_result = {"type": "voice_clone_progress", "stage": "receiving"}

    result = streamer._resolve_voice_clone_begin_result("client-1", msg, brain_control_result)

    assert result["type"] == "voice_clone_result"
    assert result["ok"] is False
    assert result["name"] == "my_voice"
    assert "invalid upload size" in result["error"]

    # No stuck session left behind -- a follow-up chunk cleanly reports
    # "no upload in progress" instead of silently vanishing.
    ok, error, name = streamer._voice_uploads.chunk("client-1", b"x")
    assert not ok
    assert error == "no upload in progress"


def test_begin_result_non_progress_types_pass_through_unchanged():
    streamer = _make_streamer()
    msg = {"type": "config_set", "voice": "alba"}
    ack = {"type": "config_ack", "ok": True}

    result = streamer._resolve_voice_clone_begin_result("client-1", msg, ack)

    assert result is ack


# ── _voice_clone_chunk (malformed payload / no session) ─────────────────


def test_chunk_rejects_malformed_base64():
    streamer = _make_streamer()
    result = streamer._voice_clone_chunk("client-1", "not valid base64!!!")
    assert result["ok"] is False
    assert result["error"] == "malformed chunk payload"


def test_chunk_without_session_reports_out_of_order():
    streamer = _make_streamer()
    result = streamer._voice_clone_chunk("client-1", "aGVsbG8=")  # "hello"
    assert result == {"type": "voice_clone_result", "ok": False, "name": None, "error": "no upload in progress"}
