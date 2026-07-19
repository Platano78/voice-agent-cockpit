"""Unit tests for websocket_streamer.py's voice-clone begin/chunk hardening
(fix #3: don't leave a client stuck on "receiving" if the streamer-side
structural check ever disagrees with BrainControl's) and for the echo-gate
wiring (send-path `note_playback`, receive-path `feed`, session-boundary
`reset`).

The echo-gate tests here cover *wiring*, not scoring: whether the gate
correctly tells echo from speech is pinned in `test_echo_gate.py` against
real WAVs, and duplicating it here would just couple this file to those
thresholds. So the ON-path tests drive a deterministic stand-in gate, while
the default-OFF test uses the REAL `EchoGate` -- that one is about the
production object being inert when unconfigured, so a stand-in would prove
nothing.

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

import asyncio
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

    from patches import echo_gate as real_echo_gate
    from patches import phone_context as real_phone_context
    from patches import transcript_buffer as real_transcript_buffer
    from patches import voice_clone as real_voice_clone

    pkg = mod("speech_to_speech")
    mod("speech_to_speech.pipeline")
    mod(
        "speech_to_speech.pipeline.control",
        # Real SESSION_END is a PipelineControlMessage; `_send_loop` reads
        # `.kind` off it, so a bare object() is not a faithful enough stub.
        SESSION_END=types.SimpleNamespace(kind="session_end"),
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
    sys.modules["speech_to_speech.phone_context"] = real_phone_context
    pkg.phone_context = real_phone_context
    sys.modules["speech_to_speech.transcript_buffer"] = real_transcript_buffer
    pkg.transcript_buffer = real_transcript_buffer
    # Real module, not a stub: the default-OFF test below asserts the actual
    # shipped EchoGate is inert when VOICE_ECHO_GATE is unset.
    sys.modules["speech_to_speech.echo_gate"] = real_echo_gate
    pkg.echo_gate = real_echo_gate

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


# ── echo gate wiring ────────────────────────────────────────────────────

CHUNK_BYTES = 512 * 2  # one VAD chunk, matching _handle_client's split


class _RecordingGate:
    """Deterministic stand-in: records every call, and drops the mic chunks
    whose index appears in `drop_indices`. Lets the wiring tests assert
    exactly which bytes reached which method without depending on how the
    real correlator scores anything."""

    def __init__(self, drop_indices=()):
        self.enabled = True
        self.played = []
        self.fed = []
        self.resets = 0
        self._drop = set(drop_indices)

    def note_playback(self, pcm):
        self.played.append(pcm)

    def feed(self, pcm):
        idx = len(self.fed)
        self.fed.append(pcm)
        return idx not in self._drop

    def reset(self):
        self.resets += 1

    def state(self):
        return "gating"


class _FakeClient:
    def __init__(self, incoming=()):
        self._incoming = list(incoming)
        self.sent = []

    def __aiter__(self):
        async def gen():
            for message in self._incoming:
                yield message

        return gen()

    async def send(self, data):
        self.sent.append(data)


def _drain(queue):
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


def test_gate_off_is_pure_passthrough(monkeypatch):
    """Default-OFF equivalence, proven against the REAL EchoGate: with
    VOICE_ECHO_GATE unset every chunk reaches input_queue and no playback
    reference is retained, so a restart with no env changes behaves exactly
    as it did before the gate was wired in."""
    monkeypatch.delenv("VOICE_ECHO_GATE", raising=False)
    streamer = _make_streamer()
    assert streamer.echo_gate.enabled is False
    assert streamer.echo_gate.state() == "off"

    # Send path: note_playback is called but must retain nothing.
    client = _FakeClient()
    streamer.clients.add(client)
    asyncio.run(streamer._broadcast_audio(b"\x11\x22" * 800))
    assert streamer.echo_gate._ref_buffer == bytearray()
    assert streamer.echo_gate._ref_total == 0

    # Receive path: every chunk passes, unconditionally.
    for pattern in (b"\x00\x00", b"\x7f\x7f", b"\x01\x02"):
        assert streamer._echo_pass(pattern * 512) is True


def test_gate_off_receive_loop_queues_every_chunk(monkeypatch):
    """The same equivalence at the loop level: three whole chunks in one
    websocket message produce three input_queue puts, byte-identical to the
    unwired behaviour."""
    monkeypatch.delenv("VOICE_ECHO_GATE", raising=False)
    streamer = _make_streamer()
    streamer.should_listen.set()
    payload = b"".join(bytes([i]) * CHUNK_BYTES for i in (1, 2, 3))

    asyncio.run(streamer._handle_client(_FakeClient([payload])))

    queued = _drain(streamer.input_queue)
    assert queued[:3] == [bytes([i]) * CHUNK_BYTES for i in (1, 2, 3)]


def test_gate_on_drops_scored_echo_and_passes_the_rest():
    streamer = _make_streamer()
    streamer.echo_gate = _RecordingGate(drop_indices=(0, 2))
    streamer.should_listen.set()
    chunks = [bytes([i]) * CHUNK_BYTES for i in (10, 11, 12, 13)]

    asyncio.run(streamer._handle_client(_FakeClient([b"".join(chunks)])))

    # The gate saw all four; only the two it passed were enqueued.
    assert streamer.echo_gate.fed == chunks
    queued = _drain(streamer.input_queue)
    assert queued[:2] == [chunks[1], chunks[3]]


def test_note_playback_receives_exactly_the_bytes_sent():
    streamer = _make_streamer()
    streamer.echo_gate = _RecordingGate()
    client = _FakeClient()
    streamer.clients.add(client)
    data = bytes(range(256)) * 8

    asyncio.run(streamer._broadcast_audio(data))

    assert streamer.echo_gate.played == [data]
    assert client.sent == [data]


def test_send_loop_audio_paths_all_record_playback():
    """Drives the real `_send_loop`: the threshold flush (>= MIN_AUDIO_BYTES)
    and the queue-empty flush must each hand their exact payload to
    note_playback."""
    streamer = _make_streamer()
    streamer.echo_gate = _RecordingGate()
    client = _FakeClient()
    streamer.clients.add(client)
    # 4000 bytes: crosses MIN_AUDIO_BYTES (3200) on the threshold path, then
    # the 800-byte remainder leaves via the queue-empty flush.
    streamer.output_queue.put(b"\xab\xcd" * 2000)

    async def drive():
        task = asyncio.create_task(streamer._send_loop())
        await asyncio.sleep(0.1)
        streamer.stop_event.set()
        await task

    asyncio.run(drive())

    assert streamer.echo_gate.played == client.sent
    assert b"".join(streamer.echo_gate.played) == b"\xab\xcd" * 2000


def test_no_audio_send_site_bypasses_the_playback_funnel():
    """Structural guard: `_send_loop` must not send audio directly. Every
    outbound-audio path goes through `_broadcast_audio` so none can be added
    later that forgets `note_playback` -- a hole in the reference signal
    means real echo correlates against nothing and is passed through as user
    speech, which is the failure this whole slice exists to prevent."""
    import inspect

    source = inspect.getsource(WebSocketStreamer._send_loop)
    assert "client.send(data)" not in source
    assert source.count("_broadcast_audio") == 4


def test_reset_fires_when_the_last_client_disconnects():
    streamer = _make_streamer()
    streamer.echo_gate = _RecordingGate()
    streamer.should_listen.set()

    asyncio.run(streamer._handle_client(_FakeClient([])))

    assert streamer.echo_gate.resets == 1


def test_reset_failure_cannot_break_disconnect():
    """Fail-open at the wiring layer: a gate that raises on reset must not
    stop the session from ending cleanly."""

    class _ExplodingGate(_RecordingGate):
        def reset(self):
            raise RuntimeError("boom")

    streamer = _make_streamer()
    streamer.echo_gate = _ExplodingGate()
    streamer.should_listen.set()

    asyncio.run(streamer._handle_client(_FakeClient([])))

    assert _drain(streamer.input_queue)  # SESSION_END still enqueued


def test_feed_failure_passes_audio_through():
    """Fail-open at the wiring layer: a gate whose feed raises must not
    silence the user."""

    class _ExplodingGate(_RecordingGate):
        def feed(self, pcm):
            raise RuntimeError("boom")

    streamer = _make_streamer()
    streamer.echo_gate = _ExplodingGate()

    assert streamer._echo_pass(b"\x01\x02" * 512) is True


def test_note_playback_failure_still_sends_audio():
    class _ExplodingGate(_RecordingGate):
        def note_playback(self, pcm):
            raise RuntimeError("boom")

    streamer = _make_streamer()
    streamer.echo_gate = _ExplodingGate()
    client = _FakeClient()
    streamer.clients.add(client)

    asyncio.run(streamer._broadcast_audio(b"\x05\x06" * 100))

    assert client.sent == [b"\x05\x06" * 100]
