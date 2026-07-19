"""Unit tests for brain_control.py's voice-delete broadcast (fix #2).

Run from repo root: python3 -m pytest patches/test_brain_control.py -v

The real ``speech_to_speech`` package is not installed in this repo, so the
handful of symbols brain_control.py imports from it are stubbed into
``sys.modules`` before the import below -- same hermetic pattern as
``test_reflex_lane.py``. ``speech_to_speech.voice_clone`` is aliased to the
REAL ``patches.voice_clone`` module (not a stub) so these tests exercise the
actual deployed logic, not a mock of it.
"""

from __future__ import annotations

import logging
import sys
import types

# ── Stub the speech_to_speech surface brain_control.py imports ─────────


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
    mod("speech_to_speech.LLM")
    mod("speech_to_speech.LLM.base_openai_compatible_language_model", BaseOpenAICompatibleHandler=type("_Stub", (), {}))
    vt = mod("speech_to_speech.voice_tools", get_tool_defs=lambda: [], execute=lambda name, kwargs: "")
    pkg.voice_tools = vt
    sys.modules["speech_to_speech.voice_clone"] = real_voice_clone
    pkg.voice_clone = real_voice_clone


_install_stubs()

from patches import brain_control  # noqa: E402

PREDEFINED_UNAVAILABLE_NOTE = "pocket_tts not installed -- _predefined_voices() returns [] safely"


# ── fakes ────────────────────────────────────────────────────────────────


class _FakeWakewordGate:
    enabled = False
    phrase = "hey_jarvis"
    # Mirrors the real gate's contract: `model_name` is the stripped display form and
    # is always one of `available_models()`, even when the raw arg is a custom path.
    model_name = "my_wake"
    _model_arg = "/opt/models/my_wake_v1.0.onnx"

    def state(self):
        return "off"

    def available_models(self):
        return ["hey_jarvis", "my_wake"]


class _FakeStreamer:
    def __init__(self):
        self.broadcasts: list[dict] = []
        self.wakeword_gate = _FakeWakewordGate()

    def broadcast_json(self, payload):
        self.broadcasts.append(payload)


class _FakeTTSHandler:
    def __init__(self, voice=None):
        self.voice = voice
        self.voice_state = None


class _FakeRuntimeConfig:
    def __init__(self):
        self.session = types.SimpleNamespace(instructions=None, tools=None)
        self.chat = types.SimpleNamespace(reset=lambda: None)


def _make_brain_control(tmp_path, **kwargs):
    brains_path = tmp_path / "brains.json"
    brains_path.write_text("{}")
    return brain_control.BrainControl(
        llm_handler=types.SimpleNamespace(model_name="test-model"),
        runtime_config=_FakeRuntimeConfig(),
        brains_path=str(brains_path),
        **kwargs,
    )


# ── wake-word block: reported model must be selectable in the dropdown ──


def test_wake_word_state_model_is_one_of_the_offered_models(tmp_path):
    """The settings panel marks the active entry by comparing `model` to each of
    `models`. Reporting the raw VOICE_WAKE_WORD_MODEL (a path, for a custom model)
    matched nothing, so the dropdown silently showed the FIRST entry as active and
    misreported the live wake phrase."""
    bc = _make_brain_control(tmp_path, streamer=_FakeStreamer())

    wake = bc._wake_word_state()

    assert wake["model"] in wake["models"]
    assert wake["model"] == "my_wake"


def test_wake_word_state_none_without_a_streamer(tmp_path):
    assert _make_brain_control(tmp_path)._wake_word_state() is None


# ── voice_delete broadcast (fix #2) ─────────────────────────────────────


def test_voice_delete_broadcasts_config_state_on_success(tmp_path, monkeypatch):
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    (voices_dir / "my_voice.safetensors").write_bytes(b"fake state")
    monkeypatch.setenv("VOICE_CLONE_DIR", str(voices_dir))

    streamer = _FakeStreamer()
    bc = _make_brain_control(tmp_path, tts_handler=_FakeTTSHandler(voice="other_voice"), streamer=streamer)

    ok, error = bc._voice_delete("my_voice")

    assert (ok, error) == (True, "")
    assert len(streamer.broadcasts) == 1
    assert streamer.broadcasts[0]["type"] == "config_state"
    assert "my_voice" not in streamer.broadcasts[0]["custom_voices"]


def test_voice_delete_no_broadcast_on_active_voice_rejection(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICE_CLONE_DIR", str(tmp_path / "voices"))
    streamer = _FakeStreamer()
    bc = _make_brain_control(tmp_path, tts_handler=_FakeTTSHandler(voice="active_voice"), streamer=streamer)

    ok, error = bc._voice_delete("active_voice")

    assert ok is False
    assert "switch to another voice" in error
    assert streamer.broadcasts == []


def test_voice_delete_no_broadcast_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICE_CLONE_DIR", str(tmp_path / "voices"))
    streamer = _FakeStreamer()
    bc = _make_brain_control(tmp_path, tts_handler=_FakeTTSHandler(voice="other_voice"), streamer=streamer)

    ok, error = bc._voice_delete("nonexistent")

    assert ok is False
    assert streamer.broadcasts == []


def test_voice_delete_none_streamer_is_safe(tmp_path, monkeypatch):
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    (voices_dir / "my_voice.safetensors").write_bytes(b"fake state")
    monkeypatch.setenv("VOICE_CLONE_DIR", str(voices_dir))

    bc = _make_brain_control(tmp_path, tts_handler=_FakeTTSHandler(voice="other_voice"), streamer=None)

    # Must not raise despite streamer=None.
    ok, error = bc._voice_delete("my_voice")

    assert (ok, error) == (True, "")


def test_voice_delete_logs_request_and_success(tmp_path, monkeypatch, caplog):
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    (voices_dir / "my_voice.safetensors").write_bytes(b"fake state")
    monkeypatch.setenv("VOICE_CLONE_DIR", str(voices_dir))

    bc = _make_brain_control(tmp_path, tts_handler=_FakeTTSHandler(voice="other_voice"), streamer=_FakeStreamer())

    with caplog.at_level(logging.INFO, logger=brain_control.__name__):
        ok, _ = bc._voice_delete("my_voice")

    assert ok is True
    assert "voice delete requested for my_voice" in caplog.text
    assert "voice delete succeeded for my_voice" in caplog.text


def test_voice_delete_logs_refusal_reason(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("VOICE_CLONE_DIR", str(tmp_path / "voices"))
    bc = _make_brain_control(tmp_path, tts_handler=_FakeTTSHandler(voice="active_voice"), streamer=_FakeStreamer())

    with caplog.at_level(logging.INFO, logger=brain_control.__name__):
        ok, error = bc._voice_delete("active_voice")

    assert ok is False
    assert "voice delete requested for active_voice" in caplog.text
    assert f"voice delete refused for active_voice: {error}" in caplog.text


def test_voice_delete_unavailable_without_tts_handler(tmp_path):
    bc = _make_brain_control(tmp_path, tts_handler=None, streamer=_FakeStreamer())

    ok, error = bc._voice_delete("anything")

    assert ok is False
    assert error == "voice switching unavailable"
