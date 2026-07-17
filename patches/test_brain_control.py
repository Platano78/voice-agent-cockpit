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

import sys
import types

# ── Stub the speech_to_speech surface brain_control.py imports ─────────


def _install_stubs():
    def mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
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
    _model_arg = "hey_jarvis"

    def state(self):
        return "off"

    def available_models(self):
        return []


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


def test_voice_delete_unavailable_without_tts_handler(tmp_path):
    bc = _make_brain_control(tmp_path, tts_handler=None, streamer=_FakeStreamer())

    ok, error = bc._voice_delete("anything")

    assert ok is False
    assert error == "voice switching unavailable"
