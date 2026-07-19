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

import json
import logging
import pathlib
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
    def __init__(self, instructions=None):
        self.session = types.SimpleNamespace(instructions=instructions, tools=None)
        self.chat = types.SimpleNamespace(reset=lambda: None)


def _make_brain_control(tmp_path, runtime_config=None, **kwargs):
    brains_path = tmp_path / "brains.json"
    brains_path.write_text("{}")
    return brain_control.BrainControl(
        llm_handler=types.SimpleNamespace(model_name="test-model"),
        runtime_config=runtime_config or _FakeRuntimeConfig(),
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


# ── persona persistence ─────────────────────────────────────────────────
#
# The persona editor worked but was in-memory only: every restart reverted to
# the CLI `--init_chat_prompt` default, so the only durable way to change it
# was editing `ExecStart` over SSH. These cover the sidecar file that fixes it.

CLI_DEFAULT = "You are the shipped default."


def _persona_env(tmp_path, monkeypatch):
    path = tmp_path / "persona_dir" / "persona.json"
    monkeypatch.setenv("VOICE_PERSONA_FILE", str(path))
    return path


def _restart(tmp_path, **kwargs):
    """Rebuild BrainControl the way a service restart would: a fresh runtime
    config carrying only the CLI default, nothing else in memory."""
    return _make_brain_control(tmp_path, runtime_config=_FakeRuntimeConfig(instructions=CLI_DEFAULT), **kwargs)


def test_persona_survives_a_restart(tmp_path, monkeypatch):
    _persona_env(tmp_path, monkeypatch)
    bc = _restart(tmp_path)

    ack = bc._config_set({"persona": "You are a laconic ship's computer."})

    assert ack["ok"] is True
    assert ack["persona"] == "You are a laconic ship's computer."
    assert ack["persona_persisted"] is True

    revived = _restart(tmp_path)
    assert revived.runtime_config.session.instructions == "You are a laconic ship's computer."
    # The CLI default is still the fallback target, not overwritten by the load.
    assert revived.default_persona == CLI_DEFAULT


def test_clearing_persona_restores_the_cli_default_and_deletes_the_file(tmp_path, monkeypatch):
    path = _persona_env(tmp_path, monkeypatch)
    bc = _restart(tmp_path)
    bc._config_set({"persona": "custom"})
    assert path.is_file()

    ack = bc._config_set({"persona": ""})

    assert ack["ok"] is True
    assert ack["persona"] == CLI_DEFAULT
    assert ack["persona_persisted"] is False
    assert not path.exists()
    assert _restart(tmp_path).runtime_config.session.instructions == CLI_DEFAULT


def test_persisted_persona_takes_precedence_over_the_cli_default(tmp_path, monkeypatch):
    path = _persona_env(tmp_path, monkeypatch)
    path.parent.mkdir(parents=True)
    path.write_text('{"version": 1, "persona": "persisted wins"}')

    state = _restart(tmp_path)._config_state()

    assert state["persona"] == "persisted wins"
    assert state["default_persona"] == CLI_DEFAULT
    assert state["persona_persisted"] is True


def test_missing_persona_file_falls_back_to_the_default(tmp_path, monkeypatch):
    _persona_env(tmp_path, monkeypatch)

    bc = _restart(tmp_path)

    assert bc.runtime_config.session.instructions == CLI_DEFAULT
    assert bc._config_state()["persona_persisted"] is False


def test_corrupt_persona_file_does_not_raise(tmp_path, monkeypatch, caplog):
    path = _persona_env(tmp_path, monkeypatch)
    path.parent.mkdir(parents=True)
    path.write_text('{"version": 1, "persona": "truncated mid-writ')

    with caplog.at_level(logging.WARNING, logger=brain_control.__name__):
        bc = _restart(tmp_path)

    assert bc.runtime_config.session.instructions == CLI_DEFAULT
    assert "unreadable persona file" in caplog.text


def test_empty_and_non_object_persona_files_fall_back(tmp_path, monkeypatch):
    path = _persona_env(tmp_path, monkeypatch)
    path.parent.mkdir(parents=True)

    for body in ("", "   ", "null", '"a bare string"', "[]", "{}", '{"persona": ""}'):
        path.write_text(body)
        assert brain_control.load_persona() is None, body
        assert _restart(tmp_path).runtime_config.session.instructions == CLI_DEFAULT


def test_unreadable_persona_file_falls_back(tmp_path, monkeypatch):
    # A directory where the file should be: open() raises IsADirectoryError
    # (an OSError), which must be caught like any other read failure. Works as
    # root, unlike a chmod-000 file.
    path = _persona_env(tmp_path, monkeypatch)
    path.mkdir(parents=True)

    assert brain_control.load_persona() is None
    assert _restart(tmp_path).runtime_config.session.instructions == CLI_DEFAULT


def test_oversized_persona_on_disk_is_ignored(tmp_path, monkeypatch):
    path = _persona_env(tmp_path, monkeypatch)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"version": 1, "persona": "x" * (brain_control.PERSONA_MAX_CHARS + 1)}))

    assert brain_control.load_persona() is None
    assert _restart(tmp_path).runtime_config.session.instructions == CLI_DEFAULT


def test_oversized_persona_is_rejected_without_persisting(tmp_path, monkeypatch):
    path = _persona_env(tmp_path, monkeypatch)
    bc = _restart(tmp_path)

    ack = bc._config_set({"persona": "x" * (brain_control.PERSONA_MAX_CHARS + 1)})

    assert ack["ok"] is False
    assert "character cap" in ack["error"]
    assert not path.exists()
    assert bc.runtime_config.session.instructions == CLI_DEFAULT


def test_control_characters_are_rejected(tmp_path, monkeypatch):
    _persona_env(tmp_path, monkeypatch)
    bc = _restart(tmp_path)

    ack = bc._config_set({"persona": "sneaky\x1b[31m escape"})

    assert ack["ok"] is False
    assert "control characters" in ack["error"]
    assert bc.runtime_config.session.instructions == CLI_DEFAULT


def test_newlines_and_tabs_are_allowed(tmp_path, monkeypatch):
    _persona_env(tmp_path, monkeypatch)
    bc = _restart(tmp_path)

    multiline = "You are terse.\n\tRules:\r\n- one\n- two"
    assert bc._config_set({"persona": multiline})["ok"] is True
    assert _restart(tmp_path).runtime_config.session.instructions == multiline


def test_save_persona_leaves_no_partial_file(tmp_path, monkeypatch):
    path = _persona_env(tmp_path, monkeypatch)
    brain_control.save_persona("first")

    assert json.loads(path.read_text())["persona"] == "first"
    # The temp file the atomic write goes through is always cleaned up, so a
    # later load can never pick up a half-written sibling.
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]


def test_save_persona_failure_does_not_raise(tmp_path, monkeypatch):
    # Parent path is a FILE, so mkdir/open both fail -- persistence is
    # best-effort and must never take down a live config_set.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    monkeypatch.setenv("VOICE_PERSONA_FILE", str(blocker / "persona.json"))

    brain_control.save_persona("anything")

    bc = _restart(tmp_path)
    assert bc.runtime_config.session.instructions == CLI_DEFAULT
    assert bc._config_state()["persona_persisted"] is False


def test_persona_path_is_env_overridable_outside_the_package(tmp_path, monkeypatch):
    """Same ruling-1 placement as the voices/ sidecar -- outside the package,
    so a pip reinstall can't reach it."""
    path = _persona_env(tmp_path, monkeypatch)
    assert brain_control.persona_path() == path

    monkeypatch.delenv("VOICE_PERSONA_FILE")
    assert brain_control.persona_path() == pathlib.Path("~/speech-to-speech/persona.json").expanduser()
