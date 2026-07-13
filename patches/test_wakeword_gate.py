"""Unit tests for wakeword_gate.py (ShayneP-borrow join-deaf gate).

Run from repo root: python3 -m pytest patches/test_wakeword_gate.py -v

No ``openwakeword`` install required for the fake-scorer tests below: the
module under test only imports ``openwakeword`` lazily inside
``WakewordGate._load_model``, which every test here monkeypatches. The one
real-model smoke test is gated behind ``pytest.importorskip("openwakeword")``
and skips cleanly when the dependency isn't installed.
"""

from __future__ import annotations

import os

import pytest

from patches import wakeword_gate as wg


class FakeModel:
    """Records every frame it's asked to score."""

    def __init__(self, score=0.0):
        self.score = score
        self.frames = []
        self.reset_calls = 0

    def predict(self, frame):
        self.frames.append(frame)
        return {"fake_model": self.score}

    def reset(self):
        self.reset_calls += 1


def _silence(num_samples: int) -> bytes:
    return b"\x00\x00" * num_samples


# ── (a) disabled by default ────────────────────────────────────────────


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VOICE_WAKE_WORD", raising=False)
    gate = wg.WakewordGate()
    assert gate.enabled is False


# ── (b) frame accumulation across misaligned boundaries ────────────────


def test_frames_accumulate_across_boundaries(monkeypatch):
    gate = wg.WakewordGate()
    fake = FakeModel(score=0.0)
    monkeypatch.setattr(gate, "_load_model", lambda: fake)

    # 4 pushes of 512 samples (1024 bytes) each = 2048 samples total.
    # That's exactly 1 complete 1280-sample frame plus a 768-sample remainder
    # that must survive across feed() calls.
    for _ in range(4):
        gate.feed(_silence(512))

    assert len(fake.frames) == 1
    assert len(fake.frames[0]) == wg._FRAME_SAMPLES
    assert fake.frames[0].dtype == wg.np.int16
    assert len(gate._buffer) == (2048 - wg._FRAME_SAMPLES) * 2


# ── (c) threshold crossing wakes + reports score ────────────────────────


def test_threshold_crossing_wakes(monkeypatch):
    monkeypatch.setenv("VOICE_WAKE_WORD_THRESHOLD", "0.5")
    gate = wg.WakewordGate()
    fake = FakeModel(score=0.9)
    monkeypatch.setattr(gate, "_load_model", lambda: fake)

    score = gate.feed(_silence(wg._FRAME_SAMPLES))

    assert gate.awake is True
    assert score == pytest.approx(0.9)


# ── (d) sub-threshold scores never wake ──────────────────────────────────


def test_subthreshold_never_wakes(monkeypatch):
    monkeypatch.setenv("VOICE_WAKE_WORD_THRESHOLD", "0.5")
    gate = wg.WakewordGate()
    fake = FakeModel(score=0.3)
    monkeypatch.setattr(gate, "_load_model", lambda: fake)

    score = gate.feed(_silence(wg._FRAME_SAMPLES))

    assert gate.awake is False
    assert score == pytest.approx(0.3)


# ── (e) loader exception -> fail open ────────────────────────────────────


def test_loader_exception_fails_open(monkeypatch):
    gate = wg.WakewordGate()

    def boom():
        raise RuntimeError("no model file")

    monkeypatch.setattr(gate, "_load_model", boom)

    score = gate.feed(_silence(wg._FRAME_SAMPLES))  # must not raise

    assert gate.awake is True
    assert score is None


# ── (f) predict exception mid-stream -> fail open ────────────────────────


def test_predict_exception_fails_open(monkeypatch):
    gate = wg.WakewordGate()

    class ExplodingModel:
        def predict(self, frame):
            raise RuntimeError("onnxruntime blew up")

    monkeypatch.setattr(gate, "_load_model", lambda: ExplodingModel())

    gate.feed(_silence(wg._FRAME_SAMPLES))  # must not raise

    assert gate.awake is True


# ── (g) reset() re-arms and clears the buffer ────────────────────────────


def test_reset_rearms_and_clears_buffer(monkeypatch):
    monkeypatch.setenv("VOICE_WAKE_WORD_THRESHOLD", "0.5")
    gate = wg.WakewordGate()
    fake = FakeModel(score=0.9)
    monkeypatch.setattr(gate, "_load_model", lambda: fake)

    gate.feed(_silence(wg._FRAME_SAMPLES))
    assert gate.awake is True
    gate.feed(_silence(256))  # leaves a remainder in the buffer

    gate.reset()

    assert gate.awake is False
    assert len(gate._buffer) == 0
    assert fake.reset_calls == 1
    # model stays loaded -- no reload on the next feed()
    assert gate._model is fake


# ── reset() must not undo fail-open across sessions ──────────────────────


def test_reset_after_loader_failure_stays_awake(monkeypatch):
    gate = wg.WakewordGate()

    def boom():
        raise RuntimeError("no model file")

    monkeypatch.setattr(gate, "_load_model", boom)

    gate.feed(_silence(wg._FRAME_SAMPLES))  # fails open
    assert gate.awake is True

    gate.reset()

    assert gate.awake is True  # a broken detector stays failed-open, session or not


def test_reset_after_predict_failure_stays_awake(monkeypatch):
    gate = wg.WakewordGate()

    class ExplodingModel:
        def predict(self, frame):
            raise RuntimeError("onnxruntime blew up")

    monkeypatch.setattr(gate, "_load_model", lambda: ExplodingModel())

    gate.feed(_silence(wg._FRAME_SAMPLES))  # fails open mid-stream
    assert gate.awake is True

    gate.reset()

    assert gate.awake is True
    score = gate.feed(_silence(wg._FRAME_SAMPLES))  # must not raise, and not re-arm deafness
    assert gate.awake is True
    assert score is None


# ── phrase derivation ────────────────────────────────────────────────────


def test_phrase_default():
    gate = wg.WakewordGate()
    assert gate.phrase == "hey jarvis"


def test_phrase_from_custom_path(monkeypatch):
    monkeypatch.setenv("VOICE_WAKE_WORD_MODEL", "/opt/models/my_word_v1.2.onnx")
    gate = wg.WakewordGate()
    assert gate.phrase == "my word"


# ── real openwakeword smoke test (skips without the dependency) ─────────


def test_real_openwakeword_smoke():
    pytest.importorskip("openwakeword")
    os.environ["VOICE_WAKE_WORD"] = "1"
    try:
        gate = wg.WakewordGate()
        assert gate.enabled is True

        rng_bytes = os.urandom(16000 * 2 * 2)  # 2s of noise, int16 mono 16kHz
        for i in range(0, len(rng_bytes), 1024):
            gate.feed(rng_bytes[i : i + 1024])

        assert gate.awake is False
    finally:
        os.environ.pop("VOICE_WAKE_WORD", None)
