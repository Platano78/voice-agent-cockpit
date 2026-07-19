"""Unit tests for echo_gate.py (server-side echo gate for voice barge-in).

Run from repo root: python3 -m pytest patches/test_echo_gate.py -v

No audio hardware, models, or network needed: every fixture below is
synthesized with the stdlib (``array``/``random``), and the module under
test only imports ``numpy`` lazily inside ``EchoGate._score`` /
``EchoGate._best_correlation``, which is exercised directly here (numpy is
already a dependency of this repo's pipeline, so we don't skip-guard it, but
signal synthesis itself never needs it).
"""

from __future__ import annotations

import array
import random
import time

import pytest

from patches import echo_gate as eg


def _noise_pcm(num_samples: int, amplitude: int = 8000, seed: int = 0) -> bytes:
    """Deterministic pseudo-random int16 PCM -- stands in for "some audio",
    echo or speech, without needing a real recording. Two different seeds are
    uncorrelated with each other, which is exactly the property the
    speech-over-echo test below needs."""
    rng = random.Random(seed)
    return array.array("h", [rng.randint(-amplitude, amplitude) for _ in range(num_samples)]).tobytes()


def _silence_pcm(num_samples: int) -> bytes:
    return b"\x00\x00" * num_samples


def _scale_pcm(pcm: bytes, factor: float) -> bytes:
    """Attenuate (or amplify) int16 PCM by `factor`, clipped to int16 range --
    models a mic picking up TTS output at a different volume than it was sent."""
    samples = array.array("h")
    samples.frombytes(pcm)
    scaled = array.array("h", [max(-32768, min(32767, int(s * factor))) for s in samples])
    return scaled.tobytes()


def _mix_pcm(a: bytes, b: bytes) -> bytes:
    """Sample-wise sum of two equal-length int16 PCM buffers, clipped -- models
    a mic picking up echo and the user's own speech at the same time."""
    sa = array.array("h")
    sa.frombytes(a)
    sb = array.array("h")
    sb.frombytes(b)
    n = min(len(sa), len(sb))
    mixed = array.array("h", [max(-32768, min(32767, sa[i] + sb[i])) for i in range(n)])
    return mixed.tobytes()


def _chunks(pcm: bytes, size: int):
    for i in range(0, len(pcm), size):
        yield pcm[i : i + size]


def _rho(mic_pcm: bytes, ref_pcm: bytes) -> float:
    """Compute the same correlation score EchoGate._score would use,
    bypassing the RMS-silence short-circuit -- lets tests assert on the
    actual number (with explicit headroom) instead of just the pass/drop
    boolean, which can otherwise hide a knife-edge margin."""
    numpy = pytest.importorskip("numpy")
    mic = numpy.frombuffer(mic_pcm, dtype=numpy.int16)
    ref = numpy.frombuffer(ref_pcm, dtype=numpy.int16)
    return eg.EchoGate._best_correlation(mic, ref)


# ── (1) disabled by default ──────────────────────────────────────────────


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VOICE_ECHO_GATE", raising=False)
    gate = eg.EchoGate()
    assert gate.enabled is False

    # Directly populate the reference buffer (bypassing note_playback(), which
    # itself no-ops while disabled) so this exercises feed()'s own disabled
    # short-circuit, not just an empty-buffer pass-through.
    tone = _noise_pcm(1600, seed=1)
    gate._ref_buffer.extend(tone)
    gate._last_playback_ts = time.monotonic()

    assert gate.feed(tone[:1024]) is True  # a perfect echo -- still passed


# ── (2) pure echo -> dropped ─────────────────────────────────────────────


def test_pure_echo_dropped(monkeypatch):
    monkeypatch.setenv("VOICE_ECHO_GATE", "1")
    gate = eg.EchoGate()
    tts = _noise_pcm(32000, seed=1)  # 2s of "assistant speech"
    gate.note_playback(tts)

    delay_samples = 4000  # 250ms -- plausible network + client-buffering lag
    echo = _scale_pcm(tts[delay_samples * 2 : delay_samples * 2 + 1024], 0.4)  # attenuated, as picked up by a mic

    assert gate.feed(echo) is False


# ── (3) user speech, no playback -> passed ───────────────────────────────


def test_user_speech_no_playback_passes(monkeypatch):
    monkeypatch.setenv("VOICE_ECHO_GATE", "1")
    gate = eg.EchoGate()

    speech = _noise_pcm(512, seed=2)

    assert gate.feed(speech) is True  # nothing in the reference buffer to gate against


# ── (4) user speech OVER playback -> passed (THE barge-in case) ─────────


def test_speech_over_playback_passes(monkeypatch):
    monkeypatch.setenv("VOICE_ECHO_GATE", "1")
    gate = eg.EchoGate()
    tts = _noise_pcm(32000, seed=1)
    gate.note_playback(tts)

    echo_slice = tts[4000 * 2 : 4000 * 2 + 512 * 2]
    # 1.5x the echo's amplitude, not 1.0x: at equal energy this rho lands at
    # 0.697 against the default 0.7 threshold -- a 0.003 margin that flips
    # with any rounding difference or signal-generator tweak (see
    # test_speech_over_playback_envelope for that exact knife-edge point,
    # pinned deliberately). This test instead proves the feature works with
    # real headroom: a user talking noticeably louder than the echo.
    speech = _noise_pcm(512, seed=99, amplitude=12000)
    mixed = _mix_pcm(echo_slice, speech)

    rho = _rho(mixed, tts)
    assert rho < gate.threshold - 0.1  # comfortable margin, not a coin flip
    assert gate.feed(mixed) is True  # this is the whole point of the feature


# ── (4b) the operating envelope: user must be >= as loud as the echo ─────
# This pins the table in the echo_gate module docstring as an executable
# contract -- default threshold 0.7 is the equal-energy crossover, so the
# gate only lets a barge-in through once the user's voice, at the mic,
# matches or exceeds the echo's amplitude. Below that ratio the echo's own
# correlation dominates and the chunk is dropped. That's the load-bearing
# trade-off of this whole approach: a quiet interruption under a loud echo
# (the common near-field phone-speaker/phone-mic and laptop-speaker/
# laptop-mic case) will NOT get through. If anyone retunes
# VOICE_ECHO_GATE_THRESHOLD later, this test says exactly what moves.
@pytest.mark.parametrize(
    ("user_gain", "expect_pass"),
    [
        (0.25, False),
        (0.50, False),
        (0.75, False),
        (1.00, True),  # the crossover itself -- passes, but only just
        (1.25, True),
        (1.50, True),
        (2.00, True),
    ],
)
def test_speech_over_playback_envelope(monkeypatch, user_gain, expect_pass):
    monkeypatch.setenv("VOICE_ECHO_GATE", "1")
    gate = eg.EchoGate()
    tts = _noise_pcm(32000, seed=1)
    gate.note_playback(tts)

    echo_slice = tts[4000 * 2 : 4000 * 2 + 512 * 2]
    speech = _noise_pcm(512, seed=99, amplitude=int(8000 * user_gain))
    mixed = _mix_pcm(echo_slice, speech)

    rho = _rho(mixed, tts)
    assert (rho < gate.threshold) is expect_pass
    assert gate.feed(mixed) is expect_pass


# ── (5) silence during playback -> dropped (deliberate choice) ──────────


def test_silence_during_playback_dropped(monkeypatch):
    monkeypatch.setenv("VOICE_ECHO_GATE", "1")
    gate = eg.EchoGate()
    gate.note_playback(_noise_pcm(3200, seed=1))

    # Deliberate policy choice, not a correlation result -- see the comment
    # in echo_gate.EchoGate._score next to the silence-RMS check.
    assert gate.feed(_silence_pcm(512)) is False


# ── (6) delay robustness ─────────────────────────────────────────────────


@pytest.mark.parametrize("delay_samples", [0, 100, 4000, 16000, 31000])
def test_delay_robustness(monkeypatch, delay_samples):
    monkeypatch.setenv("VOICE_ECHO_GATE", "1")
    gate = eg.EchoGate()
    tts = _noise_pcm(32000, seed=1)  # 2s -- spans the whole search window
    gate.note_playback(tts)

    frame_samples = 512
    echo = _scale_pcm(tts[delay_samples * 2 : delay_samples * 2 + frame_samples * 2], 0.7)
    assert len(echo) == frame_samples * 2  # sanity: didn't run off the end of tts

    assert gate.feed(echo) is False


# ── (7) arbitrary frame boundaries ───────────────────────────────────────


def test_arbitrary_frame_boundaries_same_verdict(monkeypatch):
    monkeypatch.setenv("VOICE_ECHO_GATE", "1")
    tts = _noise_pcm(32000, seed=1)
    echo = tts[4000 * 2 : 4000 * 2 + 512 * 2]

    # Reference delivered as one contiguous write...
    gate_a = eg.EchoGate()
    gate_a.note_playback(tts)
    verdict_a = gate_a.feed(echo)

    # ...vs. delivered the way the send loop actually buffers TTS output: a
    # sequence of 3200-byte (100ms) writes that don't divide evenly into the
    # 1024-byte (512-sample) chunks the receive path feeds the gate.
    gate_b = eg.EchoGate()
    for chunk in _chunks(tts, 3200):
        gate_b.note_playback(chunk)
    verdict_b = gate_b.feed(echo)

    assert verdict_a is False
    assert verdict_b is False
    assert verdict_a == verdict_b


# ── (8) fail open ─────────────────────────────────────────────────────────


def test_fail_open_on_scoring_exception(monkeypatch):
    monkeypatch.setenv("VOICE_ECHO_GATE", "1")
    gate = eg.EchoGate()
    gate.note_playback(_noise_pcm(3200, seed=1))

    def boom(mic, ref):
        raise RuntimeError("numpy blew up")

    monkeypatch.setattr(gate, "_best_correlation", boom)

    speech = _noise_pcm(512, seed=2)  # loud enough to reach the correlation step
    assert gate.feed(speech) is True  # must not raise
    assert gate._fail_open is True

    # Permanent for the rest of the session: even an exact echo now passes,
    # with no further monkeypatching in effect.
    exact_echo = bytes(gate._ref_buffer[:1024])
    assert gate.feed(exact_echo) is True


# ── (9) ring buffer is bounded ────────────────────────────────────────────


def test_ring_buffer_bounded(monkeypatch):
    monkeypatch.setenv("VOICE_ECHO_GATE", "1")
    gate = eg.EchoGate()
    chunk = _noise_pcm(1600, seed=3)  # 100ms per call

    for _ in range(500):  # 50s of "TTS" hammered in, far past the 2s cap
        gate.note_playback(chunk)

    cap_bytes = eg._REF_MAX_SAMPLES * 2
    assert len(gate._ref_buffer) <= cap_bytes


# ── state() ───────────────────────────────────────────────────────────────


def test_state_off_when_disabled(monkeypatch):
    monkeypatch.delenv("VOICE_ECHO_GATE", raising=False)
    gate = eg.EchoGate()
    assert gate.state() == "off"


def test_state_idle_with_no_reference(monkeypatch):
    monkeypatch.setenv("VOICE_ECHO_GATE", "1")
    gate = eg.EchoGate()
    assert gate.state() == "idle"


def test_state_gating_with_fresh_reference(monkeypatch):
    monkeypatch.setenv("VOICE_ECHO_GATE", "1")
    gate = eg.EchoGate()
    gate.note_playback(_noise_pcm(1600, seed=1))
    assert gate.state() == "gating"


# ── reset() ───────────────────────────────────────────────────────────────


def test_reset_clears_reference_but_not_fail_open(monkeypatch):
    monkeypatch.setenv("VOICE_ECHO_GATE", "1")
    gate = eg.EchoGate()
    gate.note_playback(_noise_pcm(1600, seed=1))
    gate._fail_open = True  # simulate a scorer that already broke this session

    gate.reset()

    assert len(gate._ref_buffer) == 0
    assert gate.state() == "idle"
    assert gate._fail_open is True  # stays failed-open across sessions, like wakeword_gate


# ── VOICE_ECHO_GATE_THRESHOLD override ───────────────────────────────────


def test_threshold_env_override(monkeypatch):
    monkeypatch.setenv("VOICE_ECHO_GATE", "1")
    monkeypatch.setenv("VOICE_ECHO_GATE_THRESHOLD", "0.42")
    gate = eg.EchoGate()
    assert gate.threshold == pytest.approx(0.42)
