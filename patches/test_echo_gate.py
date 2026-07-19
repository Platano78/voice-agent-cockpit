"""Unit tests for echo_gate.py (server-side echo gate for voice barge-in).

Run from repo root: python3 -m pytest patches/test_echo_gate.py -v

**Two tiers of test in here, and the difference matters.**

*Synthetic tests* (the ``_noise_pcm`` ones) use white noise for both "echo"
and "speech". They are cheap, dependency-free characterizations of the
plumbing -- state machine, ring buffer, fail-open, env flags -- and they are
kept for that. They are explicitly **NOT the deployability contract**. White
noise has essentially no autocorrelation structure, and trusting it as a
proxy for speech is exactly how the original wide-delay-scan defect shipped:
it measured a "0.15-0.2 noise floor" for unrelated audio that real speech
does not obey (real speech scored 0.854, causing a ~72% false-drop rate).
Every synthetic test below is named ``..._synthetic_noise`` to keep that
distinction visible at the point of failure.

*Real-audio tests* (``TestRealAudio``) use the pocket-TTS recordings in
``bench-wavs/`` and are the actual contract: false-drop rate on unrelated
speech, true-echo detection at realistic delays, and the barge-in envelope.
Those WAVs are untracked, so these skip cleanly when absent.

No audio hardware, models, or network needed. The module under test imports
``numpy`` lazily, which is exercised directly here.
"""

from __future__ import annotations

import array
import math
import pathlib
import random
import time
import wave

import pytest

from patches import echo_gate as eg

SR = 16000
# Mic frame the receive path feeds the gate: 512 samples / 1024 bytes / 32 ms.
# NOT a tunable here -- it mirrors `chunk_size_bytes = 512 * 2` at
# websocket_streamer.py:233, where the receive buffer is split for VAD. Every
# measurement in this file is taken at that size, because the wide-scan defect
# is chunk-size dependent (it damps to ~45% at 2048 samples), so measuring at
# any other size would understate it. Enlarging the chunk is not an available
# fix: it is set by the VAD path and would add latency to barge-in detection,
# which is the entire point of the gate.
FRAME = 512
SEND = 1600      # TTS burst the send loop writes, 100 ms
BENCH = pathlib.Path(__file__).resolve().parent.parent / "bench-wavs"


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
    """Wide-scan correlation score, i.e. the max over every alignment.

    NOTE this is no longer what the gate uses to decide anything -- it is the
    *lock acquisition* scan only. Maximizing over the whole buffer is the
    defect the module was rewritten to fix, so a number from this helper says
    nothing about whether a chunk would be dropped. Kept because the lock
    criterion is defined in terms of this peak."""
    numpy = pytest.importorskip("numpy")
    mic = numpy.frombuffer(mic_pcm, dtype=numpy.int16)
    ref = numpy.frombuffer(ref_pcm, dtype=numpy.int16)
    return eg.EchoGate._best_correlation(mic, ref)[0]


def _echo_frame(tts: bytes, delay_samples: int, k: int, gain: float = 1.0) -> bytes:
    """The k'th consecutive mic frame of an echo of `tts` arriving
    `delay_samples` late. Consecutive k values keep the echo's lag constant in
    stream coordinates, which is what lets the gate lock onto it."""
    off = (delay_samples + k * FRAME) * 2
    return _scale_pcm(tts[off : off + FRAME * 2], gain)


def _lock_on_echo(gate: eg.EchoGate, tts: bytes, delay_samples: int = 0, gain: float = 1.0) -> int:
    """Feed enough consecutive echo frames for the gate to acquire its delay
    lock. Returns the next frame index. Until the lock exists the gate passes
    everything by design, so every drop-asserting test must do this first."""
    for k in range(eg._LOCK_CONSECUTIVE):
        gate.feed(_echo_frame(tts, delay_samples, k, gain))
    assert gate._locked_lag is not None, "gate failed to lock onto a pure echo"
    return eg._LOCK_CONSECUTIVE


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


def test_pure_echo_dropped_synthetic_noise(monkeypatch):
    monkeypatch.setenv("VOICE_ECHO_GATE", "1")
    gate = eg.EchoGate()
    tts = _noise_pcm(32000, seed=1)  # 2s of "assistant speech"
    gate.note_playback(tts)

    delay_samples = 1600  # 100ms -- plausible network + client-buffering lag
    k = _lock_on_echo(gate, tts, delay_samples, gain=0.4)  # attenuated, as a mic picks it up

    # Now locked: subsequent echo frames are recognised and dropped.
    assert gate.feed(_echo_frame(tts, delay_samples, k, 0.4)) is False


def test_unlocked_gate_passes_even_a_perfect_echo(monkeypatch):
    """Fail-open while uncalibrated: before the delay lock exists the gate
    must not drop anything, even audio it would later recognise. A missed
    echo is far cheaper than deafness to the user."""
    monkeypatch.setenv("VOICE_ECHO_GATE", "1")
    gate = eg.EchoGate()
    tts = _noise_pcm(32000, seed=1)
    gate.note_playback(tts)

    assert gate._locked_lag is None
    assert gate.feed(_echo_frame(tts, 1600, 0)) is True


# ── (3) user speech, no playback -> passed ───────────────────────────────


def test_user_speech_no_playback_passes_synthetic_noise(monkeypatch):
    monkeypatch.setenv("VOICE_ECHO_GATE", "1")
    gate = eg.EchoGate()

    speech = _noise_pcm(512, seed=2)

    assert gate.feed(speech) is True  # nothing in the reference buffer to gate against


# ── (4) user speech OVER playback -> passed (THE barge-in case) ─────────


def test_speech_over_playback_passes_synthetic_noise(monkeypatch):
    monkeypatch.setenv("VOICE_ECHO_GATE", "1")
    gate = eg.EchoGate()
    tts = _noise_pcm(32000, seed=1)
    gate.note_playback(tts)

    delay = 1600
    k = _lock_on_echo(gate, tts, delay, gain=1.0)

    # A user talking noticeably louder than the echo, mixed into the frame the
    # gate would otherwise recognise. This is the whole point of the feature.
    echo_slice = tts[(delay + k * FRAME) * 2 : (delay + k * FRAME) * 2 + FRAME * 2]
    speech = _noise_pcm(FRAME, seed=99, amplitude=12000)
    assert gate.feed(_mix_pcm(echo_slice, speech)) is True


# ── (4b) the operating envelope, monotonicity ────────────────────────────
# The load-bearing trade-off of this whole approach: a quiet interruption
# under a loud echo does NOT get through, and the threshold sets where the
# cutoff sits. The *numeric* envelope contract lives in
# TestRealAudio.test_barge_in_envelope, measured on real speech -- pinning
# white-noise ratios here would repeat the original mistake of treating
# synthetic audio as the deployment contract.
#
# What this test pins is the property that must hold regardless of signal:
# the envelope is monotonic in loudness. A user who is louder relative to the
# echo must never be *harder* to hear than a quieter one. A regression that
# scrambles the correlation (or the lock) breaks this even when a single
# ratio still happens to land on the right side.
def test_speech_over_playback_envelope_is_monotonic_synthetic_noise(monkeypatch):
    monkeypatch.setenv("VOICE_ECHO_GATE", "1")
    passed_at = []
    for user_gain in (0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00):
        gate = eg.EchoGate()
        tts = _noise_pcm(32000, seed=1)
        gate.note_playback(tts)
        delay = 1600
        k = _lock_on_echo(gate, tts, delay, gain=1.0)

        echo_slice = tts[(delay + k * FRAME) * 2 : (delay + k * FRAME) * 2 + FRAME * 2]
        speech = _noise_pcm(FRAME, seed=99, amplitude=int(8000 * user_gain))
        passed_at.append(gate.feed(_mix_pcm(echo_slice, speech)))

    # Monotonic: once a ratio gets through, every louder ratio must too.
    assert passed_at == sorted(passed_at), f"envelope not monotonic in user loudness: {passed_at}"
    assert passed_at[-1] is True, "a user at 2x the echo must be able to interrupt"
    assert passed_at[0] is False, "a user at 0.25x the echo must still be gated as echo"


# ── (5) silence during playback -> dropped (deliberate choice) ──────────


def test_silence_during_playback_dropped(monkeypatch):
    monkeypatch.setenv("VOICE_ECHO_GATE", "1")
    gate = eg.EchoGate()
    gate.note_playback(_noise_pcm(3200, seed=1))

    # Deliberate policy choice, not a correlation result -- see the comment
    # in echo_gate.EchoGate._score next to the silence-RMS check.
    assert gate.feed(_silence_pcm(512)) is False


# ── (6) delay robustness ─────────────────────────────────────────────────


@pytest.mark.parametrize("delay_samples", [0, 320, 800, 1600, 2400, 3200])
def test_delay_robustness_synthetic_noise(monkeypatch, delay_samples):
    """0-200 ms, the realistic network-jitter-plus-client-buffering range.
    The lock discovers the delay rather than the scan finding it, so what is
    being tested is that acquisition works at any offset in that range."""
    monkeypatch.setenv("VOICE_ECHO_GATE", "1")
    gate = eg.EchoGate()
    tts = _noise_pcm(32000, seed=1)
    gate.note_playback(tts)

    k = _lock_on_echo(gate, tts, delay_samples, gain=0.7)
    # The lock must have recovered the actual delay, not merely some lag that
    # happened to correlate. With all playback pre-loaded the mic runs ahead
    # of the reference, so the recovered lag is -delay.
    assert gate._locked_lag == pytest.approx(-delay_samples, abs=eg._LAG_SEARCH_SAMPLES)
    assert gate.feed(_echo_frame(tts, delay_samples, k, 0.7)) is False


# ── (7) arbitrary frame boundaries ───────────────────────────────────────


def test_arbitrary_frame_boundaries_same_verdict_synthetic_noise(monkeypatch):
    monkeypatch.setenv("VOICE_ECHO_GATE", "1")
    tts = _noise_pcm(32000, seed=1)
    delay = 1600

    # Reference delivered as one contiguous write...
    gate_a = eg.EchoGate()
    gate_a.note_playback(tts)
    k = _lock_on_echo(gate_a, tts, delay, gain=1.0)
    verdict_a = gate_a.feed(_echo_frame(tts, delay, k))

    # ...vs. delivered the way the send loop actually buffers TTS output: a
    # sequence of 3200-byte (100ms) writes that don't divide evenly into the
    # 1024-byte (512-sample) chunks the receive path feeds the gate. This is
    # exactly why the lag lock is anchored in absolute stream coordinates
    # rather than as an offset from the end of the ring buffer.
    gate_b = eg.EchoGate()
    for chunk in _chunks(tts, 3200):
        gate_b.note_playback(chunk)
    k = _lock_on_echo(gate_b, tts, delay, gain=1.0)
    verdict_b = gate_b.feed(_echo_frame(tts, delay, k))

    assert verdict_a is False
    assert verdict_b is False


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


def test_reset_releases_the_delay_lock(monkeypatch):
    """The lock describes one session's audio path; a new session must
    re-acquire it rather than gate against a stale delay."""
    monkeypatch.setenv("VOICE_ECHO_GATE", "1")
    gate = eg.EchoGate()
    tts = _noise_pcm(32000, seed=1)
    gate.note_playback(tts)
    _lock_on_echo(gate, tts, 1600)

    gate.reset()

    assert gate._locked_lag is None


def test_stale_reference_releases_the_delay_lock(monkeypatch):
    """Playback finished long enough ago that the reference expired -- the
    delay estimate belonged to it and goes too."""
    monkeypatch.setenv("VOICE_ECHO_GATE", "1")
    gate = eg.EchoGate()
    tts = _noise_pcm(32000, seed=1)
    gate.note_playback(tts)
    k = _lock_on_echo(gate, tts, 1600)
    assert gate._locked_lag is not None

    gate._last_playback_ts = time.monotonic() - (eg._REF_STALE_S + 1.0)

    assert gate.feed(_echo_frame(tts, 1600, k)) is True  # nothing to gate against
    assert gate._locked_lag is None


# ── VOICE_ECHO_GATE_THRESHOLD override ───────────────────────────────────


def test_threshold_env_override(monkeypatch):
    monkeypatch.setenv("VOICE_ECHO_GATE", "1")
    monkeypatch.setenv("VOICE_ECHO_GATE_THRESHOLD", "0.42")
    gate = eg.EchoGate()
    assert gate.threshold == pytest.approx(0.42)


# ── REAL AUDIO: the actual deployability contract ────────────────────────
# Everything above characterizes behaviour on white noise. These measure it
# on real pocket-TTS speech, which is the only evidence that says anything
# about whether the gate is shippable -- the wide-scan defect passed every
# synthetic test above while dropping ~72% of real user speech.


def _load_wav(path: pathlib.Path):
    numpy = pytest.importorskip("numpy")
    with wave.open(str(path)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, SR)
        return numpy.frombuffer(w.readframes(w.getnframes()), dtype=numpy.int16)


@pytest.fixture(scope="module")
def real_audio():
    """(playback, unrelated_user_speech). Skips when the bench WAVs are
    absent -- they are untracked, so a clean checkout must not fail here."""
    a, b = BENCH / "batch1.wav", BENCH / "batch3.wav"
    if not (a.exists() and b.exists()):
        pytest.skip(f"bench WAVs not present under {BENCH} (untracked)")
    return _load_wav(a), _load_wav(b)


def _rms(x) -> float:
    return float(math.sqrt(sum(float(v) * v for v in x) / len(x))) if len(x) else 0.0


def _drive(gate, playback, mic_kind, other=None, delay_ms=0, gain=1.0):
    """Stream `playback` out in 100 ms bursts while feeding mic frames in, the
    way the pipeline does. `mic_kind` is "echo" (an attenuated copy of
    playback arriving `delay_ms` LATE) or "unrelated" (independent speech
    from `other`).

    The direction matters: the mic at time t hears audio that was played at
    t - delay. Running playback *ahead* of the mic instead would mean the mic
    hears the future, and -- more damagingly for a test -- the audio the echo
    matches would always be the newest sample in the ring buffer, so it could
    never age out and the buffer-depth ceiling would be invisible.

    Every frame is fed -- mic time is contiguous and the gate's lag anchor
    depends on that -- but only non-silent frames count toward the verdict
    rate, since silent frames are dropped by a separate policy rule.
    """
    numpy = pytest.importorskip("numpy")
    delay = int(delay_ms / 1000 * SR)
    sent, verdicts = 0, []
    for k in range(len(playback) // FRAME):
        pos = k * FRAME
        while sent < pos + FRAME and sent < len(playback):
            gate.note_playback(playback[sent : sent + SEND].tobytes())
            sent += SEND
        if mic_kind == "echo":
            src = pos - delay
            if src < 0:
                # Playback has started but its echo has not arrived yet: the
                # mic is picking up room silence. Still fed, to keep the mic
                # coordinate contiguous.
                gate.feed(numpy.zeros(FRAME, dtype=numpy.int16).tobytes())
                continue
            seg = playback[src : src + FRAME]
            if seg.size < FRAME:
                break
            mic = (seg.astype(numpy.float64) * gain).astype(numpy.int16)
        else:
            if pos + FRAME > len(other):
                break
            mic = other[pos : pos + FRAME]
        verdict = gate.feed(mic.tobytes())
        if _rms(mic) >= eg._SILENCE_RMS_FLOOR:
            verdicts.append(verdict)
    return verdicts


class TestRealAudio:
    def test_unrelated_speech_is_not_dropped(self, monkeypatch, real_audio):
        """THE regression test for the wide-scan defect. The shipped gate
        dropped 71.6% of these very frames; anything above a few percent
        means the delay search has gone wide again."""
        monkeypatch.setenv("VOICE_ECHO_GATE", "1")
        playback, speech = real_audio
        gate = eg.EchoGate()

        verdicts = _drive(gate, playback, "unrelated", other=speech)
        assert len(verdicts) > 100, "not enough non-silent frames to be meaningful"

        false_drop = 1.0 - (sum(verdicts) / len(verdicts))
        assert false_drop <= 0.05, f"false-drop rate {false_drop:.1%} on unrelated real speech"

    def test_wide_scan_would_fail_this(self, real_audio):
        """Pins *why* the narrow search exists, so nobody reintroduces the
        wide scan believing it was free. Correlating unrelated real speech
        against a 2 s reference over all ~31.5k lags inflates rho far past
        any usable threshold -- on white noise it would sit near 0.16."""
        numpy = pytest.importorskip("numpy")
        playback, speech = real_audio
        ref = playback[: 2 * SR]

        rhos = [
            eg.EchoGate._best_correlation(speech[i : i + FRAME], ref)[0]
            for i in range(0, len(speech) - FRAME, FRAME)
            if _rms(speech[i : i + FRAME]) >= eg._SILENCE_RMS_FLOOR
        ]
        assert float(numpy.median(rhos)) > 0.7, (
            "unrelated real speech no longer scores high under a full-buffer "
            "scan -- if this fails the defect's premise changed, re-measure"
        )

    @pytest.mark.parametrize("delay_ms", [0, 50, 100, 150, 200])
    def test_true_echo_is_dropped_at_realistic_delays(self, monkeypatch, real_audio, delay_ms):
        """Real echo delay is network jitter plus client audio buffering:
        roughly 100-200 ms. The gate must find and hold the lag across that
        whole range."""
        monkeypatch.setenv("VOICE_ECHO_GATE", "1")
        playback, _ = real_audio
        gate = eg.EchoGate()

        verdicts = _drive(gate, playback, "echo", delay_ms=delay_ms, gain=0.4)
        drop_rate = 1.0 - (sum(verdicts) / len(verdicts))

        # Not 100%: the first few frames of a turn pass by design while the
        # delay lock is still being acquired.
        assert drop_rate >= 0.90, f"only {drop_rate:.1%} of echo dropped at {delay_ms}ms"
        # Positive lag == the echo arrives that many samples after the audio
        # was played, so the recovered value is the delay itself.
        assert gate._locked_lag == pytest.approx(
            int(delay_ms / 1000 * SR), abs=eg._LAG_SEARCH_SAMPLES
        ), "lock did not converge on the true echo delay"

    @pytest.mark.parametrize("delay_ms", [300, 800, 1200, 1800])
    def test_rejection_holds_beyond_the_design_range(self, monkeypatch, real_audio, delay_ms):
        """The gate is *built* for 100-200 ms, but rejection does not decay
        just past that -- it is flat all the way to the buffer depth. Pinned
        so a future change that introduces a gradual decline is visible as a
        behaviour change rather than being mistaken for the known ceiling."""
        monkeypatch.setenv("VOICE_ECHO_GATE", "1")
        playback, _ = real_audio
        gate = eg.EchoGate()

        verdicts = _drive(gate, playback, "echo", delay_ms=delay_ms, gain=0.4)
        drop_rate = 1.0 - (sum(verdicts) / len(verdicts))
        assert drop_rate >= 0.90, f"only {drop_rate:.1%} of echo dropped at {delay_ms}ms"

    def test_beyond_buffer_depth_the_gate_fails_open(self, monkeypatch, real_audio):
        """The real ceiling, and it is a cliff rather than a slope: an echo
        lagging by more than _REF_MAX_SECONDS matches audio that has already
        been trimmed from the ring buffer, so no lock is ever acquired and
        everything passes. Deafness to the user is the failure this module
        exists to avoid, so failing open here is correct -- but it must be a
        known property, not a surprise on a laggy connection."""
        monkeypatch.setenv("VOICE_ECHO_GATE", "1")
        playback, _ = real_audio
        gate = eg.EchoGate()

        over = int(eg._REF_MAX_SECONDS * 1000) + 200
        verdicts = _drive(gate, playback, "echo", delay_ms=over, gain=0.4)

        assert gate._locked_lag is None, "locked onto an echo older than the buffer retains"
        assert all(verdicts), "dropped audio it could not possibly have matched"

    @pytest.mark.parametrize(
        ("user_ratio", "min_pass_rate"),
        [
            (0.25, 0.00),   # far quieter than the echo: still gated
            (0.75, 0.70),   # the near-field win -- was 4% at the old 0.7 threshold
            (1.00, 0.85),
            (2.00, 0.95),
        ],
    )
    def test_barge_in_envelope(self, monkeypatch, real_audio, user_ratio, min_pass_rate):
        """The operating envelope from the module docstring, on real speech:
        how much of a barge-in gets through at a given user:echo amplitude
        ratio at the mic. Raising the threshold widens this."""
        numpy = pytest.importorskip("numpy")
        monkeypatch.setenv("VOICE_ECHO_GATE", "1")
        playback, user = real_audio
        gate = eg.EchoGate()

        delay, barge_at = int(0.1 * SR), 40
        sent, verdicts = 0, []
        for k in range((len(playback) - delay) // FRAME):
            pos = k * FRAME
            while sent < pos + delay + FRAME and sent < len(playback):
                gate.note_playback(playback[sent : sent + SEND].tobytes())
                sent += SEND
            seg = playback[pos + delay : pos + delay + FRAME]
            if seg.size < FRAME:
                break
            echo = seg.astype(numpy.float64) * 0.4
            barging = False
            if k >= barge_at and pos + FRAME <= len(user):
                u = user[pos : pos + FRAME].astype(numpy.float64)
                if _rms(u) > eg._SILENCE_RMS_FLOOR and _rms(echo) > eg._SILENCE_RMS_FLOOR:
                    echo = echo + u * (_rms(echo) / _rms(u)) * user_ratio
                    barging = True
            mic = numpy.clip(echo, -32768, 32767).astype(numpy.int16)
            verdict = gate.feed(mic.tobytes())
            if barging:
                verdicts.append(verdict)

        assert len(verdicts) > 20
        pass_rate = sum(verdicts) / len(verdicts)
        if min_pass_rate == 0.0:
            assert pass_rate <= 0.05, f"quiet barge-in leaked through at {pass_rate:.0%}"
        else:
            assert pass_rate >= min_pass_rate, f"barge-in at {user_ratio}x only passed {pass_rate:.0%}"
