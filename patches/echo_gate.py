"""Server-side echo gate: tell "the assistant's own TTS bleeding back into the
mic" apart from "the user is actually talking", so voice barge-in can work.

The client can't do this reliably -- see the comment at
``webclient/index.html:3234-3238`` explaining why WebAudio-side echo
cancellation isn't trustworthy enough, which is why the client currently
drops all mic frames while the assistant speaks (no barge-in at all). The
server has something the browser doesn't: :class:`WebSocketStreamer` owns
both the mic path (``input_queue``) and the TTS path (``output_queue``) in
one process, so the reference signal (what we just sent) and the incoming
mic audio are in the same place at the same time.

**This is deliberately not acoustic echo cancellation.** AEC tries to
*subtract* the echo cleanly enough to recover a clean residual signal. That's
a much harder problem (needs to model the room, the speaker, the mic, and
adapt continuously) than what barge-in actually requires, which is just a
yes/no *decision*: "is there energy here that the TTS we just played does not
explain?" Getting away with the cheap decision problem instead of the hard
signal-recovery problem is the whole point of this module.

Decision rule
--------------
:meth:`EchoGate.note_playback` appends every chunk of outbound TTS audio,
raw, to a ring buffer (``_ref_buffer``) covering the last
``_REF_MAX_SECONDS`` seconds. :meth:`EchoGate.feed` takes one chunk of
incoming mic audio and computes the best *normalized* cross-correlation
coefficient (rho, in [0, 1]) between that chunk and every same-length window
of the reference buffer -- i.e. it searches every possible delay at once,
because network jitter and client-side audio buffering mean the echo can
arrive at an unpredictable, varying offset from when we sent it (see
"Delay window" below). Normalizing by the energy of both signals makes rho
scale-invariant, so playback volume and mic gain don't matter -- only
*shape* does.

Why this rule handles the barge-in case (the one that matters): if the user
talks *over* the assistant, the mic signal is (approximately) the echo plus
an independent, uncorrelated speech signal. For two independent signals of
comparable energy, textbook correlation algebra says the best-case rho drops
to roughly ``1 / sqrt(1 + speech_energy / echo_energy)`` -- e.g. ~0.71 at
equal energy, ~0.34 if the user's voice is *louder* than the echo -- well
below a pure echo's rho of ~1.0. Verified empirically against the default
threshold in ``test_echo_gate.py`` (case 4) using independent synthesized
noise for "echo" and "speech": pure echo scores ~1.0, equal-energy
speech-over-echo scores ~0.3-0.6, unrelated speech alone (no playback at
all) sits around ~0.15-0.2 noise floor. That's a comfortable margin either
side of the default threshold.

The operating envelope this implies: **the default threshold (0.7) is the
equal-energy crossover.** A parametrized sweep in ``test_echo_gate.py``
(``test_speech_over_playback_envelope``) over user-speech-to-echo amplitude
ratios at the mic, using this exact algorithm, gives:

======================  ========  ========
user : echo amplitude   best rho  verdict
======================  ========  ========
0.25x                   0.969     drop
0.50x                   0.892     drop
0.75x                   0.784     drop
1.00x                   0.697     **pass** (by 0.003 -- the crossover itself)
1.25x                   0.626     pass
1.50x                   0.516     pass
2.00x                   0.429     pass
======================  ========  ========

In plain terms: **the user must be speaking at least as loud as the echo
arriving at the mic to interrupt** -- not merely "any perceptible speech
passes". Below that ratio, the echo's correlation dominates and the gate
drops the chunk. This bites hardest in **near-field configurations** --
phone speaker to phone mic, laptop speaker to laptop mic -- where the echo
path is short and the played-back audio can easily be as loud as, or louder
than, a normally-spoken interruption at the same mic. That is exactly this
project's primary deployment shape, so this is not a corner case to note and
move past. Retuning the threshold against real audio (not synthetic noise)
is a later slice; the sweep above is preserved as an executable contract in
``test_echo_gate.py`` specifically so a future retune shows, in one test
run, exactly what envelope it trades away.

Delay window: the ring buffer cap (``_REF_MAX_SECONDS`` = 2.0s) doubles as
the delay search window -- ``np.correlate(ref, mic, mode="valid")`` computes
the dot product at *every* alignment of the mic chunk within the reference
buffer in one vectorized call, so there's no coarse stride to tune and no
risk of missing the true lag between a stride step. 2 seconds is generously
above any realistic network-jitter-plus-client-buffering lag (typically
tens to a couple hundred milliseconds) while keeping the buffer, and the
O(ref_len * mic_len) correlation cost, small: ~1.8ms measured for a 512
-sample mic chunk against a full 2s/32000-sample reference on a modern CPU
core -- against a chunk that represents 32ms of audio, i.e. a few percent of
one audio-loop tick, nowhere near enough to move the needle on TTFC.

Silence during playback is treated as echo-tail (dropped), not user speech --
see the comment at the check itself for the reasoning. It's a deliberate
choice, not something the correlation math decides.

Design (mirrors ``wakeword_gate.py``; read that module first):

* **Env-flag gated, default OFF.** Controlled by ``VOICE_ECHO_GATE``; unset
  or falsy leaves ``feed()`` passing everything through, so a plain restart
  with no env changes is byte-identical to today's behaviour.
* **Fail open, permanently.** Any exception anywhere in scoring is logged
  once and the gate passes all audio, unconditionally, for the rest of the
  process -- a broken gate must never brick the assistant, and a half-broken
  gate that sometimes still drops real speech is worse than an inert one.
* **Lazy, dependency-free import.** ``numpy`` is only imported inside
  :meth:`EchoGate._score`, on first use that actually needs it -- this
  module (and its tests) import fine with neither ``speech_to_speech`` nor
  ``numpy`` installed. If numpy is missing, the first real :meth:`feed` call
  raises on that import, which the fail-open handler in :meth:`feed` catches
  like any other scoring exception.
* **Calibration affordance.** Correlations at or above ``_NEAR_MISS_FLOOR``
  that don't cross ``self.threshold`` are logged at INFO, mirroring
  wakeword's near-miss logging -- visible at the live log level so the
  threshold can be tuned against a real microphone later.
* **Cross-thread safe.** :meth:`note_playback` runs on the send-loop thread;
  :meth:`feed` runs on the audio-loop thread (receive path). Both touch
  ``_ref_buffer``/``_last_playback_ts`` and take ``_lock`` around that
  access, but ``feed()`` only holds the lock long enough to snapshot the
  buffer -- the (comparatively expensive) correlation itself runs unlocked
  so it never blocks the send loop. ``_fail_open`` is written only from
  :meth:`feed`'s own thread (a single reader/writer, per the above), so
  reading it in :meth:`feed` and :meth:`state` is lock-free by construction.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}

_SAMPLE_RATE = 16000
# Ring buffer cap == delay search window; see module docstring "Delay window".
_REF_MAX_SECONDS = 2.0
_REF_MAX_SAMPLES = int(_REF_MAX_SECONDS * _SAMPLE_RATE)
# If nothing has played in this long, treat the reference as expired rather
# than correlate against stale audio from a much earlier turn.
_REF_STALE_S = 1.5
# int16 RMS below this counts as silence (roughly -56 dBFS) -- comfortably
# above quantization/DC noise, comfortably below any audible speech.
_SILENCE_RMS_FLOOR = 50.0
# Calibration affordance: log a correlation this high even when it doesn't
# cross the (higher) drop threshold, so near-misses leave evidence in the log.
_NEAR_MISS_FLOOR = 0.5


class EchoGate:
    """Scores incoming mic PCM against recently-sent TTS PCM to decide
    whether it's an echo (drop) or real user audio, including user speech
    layered on top of an echo (still pass)."""

    def __init__(self) -> None:
        self.enabled = os.environ.get("VOICE_ECHO_GATE", "").strip().lower() in _TRUTHY
        self.threshold = float(os.environ.get("VOICE_ECHO_GATE_THRESHOLD", "0.7"))
        self._ref_buffer = bytearray()
        self._last_playback_ts = 0.0
        self._fail_open = False
        # Guards _ref_buffer/_last_playback_ts across note_playback() (send-loop
        # thread) vs feed() (audio-loop thread). See module docstring.
        self._lock = threading.Lock()

    def state(self) -> str:
        """Coarse status for the settings panel: ``"off"`` (not enabled),
        ``"gating"`` (a fresh playback reference exists -- mic audio is
        actively being compared against it), or ``"idle"`` (enabled, but
        nothing recent to gate against, so everything passes through)."""
        if not self.enabled:
            return "off"
        with self._lock:
            last_ts = self._last_playback_ts
        if last_ts and (time.monotonic() - last_ts) <= _REF_STALE_S:
            return "gating"
        return "idle"

    def note_playback(self, pcm: bytes) -> None:
        """Record a chunk of outbound TTS audio as it's sent to clients.
        Called from the send loop -- keep this cheap. No-ops when disabled
        so a shipped-off gate never pays even the bytearray append cost."""
        if not self.enabled or not pcm:
            return
        with self._lock:
            self._ref_buffer.extend(pcm)
            self._last_playback_ts = time.monotonic()
            cap_bytes = _REF_MAX_SAMPLES * 2  # int16 mono
            if len(self._ref_buffer) > cap_bytes:
                del self._ref_buffer[: len(self._ref_buffer) - cap_bytes]

    def feed(self, pcm: bytes) -> bool:
        """Feed one chunk of incoming mic PCM. Returns True to pass it on to
        VAD/ASR, False to drop it as echo. Never raises -- any scoring
        failure fails open (returns True) for the rest of the session."""
        if not self.enabled or self._fail_open:
            return True
        try:
            return self._score(pcm)
        except Exception:
            logger.exception("echo gate scoring failed; failing open (passing all audio) for the rest of the session")
            self._fail_open = True
            return True

    def _score(self, pcm: bytes) -> bool:
        with self._lock:
            ref_snapshot = bytes(self._ref_buffer)
            last_ts = self._last_playback_ts

        if not ref_snapshot or (time.monotonic() - last_ts) > _REF_STALE_S:
            return True  # nothing recently played -- nothing to gate against

        import numpy as np  # lazy: see module docstring

        mic = np.frombuffer(pcm, dtype=np.int16)
        if mic.size == 0:
            return True

        mic_rms = float(np.sqrt(np.mean(mic.astype(np.float64) ** 2)))
        if mic_rms < _SILENCE_RMS_FLOOR:
            # Silence while TTS is (recently) playing carries nothing worth
            # passing to ASR either way -- treat it as an echo-tail/inter-word
            # gap rather than user speech. This is a deliberate policy choice
            # (either answer is "safe" here), not a result of the correlation
            # math below.
            return False

        ref = np.frombuffer(ref_snapshot, dtype=np.int16)
        rho = self._best_correlation(mic, ref)
        if rho >= self.threshold:
            logger.debug("echo gate: rho=%.3f (dropped)", rho)
            return False
        if rho >= _NEAR_MISS_FLOOR:
            logger.info("echo gate near miss: rho=%.3f (passed, below threshold %.2f)", rho, self.threshold)
        return True

    @staticmethod
    def _best_correlation(mic: Any, ref: Any) -> float:
        """Best normalized cross-correlation of `mic` against every
        same-length window of `ref`, i.e. every possible delay at once.
        Returns a value in [0, 1] where 1.0 means some window of `ref`
        matches `mic`'s shape exactly (up to scale) and 0.0 means no
        alignment resembles it at all. Isolated as its own static method so
        it can be monkeypatched to fail deliberately in tests, and so the
        (only mildly expensive) numeric work is easy to spot separately from
        the bookkeeping in :meth:`_score`."""
        import numpy as np  # lazy: see module docstring

        frame_len = mic.size
        if ref.size < frame_len:
            return 0.0

        mic_f = mic.astype(np.float64)
        ref_f = ref.astype(np.float64)
        # Dot product of `mic` against every same-length window of `ref` --
        # one vectorized call covers the whole delay search window (see
        # module docstring "Delay window"), no per-lag Python loop needed.
        raw = np.correlate(ref_f, mic_f, mode="valid")
        # Energy of each of those same-length ref windows, via a cumulative
        # sum (O(n)) rather than recomputing a sum over `frame_len` samples
        # per window.
        csum = np.concatenate(([0.0], np.cumsum(ref_f * ref_f)))
        ref_energy = csum[frame_len:] - csum[:-frame_len]
        mic_energy = float(np.dot(mic_f, mic_f))
        denom = np.sqrt(ref_energy * mic_energy) + 1e-9
        rho = np.abs(raw) / denom
        return float(np.max(rho))

    def reset(self) -> None:
        """Re-arm for a new session: drop the playback reference. Unlike
        ``wakeword_gate.WakewordGate.reset``, there's no per-session state to
        preserve on success here -- but ``_fail_open`` is deliberately NOT
        cleared, mirroring that same method: a scorer that already proved
        broken this run stays fail-open for the rest of the process rather
        than silently going back to gating on a detector that just failed."""
        with self._lock:
            self._ref_buffer.clear()
            self._last_playback_ts = 0.0
