"""Wakeword join-deaf gate (ShayneP-borrow, see
knowledge-base/research-notes/shaynep-local-voice-ai-2026-07-12.md).

While armed, the pipeline stays deaf: incoming mic audio is scored against a
cheap openWakeWord ONNX model instead of being pushed onto ``input_queue``.
Only once the wake phrase is detected does audio start flowing to VAD/ASR.
Deploy-safe: disabled unless ``VOICE_WAKE_WORD=1`` (see
``websocket_streamer.WebSocketStreamer``), so a plain restart leaves
behaviour unchanged.

Design:

* **Lazy, dependency-free import.** ``openwakeword`` is only imported inside
  :meth:`WakewordGate._load_model`, on first use -- this module (and its
  tests) can be imported with neither ``speech_to_speech`` nor
  ``openwakeword`` installed.
* **Fail open.** Any exception while loading the model or scoring a frame is
  logged once and treated as an immediate, permanent wake for the rest of
  the session -- a broken detector must never brick the assistant.
* **Calibration affordance.** Any score above 0.05 is logged at DEBUG so the
  threshold can be tuned against a real microphone.
* **Arbitrary frame boundaries.** ``feed()`` accumulates raw bytes into an
  internal buffer and only scores complete 1280-sample (80ms) frames --
  WebSocket frames arrive as multiples of 512 samples, not 1280, so a
  remainder must survive across calls.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}

_FRAME_SAMPLES = 1280  # 80ms @ 16kHz, openWakeWord's native hop size
_FRAME_BYTES = _FRAME_SAMPLES * 2  # int16 mono
_SCORE_LOG_FLOOR = 0.05  # calibration affordance: log anything above the noise floor


class WakewordGate:
    """Scores raw int16 PCM against an openWakeWord model until the
    configured wake phrase is heard, then stays awake for the rest of the
    session."""

    def __init__(self) -> None:
        self.enabled = os.environ.get("VOICE_WAKE_WORD", "").strip().lower() in _TRUTHY
        self._model_arg = os.environ.get("VOICE_WAKE_WORD_MODEL", "hey_jarvis")
        self.threshold = float(os.environ.get("VOICE_WAKE_WORD_THRESHOLD", "0.5"))
        self._awake = False
        self._model: Any = None
        self._load_failed = False
        self._buffer = bytearray()

    @property
    def awake(self) -> bool:
        return self._awake

    @property
    def phrase(self) -> str:
        """Human-readable wake phrase derived from the model name/path, e.g.
        ``"hey_jarvis"`` -> ``"hey jarvis"``."""
        base = os.path.basename(self._model_arg)
        base = re.sub(r"\.(onnx|tflite)$", "", base)
        base = re.sub(r"_v[\d.]+$", "", base)  # strip a pretrained model's version suffix
        return base.replace("_", " ")

    def reset(self) -> None:
        """Re-arm for a new session: clear the buffer, go back to sleep, but
        keep the loaded model (and its own streaming state, if any)."""
        self._buffer.clear()
        # A broken detector must stay failed-open across sessions too -- don't
        # let a new session go deaf again on a model that already proved unusable.
        self._awake = self._load_failed
        if self._model is not None:
            reset_fn = getattr(self._model, "reset", None)
            if callable(reset_fn):
                try:
                    reset_fn()
                except Exception:
                    logger.debug("wakeword model reset() failed", exc_info=True)

    def feed(self, data: bytes) -> float | None:
        """Feed raw int16 PCM bytes; score every complete 1280-sample frame
        accumulated across this and prior calls. Returns the max score seen
        in this call, or ``None`` if no complete frame was consumed."""
        if not self._ensure_model():
            return None

        self._buffer.extend(data)
        max_score: float | None = None
        while len(self._buffer) >= _FRAME_BYTES:
            frame_bytes = bytes(self._buffer[:_FRAME_BYTES])
            del self._buffer[:_FRAME_BYTES]
            frame = np.frombuffer(frame_bytes, dtype=np.int16)

            try:
                predictions = self._model.predict(frame)
            except Exception:
                logger.exception("wakeword predict failed; failing open for the rest of the session")
                self._model = None
                self._load_failed = True
                self._awake = True
                return max_score

            score = float(max(predictions.values())) if predictions else 0.0
            if max_score is None or score > max_score:
                max_score = score
            if score > _SCORE_LOG_FLOOR:
                logger.debug("wake word score: %.3f", score)
            if not self._awake and score >= self.threshold:
                self._awake = True
                logger.info("wake word %r detected (score=%.3f)", self.phrase, score)

        return max_score

    def _ensure_model(self) -> bool:
        """Lazily load the openWakeWord model. Returns False (and fails
        open, permanently, for the session) if loading ever fails."""
        if self._load_failed:
            return False
        if self._model is not None:
            return True
        try:
            self._model = self._load_model()
        except Exception:
            logger.exception("wakeword model load failed; failing open for the rest of the session")
            self._load_failed = True
            self._awake = True
            return False
        return True

    def _load_model(self) -> Any:
        """Construct the openWakeWord ``Model``. Not wrapped in try/except
        itself -- :meth:`_ensure_model` is the single fail-open seam so it
        can be monkeypatched directly in tests."""
        from openwakeword import Model

        try:
            # openwakeword >=0.5: `wakeword_models` + explicit inference framework
            # (defaults to tflite, which we don't ship a runtime for).
            return Model(wakeword_models=[self._model_arg], inference_framework="onnx")
        except TypeError:
            # openwakeword <0.5: `wakeword_model_paths`, always onnx internally.
            return Model(wakeword_model_paths=[self._model_arg])
