from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from typing import Any, Optional

import httpx
from openai import OpenAI

from speech_to_speech.LLM.base_openai_compatible_language_model import BaseOpenAICompatibleHandler
from speech_to_speech import voice_clone
from speech_to_speech import voice_tools

logger = logging.getLogger(__name__)


class BrainControl:
    """Live brain (LLM backend) + persona switch driven by WebSocket control messages.

    Instantiated once per websocket pipeline and wired as
    `websocket_streamer.control_callback`. `handle()` is invoked via
    `asyncio.to_thread` by the streamer, so blocking network calls here are safe
    and do not stall the audio loop. Never raises out of `handle()`.
    """

    def __init__(
        self,
        llm_handler: Any,
        runtime_config: Any,
        brains_path: str,
        tts_handler: Any = None,
        cockpit: Any = None,
        streamer: Any = None,
        tts_queue: Any = None,
    ) -> None:
        self.llm_handler = llm_handler
        self.runtime_config = runtime_config
        self.brains_path = brains_path
        self.brains: dict[str, dict[str, Any]] = self._load_brains(brains_path)
        self.active_brain = "coder"
        self.tts_handler = tts_handler
        self.cockpit = cockpit
        # lm_processed_queue -- lets BrainControl inject an audition TTSInput
        # after a voice change (ruling 8). None keeps every pre-existing
        # call site/test valid -- auditioning is simply unavailable then,
        # same pattern as `streamer`.
        self.tts_queue = tts_queue
        # WebSocketStreamer, for wake-word control (gate lives on it) and
        # broadcasting wakeword_state after a config_set flips it. None keeps
        # every pre-existing call site/test valid -- wake word control is
        # simply unavailable then.
        self.streamer = streamer
        # Captured at construction time, before any config_set — this IS the
        # args-class init_chat_prompt default. Empty persona restores this,
        # it never means "no system prompt".
        self.default_persona = runtime_config.session.instructions or ""
        self._config_lock = threading.Lock()
        self.tools_armed = 0
        try:
            armed = voice_tools.get_tool_defs()
            runtime_config.session.tools = armed
            self.tools_armed = len(armed)
        except Exception as e:
            logger.warning("BrainControl: failed to arm voice tools: %s", e)

    def _load_brains(self, path: str) -> dict[str, dict[str, Any]]:
        with open(path, "r") as f:
            return json.load(f)

    def _resolve_api_key(self, entry: dict[str, Any]) -> Optional[str]:
        """Resolve a brain's API key: literal `api_key`, or lazily parsed from
        `api_key_file` (env-style `VAR=value` lines) looking up `api_key_var`.
        Never logs the resolved value."""
        if entry.get("api_key"):
            return entry["api_key"]
        api_key_file = entry.get("api_key_file")
        api_key_var = entry.get("api_key_var")
        if not api_key_file or not api_key_var:
            return None
        try:
            with open(api_key_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    if key.strip() == api_key_var:
                        return value.strip().strip('"').strip("'")
        except OSError as e:
            logger.warning("BrainControl: failed to read api_key_file %s: %s", api_key_file, e)
        return None

    def _resolve_model(self, base_url: str, model: str, api_key: Optional[str]) -> Optional[str]:
        """GET {base_url}/models to resolve the model id and probe reachability.

        For `model == "auto"`, picks the entry whose `status.value == "loaded"`
        (llama-cpp router schema), falling back to the first listed model id for
        plain OpenAI-style lists. For a fixed model name, the GET is purely a
        reachability probe (fail closed on error). Returns None on any failure.
        """
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        try:
            resp = httpx.get(f"{base_url.rstrip('/')}/models", timeout=3.0, headers=headers)
            resp.raise_for_status()
            data = resp.json().get("data", [])
        except Exception as e:
            logger.warning("BrainControl: model probe failed for %s: %s", base_url, e)
            return None

        if not data:
            return None

        if model != "auto":
            return model

        for entry in data:
            status = entry.get("status")
            if isinstance(status, dict) and status.get("value") == "loaded":
                return entry.get("id")

        return data[0].get("id")

    def _predefined_voices(self) -> list[str]:
        """Names of pocket_tts's built-in preset voices, sorted. Empty on any
        import failure (pocket_tts not installed, upstream rename, etc)."""
        try:
            from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

            return sorted(_ORIGINS_OF_PREDEFINED_VOICES.keys())
        except Exception as e:
            logger.warning("BrainControl: predefined voice list unavailable: %s", e)
            return []

    def handle(self, msg: dict[str, Any]) -> dict[str, Any]:
        try:
            msg_type = msg.get("type")
            if msg_type == "config_get":
                state = self._config_state()
                if self.cockpit is not None:
                    # Push a fresh cockpit_state to all clients so a page that
                    # just loaded renders the card without waiting for the
                    # next poll tick.
                    self.cockpit.broadcast_now()
                return state
            if msg_type == "config_set":
                return self._config_set(msg)
            if msg_type == "voice_clone_begin":
                return self._voice_clone_begin(msg)
            if msg_type == "voice_clone_end":
                return self._voice_clone_end(msg)
            return {"type": "config_ack", "ok": False, "error": f"unknown type: {msg_type}"}
        except Exception as e:
            logger.exception("BrainControl.handle failed")
            return {"type": "config_ack", "ok": False, "error": str(e)}

    def _wake_word_state(self) -> Optional[dict[str, Any]]:
        """Wake-word block for `config_state`/`config_ack` -- None when there's
        no streamer to control (wake word control unavailable)."""
        if self.streamer is None:
            return None
        gate = self.streamer.wakeword_gate
        return {
            "enabled": gate.enabled,
            "state": gate.state(),
            "phrase": gate.phrase,
            "model": gate._model_arg,
            "models": gate.available_models(),
        }

    def _config_state(self) -> dict[str, Any]:
        return {
            "type": "config_state",
            "active_brain": self.active_brain,
            "persona": self.runtime_config.session.instructions or "",
            "default_persona": self.default_persona or "",
            "voice": (self.tts_handler.voice or "") if self.tts_handler else "",
            "voices": self._predefined_voices() if self.tts_handler else [],
            "custom_voices": voice_clone.list_custom_voices() if self.tts_handler else [],
            "tools_armed": self.tools_armed,
            "wake_word": self._wake_word_state(),
            "brains": [
                {
                    "name": name,
                    "label": entry.get("label", name),
                    "model": entry.get("model", ""),
                    "available": entry.get("available", False),
                    "note": entry.get("note", ""),
                }
                for name, entry in self.brains.items()
            ],
        }

    def _config_set(self, msg: dict[str, Any]) -> dict[str, Any]:
        if "permission_respond" in msg:
            return self._handle_permission_respond(msg["permission_respond"])

        with self._config_lock:
            chat_reset = False

            if "brain" in msg:
                prev_brain = self.active_brain
                ok, error = self._set_brain(msg["brain"])
                if not ok:
                    return {"type": "config_ack", "ok": False, "error": error}
                if self.active_brain != prev_brain:
                    chat_reset = True
            if "voice" in msg:
                ok, error = self._set_voice(msg["voice"])
                if not ok:
                    return {"type": "config_ack", "ok": False, "error": error}
            if "voice_delete" in msg:
                ok, error = self._voice_delete(msg["voice_delete"])
                if not ok:
                    return {"type": "config_ack", "ok": False, "error": error}
            if "persona" in msg:
                persona = msg["persona"]
                new_instructions = persona if persona else (self.default_persona or None)
                if new_instructions != self.runtime_config.session.instructions:
                    self.runtime_config.session.instructions = new_instructions
                    chat_reset = True
            if "wake_word" in msg:
                if self.streamer is None:
                    return {"type": "config_ack", "ok": False, "error": "wake word control unavailable"}
                gate = self.streamer.wakeword_gate
                if msg["wake_word"]:
                    gate.enabled = True
                    gate.rearm()
                else:
                    gate.enabled = False
                self.streamer.broadcast_wakeword_state()
            if "wake_word_model" in msg:
                if self.streamer is None:
                    return {"type": "config_ack", "ok": False, "error": "wake word control unavailable"}
                ok, error = self.streamer.wakeword_gate.set_model(msg["wake_word_model"])
                if not ok:
                    return {"type": "config_ack", "ok": False, "error": error}
                self.streamer.broadcast_wakeword_state()
            if msg.get("reset_chat"):
                chat_reset = True
            if chat_reset:
                self.runtime_config.chat.reset()
            if msg.get("reload_tools"):
                try:
                    armed = voice_tools.get_tool_defs()
                    self.runtime_config.session.tools = armed
                    self.tools_armed = len(armed)
                    logger.info("BrainControl: tools reloaded (%d armed)", self.tools_armed)
                except Exception as e:
                    logger.warning("BrainControl: tool reload failed: %s", e)
                    return {"type": "config_ack", "ok": False, "error": f"tool reload failed: {e}"}

            return {
                "type": "config_ack",
                "ok": True,
                "active_brain": self.active_brain,
                "model": self.llm_handler.model_name,
                "persona": self.runtime_config.session.instructions or "",
                "voice": (self.tts_handler.voice or "") if self.tts_handler else "",
                "custom_voices": voice_clone.list_custom_voices() if self.tts_handler else [],
                "tools_armed": self.tools_armed,
                "wake_word": self._wake_word_state(),
                "chat_reset": chat_reset,
            }

    def _handle_permission_respond(self, payload: Any) -> dict[str, Any]:
        if self.cockpit is None:
            return {"type": "config_ack", "ok": False, "error": "hermes cockpit unavailable"}
        if not isinstance(payload, dict):
            return {"type": "config_ack", "ok": False, "error": "invalid permission_respond payload"}
        perm_id = payload.get("id")
        approve = bool(payload.get("approve"))
        if not perm_id:
            return {"type": "config_ack", "ok": False, "error": "permission_respond requires an id"}
        ok, message = self.cockpit.respond(perm_id, approve)
        return {
            "type": "config_ack",
            "ok": ok,
            "permission_respond": {"id": perm_id, "approve": approve, "message": message},
        }

    def _set_brain(self, name: str) -> tuple[bool, str]:
        entry = self.brains.get(name)
        if entry is None:
            return False, f"unknown brain: {name}"
        if not entry.get("available", False):
            return False, entry.get("note") or f"brain not available: {name}"

        base_url = entry["base_url"]
        model = entry.get("model", "auto")
        api_key = self._resolve_api_key(entry)
        resolved = self._resolve_model(base_url, model, api_key)
        if resolved is None:
            return False, f"model probe failed for {name}"

        # Same swap + _extra_body rule as BaseOpenAICompatibleHandler.setup()
        # (disable_thinking=True, no reasoning_effort — matches our CLI default).
        self.llm_handler.client = OpenAI(api_key=api_key or "dummy", base_url=base_url)
        self.llm_handler.model_name = resolved
        self.llm_handler._extra_body = BaseOpenAICompatibleHandler._build_extra_body(base_url, True, None)
        self.active_brain = name
        return True, ""

    def _set_voice(self, name: str) -> tuple[bool, str]:
        if self.tts_handler is None:
            return False, "voice switching unavailable"

        if name in self._predefined_voices():
            source: Any = name
        else:
            # Custom (cloned) voice -- resolve to its sidecar state file.
            path = voice_clone.voice_path(name)
            if not path.is_file():
                return False, f"unknown voice: {name}"
            source = str(path)

        try:
            # Build the new state first so a failed load never leaves the
            # handler with a half-swapped voice.
            new_state = self.tts_handler.model.get_state_for_audio_prompt(source)
        except Exception as e:
            logger.warning("BrainControl: voice load failed for %s: %s", name, e)
            return False, f"voice load failed: {name}"

        self.tts_handler.voice_state = new_state
        self.tts_handler.voice = name
        self._audition(name)
        return True, ""

    def _audition(self, name: str) -> None:
        """After ANY successful voice change, speak a short sample through
        the normal TTS path so the change is audible without a manual test
        (ruling 8). `None` `turn_id`/`turn_revision` pass
        `SpeculativeTurnTracker`'s staleness gate unconditionally (an
        untracked turn id is always treated as latest). Best-effort: never
        raises; silently no-ops with no `tts_queue` wired or when
        `VOICE_AUDITION_TEXT=off`."""
        if self.tts_queue is None:
            return
        text = voice_clone.resolve_audition_text(name)
        if not text:
            return
        try:
            from speech_to_speech.pipeline.messages import TTSInput

            self.tts_queue.put(TTSInput(text=text, turn_id=None, turn_revision=None))
        except Exception as e:
            logger.warning("BrainControl: audition failed for %s: %s", name, e)

    def _voice_delete(self, name: str) -> tuple[bool, str]:
        # Logged on every path: a silent handler made it impossible to tell from the
        # logs whether a user's delete had ever reached the server at all.
        logger.info("BrainControl: voice delete requested for %s", name)
        if self.tts_handler is None:
            logger.warning("BrainControl: voice delete refused for %s: voice switching unavailable", name)
            return False, "voice switching unavailable"
        ok, error = voice_clone.check_delete_allowed(name, self.tts_handler.voice, self._predefined_voices())
        if not ok:
            logger.warning("BrainControl: voice delete refused for %s: %s", name, error)
            return False, error
        ok, error = voice_clone.delete_voice(name)
        if ok:
            logger.info("BrainControl: voice delete succeeded for %s", name)
        else:
            logger.warning("BrainControl: voice delete failed for %s: %s", name, error)
        if ok and self.streamer is not None:
            # Same broadcast as _voice_clone_end -- every client's
            # custom-voices dropdown needs to drop the deleted entry, not
            # just the requesting one.
            self.streamer.broadcast_json(self._config_state())
        return ok, error

    def _voice_clone_begin(self, msg: dict[str, Any]) -> dict[str, Any]:
        name = msg.get("name")
        ext = msg.get("ext")
        size = msg.get("size")

        if self.tts_handler is None:
            return {"type": "voice_clone_result", "ok": False, "name": name, "error": "voice cloning unavailable"}

        ok, error = voice_clone.validate_name(name, self._predefined_voices())
        if not ok:
            return {"type": "voice_clone_result", "ok": False, "name": name, "error": error}
        ok, error = voice_clone.validate_extension(ext)
        if not ok:
            return {"type": "voice_clone_result", "ok": False, "name": name, "error": error}
        ok, error = voice_clone.validate_size(size)
        if not ok:
            return {"type": "voice_clone_result", "ok": False, "name": name, "error": error}

        if not getattr(self.tts_handler.model, "has_voice_cloning", False):
            return {
                "type": "voice_clone_result",
                "ok": False,
                "name": name,
                "error": voice_clone.CLONING_UNAVAILABLE_MSG,
            }

        return {"type": "voice_clone_progress", "stage": "receiving"}

    def _voice_clone_end(self, msg: dict[str, Any]) -> dict[str, Any]:
        name = msg.get("name")
        ext = msg.get("ext") or ""
        raw = msg.get("data")

        if self.tts_handler is None:
            return {"type": "voice_clone_result", "ok": False, "name": name, "error": "voice cloning unavailable"}
        if not isinstance(raw, (bytes, bytearray)) or not raw:
            return {"type": "voice_clone_result", "ok": False, "name": name, "error": "empty upload"}

        # Re-validate at build time -- defense in depth against a client that
        # skipped or raced the begin-time check (chunk bytes are buffered by
        # websocket_streamer independently of that check).
        ok, error = voice_clone.validate_name(name, self._predefined_voices())
        if not ok:
            return {"type": "voice_clone_result", "ok": False, "name": name, "error": error}
        if not getattr(self.tts_handler.model, "has_voice_cloning", False):
            return {
                "type": "voice_clone_result",
                "ok": False,
                "name": name,
                "error": voice_clone.CLONING_UNAVAILABLE_MSG,
            }

        with self._config_lock:
            try:
                wav_bytes = voice_clone.normalize_to_wav(bytes(raw), ext)
            except voice_clone.VoiceCloneError as e:
                return {"type": "voice_clone_result", "ok": False, "name": name, "error": str(e)}

            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
                    fh.write(wav_bytes)
                    tmp_path = fh.name
                # Build the new state before touching the sidecar dir or the
                # live handler -- same fail-safe build-before-swap order as
                # _set_voice (ruling 2/7). Export is atomic (temp file +
                # os.replace) so a failed export never leaves a partial
                # .safetensors for list_custom_voices to serve, and never
                # clobbers a pre-existing GOOD voice on a failed overwrite.
                new_state = self.tts_handler.model.get_state_for_audio_prompt(tmp_path, truncate=True)
                voice_clone.atomic_export_state(new_state, voice_clone.voice_path(name))
            except Exception as e:
                logger.warning("BrainControl: voice_clone build failed for %s: %s", name, e)
                return {"type": "voice_clone_result", "ok": False, "name": name, "error": f"voice build failed: {e}"}
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

            self.tts_handler.voice_state = new_state
            self.tts_handler.voice = name

        self._audition(name)
        if self.streamer is not None:
            self.streamer.broadcast_json(self._config_state())
        return {"type": "voice_clone_result", "ok": True, "name": name, "error": None}
