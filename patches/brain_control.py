from __future__ import annotations

import json
import logging
from typing import Any, Optional

import httpx
from openai import OpenAI

from speech_to_speech.LLM.base_openai_compatible_language_model import BaseOpenAICompatibleHandler
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
    ) -> None:
        self.llm_handler = llm_handler
        self.runtime_config = runtime_config
        self.brains_path = brains_path
        self.brains: dict[str, dict[str, Any]] = self._load_brains(brains_path)
        self.active_brain = "coder"
        self.tts_handler = tts_handler
        self.cockpit = cockpit
        # Captured at construction time, before any config_set — this IS the
        # args-class init_chat_prompt default. Empty persona restores this,
        # it never means "no system prompt".
        self.default_persona = runtime_config.session.instructions or ""
        self.tools_armed = 0
        try:
            runtime_config.session.tools = voice_tools.TOOL_DEFS
            self.tools_armed = len(voice_tools.TOOL_DEFS)
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
            return {"type": "config_ack", "ok": False, "error": f"unknown type: {msg_type}"}
        except Exception as e:
            logger.exception("BrainControl.handle failed")
            return {"type": "config_ack", "ok": False, "error": str(e)}

    def _config_state(self) -> dict[str, Any]:
        return {
            "type": "config_state",
            "active_brain": self.active_brain,
            "persona": self.runtime_config.session.instructions or "",
            "voice": (self.tts_handler.voice or "") if self.tts_handler else "",
            "voices": self._predefined_voices() if self.tts_handler else [],
            "tools_armed": self.tools_armed,
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
        if "brain" in msg:
            ok, error = self._set_brain(msg["brain"])
            if not ok:
                return {"type": "config_ack", "ok": False, "error": error}
        if "voice" in msg:
            ok, error = self._set_voice(msg["voice"])
            if not ok:
                return {"type": "config_ack", "ok": False, "error": error}
        if "persona" in msg:
            persona = msg["persona"]
            self.runtime_config.session.instructions = persona if persona else (self.default_persona or None)
        if msg.get("reset_chat"):
            self.runtime_config.chat.reset()

        return {
            "type": "config_ack",
            "ok": True,
            "active_brain": self.active_brain,
            "model": self.llm_handler.model_name,
            "persona": self.runtime_config.session.instructions or "",
            "voice": (self.tts_handler.voice or "") if self.tts_handler else "",
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
        if name not in self._predefined_voices():
            return False, f"unknown voice: {name}"

        try:
            # Build the new state first so a failed load never leaves the
            # handler with a half-swapped voice.
            new_state = self.tts_handler.model.get_state_for_audio_prompt(name)
        except Exception as e:
            logger.warning("BrainControl: voice load failed for %s: %s", name, e)
            return False, f"voice load failed: {name}"

        self.tts_handler.voice_state = new_state
        self.tts_handler.voice = name
        return True, ""
