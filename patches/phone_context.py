"""Ambient phone context: browser-supplied location/timezone/battery state.

Motivation: the model spun 14 tool rounds trying to guess the user's
location for an air-quality question. The webclient (shipped separately)
sends opt-in ``{"type":"phone_context", "lat":.., "lon":.., "accuracy":..,
"tz":.., "battery_pct":.., "charging":..}`` text frames over the existing
WebSocket; :func:`update` stores the last-known values here, keyed by
nothing but the process (single in-flight session, same as the rest of this
patch pack). :func:`location` gives tools (``get_weather``) a fallback when
the user doesn't name a place; :func:`apply_ambient` appends a one-line
ambient-context sentence to the LLM request, mirroring
``voice_rules.apply_system_rules``'s copy-don't-mutate, append-to-system-
message contract.

Dependency-light like ``voice_rules.py``/``think_filter.py``: no
``speech_to_speech`` import, so this module (and its tests) is importable
standalone. ``requests`` is imported lazily inside the reverse-geocode
helper only -- the only network call this module makes -- keeping module
import itself free of non-stdlib dependencies.

Thread-safe: the WebSocket thread writes via :func:`update`, the LLM thread
reads via :func:`location`/:func:`snapshot`/:func:`apply_ambient`. A single
``threading.Lock`` guards the module-level state dict.

``VOICE_PHONE_CONTEXT=off`` (case-insensitive) disables the whole feature:
:func:`update` becomes a no-op and every reader returns ``None``.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from threading import Lock
from typing import Any, Callable

logger = logging.getLogger(__name__)

_DEFAULT_LOCATION_MAX_AGE_S = 1800.0
_GEOCODE_TIMEOUT_S = 5.0
_GEOCODE_URL = "https://nominatim.openstreetmap.org/reverse"

_lock = Lock()
_state: dict[str, Any] | None = None

_geocode_cache: dict[tuple[float, float], str] = {}


def _enabled() -> bool:
    """Read VOICE_PHONE_CONTEXT fresh on every call (cheap; keeps a runtime
    toggle -- and tests -- working without a module reload)."""
    return os.environ.get("VOICE_PHONE_CONTEXT", "").strip().lower() != "off"


def _is_number(value: Any) -> bool:
    # bool is a subclass of int in Python -- exclude it explicitly so
    # {"lat": true} doesn't slip through a numeric-range check.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _valid_lat(value: Any) -> bool:
    return _is_number(value) and -90 <= value <= 90


def _valid_lon(value: Any) -> bool:
    return _is_number(value) and -180 <= value <= 180


def _valid_accuracy(value: Any) -> bool:
    return _is_number(value) and value > 0


def _valid_tz(value: Any) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 64


def _valid_battery_pct(value: Any) -> bool:
    return _is_number(value) and 0 <= value <= 100


def _valid_charging(value: Any) -> bool:
    return isinstance(value, bool)


_VALIDATORS: dict[str, Callable[[Any], bool]] = {
    "lat": _valid_lat,
    "lon": _valid_lon,
    "accuracy": _valid_accuracy,
    "tz": _valid_tz,
    "battery_pct": _valid_battery_pct,
    "charging": _valid_charging,
}


def update(data: dict) -> bool:
    """Validate and store a phone-context update, merged onto whatever was
    already known (a partial update -- e.g. battery-only -- keeps the rest
    of the last-known state). Each field is independently optional; an
    invalid individual field is dropped and logged rather than rejecting
    the whole update. Rejects (no-op, returns False) a non-dict payload or
    one where nothing valid came through. Returns whether anything was
    stored. No-op when VOICE_PHONE_CONTEXT=off.
    """
    if not _enabled():
        return False
    if not isinstance(data, dict):
        return False
    validated: dict[str, Any] = {}
    for key, validator in _VALIDATORS.items():
        if key not in data:
            continue
        value = data[key]
        if validator(value):
            validated[key] = value
        else:
            logger.debug("phone_context.update: rejected field %r=%r", key, value)
    if not validated:
        return False
    global _state
    with _lock:
        merged = dict(_state) if _state else {}
        merged.update(validated)
        merged["ts"] = time.time()
        _state = merged
    return True


def _get_state() -> dict[str, Any] | None:
    with _lock:
        return dict(_state) if _state is not None else None


def location(max_age_s: float = _DEFAULT_LOCATION_MAX_AGE_S) -> tuple[float, float] | None:
    """Return the last-known (lat, lon), or None if disabled, never set, or
    older than ``max_age_s``."""
    if not _enabled():
        return None
    state = _get_state()
    if state is None:
        return None
    lat, lon, ts = state.get("lat"), state.get("lon"), state.get("ts")
    if lat is None or lon is None or ts is None:
        return None
    if time.time() - ts > max_age_s:
        return None
    return (lat, lon)


def snapshot() -> dict[str, Any] | None:
    """Return a copy of the full last-known state (plus ``age_s``), or None
    if disabled or never set. For future tools/UI use."""
    if not _enabled():
        return None
    state = _get_state()
    if state is None:
        return None
    state["age_s"] = time.time() - state.get("ts", time.time())
    return state


def _place_from_geocode_payload(payload: Any) -> str | None:
    """Extract a short, spoken-friendly place name from a Nominatim jsonv2
    reverse-geocode response. None on any unexpected shape."""
    if not isinstance(payload, dict):
        return None
    address = payload.get("address") or {}
    locality = (
        address.get("city") or address.get("town") or address.get("village") or address.get("hamlet") or address.get("county")
    )
    parts = [p for p in (locality, address.get("state"), address.get("country")) if p]
    if parts:
        return ", ".join(parts[:2])
    display_name = payload.get("display_name")
    if isinstance(display_name, str) and display_name.strip():
        return display_name.split(",")[0].strip()
    return None


def _reverse_geocode(lat: float, lon: float) -> str:
    """Reverse-geocode (lat, lon) to a short place name via Nominatim,
    cached in-module keyed by coords rounded to 3 decimals. Fails soft to
    'latitude X, longitude Y' on any network/parse error -- never raises.
    """
    key = (round(lat, 3), round(lon, 3))
    cached = _geocode_cache.get(key)
    if cached is not None:
        return cached
    fallback = f"latitude {lat:.3f}, longitude {lon:.3f}"
    place = None
    try:
        import requests

        resp = requests.get(
            _GEOCODE_URL,
            params={"format": "jsonv2", "lat": lat, "lon": lon},
            headers={"User-Agent": "voice-agent-cockpit"},
            timeout=_GEOCODE_TIMEOUT_S,
        )
        resp.raise_for_status()
        place = _place_from_geocode_payload(resp.json())
    except Exception as e:
        logger.debug("phone_context: reverse geocode failed: %r", e)
    result = place or fallback
    _geocode_cache[key] = result
    return result


def _local_time_str(tz_name: str | None) -> str | None:
    """HH:MM in ``tz_name``, or None if unset/invalid (unknown zoneinfo
    key) -- ambient_line omits the time clause entirely in that case."""
    if not tz_name:
        return None
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo(tz_name)).strftime("%H:%M")
    except Exception:
        return None


def ambient_line() -> str | None:
    """One-line ambient context for the LLM request, or None when disabled
    or there's no fresh-enough location."""
    if not _enabled():
        return None
    loc = location()
    if loc is None:
        return None
    lat, lon = loc
    place = _reverse_geocode(lat, lon)
    state = _get_state() or {}
    time_str = _local_time_str(state.get("tz"))
    if time_str:
        return f"The user's approximate location is {place}; their local time is {time_str} ({state['tz']})."
    return f"The user's approximate location is {place}."


def apply_ambient(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append (or merge into) the system message a line built from
    :func:`ambient_line`. Mirrors ``voice_rules.apply_system_rules``: never
    mutates the input list/dicts, returns ``messages`` unchanged (same
    object) when there's nothing to add.

    Assumes ``messages`` is a FRESH per-request serialization (as
    ``_serialize`` guarantees) — it detects its own current line but does
    not hunt for stale ambient lines from earlier calls, so feeding it an
    already-ambient-stamped list after the location changed would keep
    both lines. Do not persist its output back into chat history."""
    line = ambient_line()
    if not line:
        return messages

    system_index = next((i for i, m in enumerate(messages) if m.get("role") == "system"), None)

    if system_index is None:
        return [{"role": "system", "content": line}, *messages]

    system_message = messages[system_index]
    content = system_message.get("content")

    if isinstance(content, str):
        if line in content:
            return messages
        new_content: Any = content + "\n\n" + line
    elif isinstance(content, list):
        if any(isinstance(part, dict) and line in (part.get("text") or "") for part in content):
            return messages
        new_content = [*content, {"type": "text", "text": "\n\n" + line}]
    else:
        new_content = line

    new_messages = list(messages)
    new_messages[system_index] = {**system_message, "content": new_content}
    return new_messages
