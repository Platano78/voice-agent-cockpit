"""
Server-side tools the voice agent's LLM can call: weather, web search, and
knowledge-base lookup (QMD). Definitions are wired onto ``session.tools`` by
``BrainControl``; execution happens here, off the audio thread, with a hard
per-tool timeout and a short, TTS-friendly plain-text result.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

_MAX_RESULT_CHARS = 600

_WEATHER_TIMEOUT_S = 6.0
_KNOWLEDGE_TIMEOUT_S = 6.0
_SEARCH_TIMEOUT_S = 10.0
_MOOD_TIMEOUT_S = 2.0
_HERMES_DELEGATE_TIMEOUT_S = 3.0
_HERMES_STATUS_TIMEOUT_S = 2.0
_HERMES_RESPOND_TIMEOUT_S = 6.0
_SEND_TIMEOUT_S = 6.0

_MOODS = ("neutral", "happy", "excited", "thinking", "concerned", "playful", "serious")

_QMD_URL = os.environ.get("QMD_MCP_URL", "http://localhost:8070/mcp")
_HERMES_MCP_URL = os.environ.get("HERMES_MCP_URL", "http://localhost:8088/mcp")

# Optional directory of drop-in local tools (one .py per tool). Empty = off,
# which is the public default (no behavior change). See _load_dropin_tools().
_TOOLS_DIR = os.environ.get("VOICE_TOOLS_DIR", "")

# Set once by build_pipeline via set_cockpit(); None until the websocket/
# chat-completions pipeline wires it up.
_cockpit: Optional[Any] = None

# Names armed by the last get_tool_defs() call. execute() refuses names
# outside this set so an LLM hallucinating a known-but-unarmed tool cannot
# run it (arming gates execution, not just advertisement). Empty = arming
# has not run (standalone/unit use) = permissive.
_ARMED_NAMES: set[str] = set()


def set_cockpit(cockpit: Any) -> None:
    global _cockpit
    _cockpit = cockpit

# One shared worker pool for the hard wall-clock deadlines below; tool calls
# are infrequent and short-lived so a small pool is plenty.
_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="voice-tool")

_WMO_WEATHER: dict[int, str] = {
    0: "clear sky",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "foggy with icy fog",
    51: "light drizzle",
    53: "drizzle",
    55: "dense drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    80: "light rain showers",
    81: "rain showers",
    82: "violent rain showers",
    95: "thunderstorms",
    96: "thunderstorms with hail",
    99: "thunderstorms with heavy hail",
}

TOOL_DEFS: list[dict] = [
    {
        "type": "function",
        "name": "get_weather",
        "description": (
            "Get the current weather for a place. Use when the user asks about "
            "weather, temperature, or conditions somewhere. Answer in 1-2 spoken "
            "sentences."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "place": {
                    "type": "string",
                    "description": "City or place name, e.g. 'Tokyo' or 'Paris, France'.",
                }
            },
            "required": ["place"],
        },
    },
    {
        "type": "function",
        "name": "web_search",
        "description": (
            "Search the public web for current events, software versions and "
            "releases, sports scores, news, prices, or anything that may have "
            "changed recently — prefer this over answering from memory for "
            "time-sensitive facts. Answer in 1-2 spoken sentences summarizing "
            "what you find. The result links appear on the user's screen to "
            "click — when the user asks where to get, download, or buy "
            "something, mention the links are on screen instead of reading "
            "URLs aloud."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."}
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "knowledge_lookup",
        "description": (
            "Search the user's personal notes, projects, infrastructure, and "
            "research. Use for ANY question about the user's own work or setup, "
            "even if the phrasing is approximate or partially misheard. Answer "
            "in 1-2 spoken sentences."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look up."}
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "set_mood",
        "description": (
            "Set the visual mood of your interface. Call this whenever your "
            "emotional tone shifts — when sharing good news, thinking hard, "
            "expressing concern, being playful. The user sees your mood as "
            "the interface's appearance. Mood persists until you change it. "
            "Never announce or narrate the mood change in your reply."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mood": {
                    "type": "string",
                    "enum": list(_MOODS),
                    "description": "The mood to display.",
                }
            },
            "required": ["mood"],
        },
    },
    {
        "type": "function",
        "name": "delegate_to_hermes",
        "description": (
            "Delegate a long-running or multi-step task to Hermes, the "
            "household's background agent. Use for tasks that need research, "
            "multiple steps, or background work beyond a quick spoken answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The task to hand off to Hermes."}
            },
            "required": ["task"],
        },
    },
    {
        "type": "function",
        "name": "hermes_status",
        "description": (
            "Check on the status of a task delegated to Hermes, or whether any "
            "approvals are waiting. Use when the user asks how Hermes is doing, "
            "what happened with a delegated task, or if anything needs approval."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "detail": {
                    "type": "string",
                    "enum": ["summary", "last_result", "steps", "approvals"],
                    "description": (
                        "What to report. Default summary. Use last_result when the "
                        "user asks what Hermes found/answered; steps for what it's "
                        "been doing; approvals for pending approvals."
                    ),
                }
            },
        },
    },
    {
        "type": "function",
        "name": "respond_permission",
        "description": (
            "Approve or deny the oldest pending approval request from Hermes. "
            "Use when the user says to approve, allow, deny, or reject a Hermes "
            "request."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["approve", "deny"],
                    "description": "Whether to approve or deny the request.",
                }
            },
            "required": ["decision"],
        },
    },
    {
        "type": "function",
        "name": "send_to_hermes",
        "description": (
            "Send a quick spoken message or follow-up to Hermes — use when the "
            "user says 'tell Hermes', 'ask Hermes', 'let Hermes know', or answers "
            "a question Hermes asked. For a NEW multi-step task use "
            "delegate_to_hermes instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "The message to send to Hermes."}
            },
            "required": ["message"],
        },
    },
]

# Availability tiers used by get_tool_defs() to arm only the tools whose
# backing service is actually reachable at pipeline start.
CORE_TOOLS = ("get_weather", "web_search", "set_mood")
QMD_TOOLS = ("knowledge_lookup",)
HERMES_TOOLS = ("delegate_to_hermes", "hermes_status", "respond_permission", "send_to_hermes")


def _probe(url: str, payload: dict) -> bool:
    """Return True iff ``url`` answers ``payload`` with a 2xx and (for JSON
    bodies) no top-level "error". False on ANY exception."""
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    try:
        with httpx.Client(timeout=1.5) as client:
            resp = client.post(url, headers=headers, json=payload)
        if not (200 <= resp.status_code < 300):
            return False
        try:
            body = resp.json()
        except ValueError:
            return True
        return not (isinstance(body, dict) and "error" in body)
    except Exception:
        return False


def _load_dropin_tools() -> list[dict]:
    """Load drop-in tools from ``VOICE_TOOLS_DIR`` (one ``*.py`` per tool),
    registering each into ``_DISPATCH`` and returning its TOOL_DEF.

    Off unless the env var points at a directory. Each module must expose a
    ``TOOL_DEF`` dict (with a string "name" and "parameters") and a callable
    ``run``; optional ``TIMEOUT_S`` / ``ARG_KEY`` / ``ARG_LABEL`` /
    ``TOOL_LABEL`` / ``REQUIRED`` override the defaults. Any load/validation
    failure warns and skips — a broken drop-in never crashes the pipeline.
    Built-ins and earlier-loaded drop-ins win name collisions. Idempotent:
    re-registration overwrites the ``_DISPATCH`` entry rather than duplicating."""
    if not _TOOLS_DIR or not os.path.isdir(_TOOLS_DIR):
        return []
    builtin_names = {t["name"] for t in TOOL_DEFS}
    defs: list[dict] = []
    seen: set[str] = set()
    for path in sorted(Path(_TOOLS_DIR).glob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(f"voice_dropin_{path.stem}", str(path))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            logger.warning("voice_tools: drop-in %s failed to load: %r", path, e)
            continue
        tool_def = getattr(module, "TOOL_DEF", None)
        run = getattr(module, "run", None)
        if not (
            isinstance(tool_def, dict)
            and isinstance(tool_def.get("name"), str)
            and "parameters" in tool_def
            and callable(run)
        ):
            logger.warning("voice_tools: drop-in %s missing TOOL_DEF/run; skipped", path)
            continue
        name = tool_def["name"]
        if name in builtin_names or name in seen:
            logger.warning("voice_tools: drop-in %s name %r collides with an existing tool; skipped", path, name)
            continue
        required_list = (tool_def.get("parameters") or {}).get("required") or []
        arg_key = getattr(module, "ARG_KEY", required_list[0] if required_list else None)
        timeout_s = getattr(module, "TIMEOUT_S", 8.0)
        arg_label = getattr(module, "ARG_LABEL", f"a value for {name}")
        tool_label = getattr(module, "TOOL_LABEL", name.replace("_", " "))
        required = getattr(module, "REQUIRED", arg_key is not None)
        tool_def.setdefault("type", "function")
        _DISPATCH[name] = (run, timeout_s, arg_key, arg_label, tool_label, required)
        seen.add(name)
        defs.append(tool_def)
    return defs


def get_tool_defs() -> list[dict]:
    """Arm the subset of the tool catalog whose backing service is reachable.

    If the ``VOICE_TOOLS`` env var is set (non-empty), it pins the set to the
    named tools with no probing. Otherwise CORE_TOOLS are always armed, QMD and
    Hermes tools are added only if their MCP endpoint answers a probe. Drop-in
    tools from ``VOICE_TOOLS_DIR`` load once per call and are appended AFTER the
    built-in armed list; they arm unconditionally except under ``VOICE_TOOLS``,
    which filters the combined set by name. Probes run once per call —
    BrainControl calls this once at pipeline start, so a restart is required to
    rearm. Returns defs in catalog order."""
    dropins = _load_dropin_tools()
    catalog = TOOL_DEFS + dropins
    by_name = {t["name"]: t for t in catalog}
    dropin_suffix = f" +{len(dropins)} drop-in" if dropins else ""

    env = os.environ.get("VOICE_TOOLS")
    if env:
        wanted = set()
        for raw in env.split(","):
            name = raw.strip()
            if not name:
                continue
            if name in by_name:
                wanted.add(name)
            else:
                logger.warning("voice_tools: VOICE_TOOLS names unknown tool %r (ignored)", name)
        armed = [t for t in catalog if t["name"] in wanted]
        _ARMED_NAMES.clear()
        _ARMED_NAMES.update(t["name"] for t in armed)
        logger.info("voice_tools: armed %d/%d (env)%s", len(armed), len(catalog), dropin_suffix)
        return armed

    names = set(CORE_TOOLS)
    qmd_ok = _probe(
        _QMD_URL,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "voice-agent", "version": "1"},
            },
        },
    )
    if qmd_ok:
        names.update(QMD_TOOLS)
    hermes_ok = _probe(
        _HERMES_MCP_URL,
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
    )
    if hermes_ok:
        names.update(HERMES_TOOLS)

    armed = [t for t in TOOL_DEFS if t["name"] in names] + dropins
    _ARMED_NAMES.clear()
    _ARMED_NAMES.update(t["name"] for t in armed)
    logger.info(
        "voice_tools: armed %d/%d (probe: qmd=%s hermes=%s)%s",
        len(armed),
        len(catalog),
        "up" if qmd_ok else "down",
        "up" if hermes_ok else "down",
        dropin_suffix,
    )
    return armed


# Tools that only change the client's visual presentation — no network call,
# no spoken result to report, so the "Let me check." filler in
# LMOutputProcessor would be misleading and is skipped for these.
VISUAL_TOOLS = {"set_mood"}


def _truncate(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= _MAX_RESULT_CHARS:
        return text
    return text[: _MAX_RESULT_CHARS - 1].rstrip() + "…"


def _run_get_weather(place: str) -> str:
    with httpx.Client(timeout=_WEATHER_TIMEOUT_S) as client:
        geo = client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": place, "count": 1},
        )
        geo.raise_for_status()
        results = geo.json().get("results") or []
        if not results:
            return f"I couldn't find a place called {place!r}."
        hit = results[0]
        lat, lon = hit["latitude"], hit["longitude"]
        label = hit.get("name", place)
        country = hit.get("country")
        if country:
            label = f"{label}, {country}"

        fc = client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": "true"},
        )
        fc.raise_for_status()
        current = fc.json().get("current_weather") or {}
        temp = current.get("temperature")
        wind = current.get("windspeed")
        code = current.get("weathercode")
        conditions = _WMO_WEATHER.get(code, "unknown conditions")
        if temp is None:
            return f"I couldn't get current conditions for {label}."
        sentence = f"It's currently {temp}°C and {conditions} in {label}"
        if wind is not None:
            sentence += f", with wind around {wind} km/h."
        else:
            sentence += "."
        return sentence


def _run_web_search(query: str) -> str:
    from ddgs import DDGS

    with DDGS(timeout=int(_SEARCH_TIMEOUT_S)) as ddgs:
        hits = list(ddgs.text(query, max_results=3))
    if not hits:
        return f"I didn't find any web results for {query!r}."
    lines = []
    links = []
    for hit in hits[:3]:
        title = (hit.get("title") or "").strip()
        body = (hit.get("body") or "").strip()
        # DDG instant-answer boxes sometimes return one run-on sentence with no
        # ". " breaks; a hard character cap keeps each line short regardless.
        snippet = body.split(". ")[0][:120].strip() if body else ""
        if title and snippet:
            lines.append(f"{title} — {snippet}")
        elif title:
            lines.append(title)
        # ddgs text() hits carry the URL as `href`; fall back to `url`.
        href = (hit.get("href") or hit.get("url") or "").strip()
        if href.startswith(("http://", "https://")):
            links.append({"title": title, "url": href, "host": urlparse(href).netloc})
    # Surface clickable links on the cockpit screen; a UI push failure must
    # never break the spoken result.
    if _cockpit is not None and links:
        try:
            _cockpit.push_links(query, links)
        except Exception as e:
            logger.warning("voice_tools: push_links failed: %r", e)
    return _truncate(" | ".join(lines))


def _run_knowledge_lookup(query: str) -> str:
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    with httpx.Client(timeout=_KNOWLEDGE_TIMEOUT_S) as client:
        init_resp = client.post(
            _QMD_URL,
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "voice-agent", "version": "1"},
                },
            },
        )
        init_resp.raise_for_status()
        session_id = init_resp.headers.get("mcp-session-id")
        if not session_id:
            return "The knowledge base is unavailable right now."

        call_headers = dict(headers)
        call_headers["mcp-session-id"] = session_id
        call_resp = client.post(
            _QMD_URL,
            headers=call_headers,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "query",
                    "arguments": {
                        # lex alone misses approximate/misheard queries (STT can turn
                        # "pi harness" into "pie harness"); vec catches those by
                        # meaning. QMD merges/ranks/dedupes both into one result list
                        # server-side, so no client-side merge is needed here.
                        "searches": [
                            {"type": "lex", "query": query},
                            {"type": "vec", "query": query},
                        ],
                        "intent": "voice agent knowledge lookup",
                        "rerank": False,
                    },
                },
            },
        )
        call_resp.raise_for_status()
        payload = call_resp.json()

    # structuredContent.results[].snippet is diff-formatted (line numbers, hunk
    # markers) and unusable for speech; title + file read cleanly instead.
    results = ((payload.get("result") or {}).get("structuredContent") or {}).get("results") or []
    if not results:
        return f"I didn't find anything in your notes about {query!r}."
    lines = [f"{r.get('title', 'untitled')} in {r.get('file', 'notes')}" for r in results[:3]]
    return _truncate("; ".join(lines))


def _run_set_mood(mood: str) -> str:
    if mood not in _MOODS:
        return f"I don't have a {mood!r} mood — staying as is."
    return f"Mood set to {mood}. Do not mention or announce the mood change — just continue the conversation naturally."


def _run_delegate_to_hermes(task: str) -> str:
    if _cockpit is None:
        return "Hermes isn't connected right now."
    return _cockpit.delegate(task)


def _run_hermes_status(detail: Any) -> str:
    if _cockpit is None:
        return "Hermes isn't connected right now."
    return _cockpit.status_summary(detail or "summary")


def _run_respond_permission(decision: str) -> str:
    if _cockpit is None:
        return "Hermes isn't connected right now."
    return _cockpit.respond_permission(decision)


def _run_send_to_hermes(message: str) -> str:
    if _cockpit is None:
        return "Hermes isn't connected right now."
    return _cockpit.send_message(message)


# (fn, timeout_s, arg kwarg name (None for zero-arg tools), spoken label for
# that arg, spoken label for the tool itself, required: whether a missing arg
# should be refused ("I need ...") rather than passed through as None)
_DISPATCH = {
    "get_weather": (_run_get_weather, _WEATHER_TIMEOUT_S, "place", "a place name for the weather", "weather lookup", True),
    "web_search": (_run_web_search, _SEARCH_TIMEOUT_S, "query", "a search query", "web search", True),
    "knowledge_lookup": (
        _run_knowledge_lookup,
        _KNOWLEDGE_TIMEOUT_S,
        "query",
        "something to look up",
        "knowledge base lookup",
        True,
    ),
    "set_mood": (_run_set_mood, _MOOD_TIMEOUT_S, "mood", "a mood", "mood change", True),
    "delegate_to_hermes": (
        _run_delegate_to_hermes,
        _HERMES_DELEGATE_TIMEOUT_S,
        "task",
        "a task to hand off",
        "Hermes handoff",
        True,
    ),
    "hermes_status": (_run_hermes_status, _HERMES_STATUS_TIMEOUT_S, "detail", "a detail to report", "Hermes status check", False),
    "respond_permission": (
        _run_respond_permission,
        _HERMES_RESPOND_TIMEOUT_S,
        "decision",
        "approve or deny",
        "Hermes approval response",
        True,
    ),
    "send_to_hermes": (
        _run_send_to_hermes,
        _SEND_TIMEOUT_S,
        "message",
        "a message for Hermes",
        "Hermes message",
        True,
    ),
}


def execute(name: str, kwargs: dict) -> str:
    """Run tool ``name`` with ``kwargs`` under a hard timeout.

    Never raises. Always returns a short plain-text string suitable for
    speaking back to the user.
    """
    if _ARMED_NAMES and name not in _ARMED_NAMES:
        logger.warning("voice_tools.execute: tool %r not armed this session", name)
        return "That tool isn't available."
    entry = _DISPATCH.get(name)
    if entry is None:
        logger.warning("voice_tools.execute: unknown tool %r", name)
        return "That tool isn't available."
    fn, timeout_s, arg_key, arg_label, tool_label, required = entry
    if arg_key is None:
        arg = None
    else:
        arg = (kwargs or {}).get(arg_key)
        if not arg:
            if required:
                logger.warning("voice_tools.execute: %s missing required arg %r (got %r)", name, arg_key, kwargs)
                return f"I need {arg_label}."
            arg = None
    try:
        future = _EXECUTOR.submit(fn, arg)
        try:
            result = future.result(timeout=timeout_s)
        except FutureTimeoutError:
            logger.warning("voice_tools.execute: %s timed out after %.1fs", name, timeout_s)
            future.cancel()
            return f"The {tool_label} timed out."
        return _truncate(result) if result else f"The {tool_label} failed."
    except Exception as e:
        logger.warning("voice_tools.execute: %s failed: %r", name, e)
        return f"The {tool_label} failed."
