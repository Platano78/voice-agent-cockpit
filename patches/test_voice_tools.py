"""Unit tests for voice_tools.py's knowledge lane -- QMD content retrieval,
collection filtering, the QMD+Genesis fan-out, and the Faulkner decision tool.

Run from repo root: python3 -m pytest patches/test_voice_tools.py -v

voice_tools.py's only speech_to_speech import is `from speech_to_speech import
phone_context`, so the REAL patches/phone_context module is aliased in as that
name before importing it -- same pattern as test_phone_context.py's
_install_voice_tools_stub(). No live service is contacted: every test replaces
voice_tools._mcp_call, the single choke point through which all three backends
(QMD, Agent Genesis, Faulkner-DB) are reached.

The regression these tests exist for: _run_knowledge_lookup used to return
"<title> in <file>" for the top three hits and NO document content whatsoever,
so the LLM answered the user's question from three filenames and confabulated
the rest.
"""

from __future__ import annotations

import sys
import time
import types

import pytest

from patches import phone_context


def _install_voice_tools_stub():
    pkg = sys.modules.get("speech_to_speech")
    if pkg is None:
        pkg = types.ModuleType("speech_to_speech")
        sys.modules["speech_to_speech"] = pkg
    sys.modules["speech_to_speech.phone_context"] = phone_context
    pkg.phone_context = phone_context


_install_voice_tools_stub()

from patches import voice_tools as vt  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────

def _qmd_query_result(hits):
    """Shape returned by QMD's `query` tool (structuredContent.results)."""
    return {"structuredContent": {"results": hits}}


def _qmd_get_result(text):
    """Shape returned by QMD's `get` tool -- the text lives under
    content[0].resource.text, NOT content[0].text. Verified against the live
    server; reading content[0].text yields an empty string."""
    return {"content": [{"type": "resource", "resource": {"text": text}}]}


def _hit(file, title, line=10):
    return {"file": file, "title": title, "line": line, "score": 1.0}


class _Router:
    """Stands in for _mcp_call, recording every call and replying per-tool."""

    def __init__(self, handlers):
        self.handlers = handlers
        self.calls = []

    def __call__(self, url, tool, arguments, timeout_s):
        self.calls.append({"url": url, "tool": tool, "args": arguments})
        handler = self.handlers.get(tool)
        if handler is None:
            return {}
        return handler(arguments) if callable(handler) else handler


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _FakeHTTP:
    """Minimal stand-in for httpx.Client covering the plain-REST backends.
    Genesis is reached with real httpx rather than _mcp_call, so it needs its
    own seam."""

    def __init__(self, post=None, get=None):
        self._post = post
        self._get = get
        self.posts = []
        self.gets = []

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json=None, **kwargs):
        self.posts.append({"url": url, "json": json})
        if callable(self._post):
            return self._post(url, json)
        if isinstance(self._post, Exception):
            raise self._post
        return _FakeResponse(self._post)

    def get(self, url, **kwargs):
        self.gets.append({"url": url})
        if isinstance(self._get, Exception):
            raise self._get
        return _FakeResponse(None, status=self._get if self._get else 200)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Default every test to QMD-only with no gateway configured."""
    monkeypatch.setattr(vt, "_GENESIS_ARMED", False)
    monkeypatch.setattr(vt, "_GENESIS_URL", "")
    monkeypatch.setattr(vt, "_ARMED_NAMES", set())
    yield


def _arm_genesis(monkeypatch, post):
    """Arm the Genesis lane and route its HTTP through a fake client. `post`
    is the JSON body it should answer with, a callable, or an Exception."""
    monkeypatch.setattr(vt, "_GENESIS_ARMED", True)
    monkeypatch.setattr(vt, "_GENESIS_URL", "http://localhost:8080")
    fake = _FakeHTTP(post=post)
    monkeypatch.setattr(vt.httpx, "Client", fake)
    return fake


def _genesis_payload(*docs):
    """Shape returned by Agent Genesis POST /search -- `document` is the raw
    conversation text, already readable, so there is no second content fetch."""
    return {"results_count": len(docs), "results": [{"id": "x", "document": d} for d in docs]}


# ── the regression: real content, not titles ─────────────────────────────

def test_knowledge_lookup_returns_document_content_not_titles():
    body = (
        "The echo gate scores incoming audio against the last TTS frame and "
        "drops it when the correlation clears the threshold."
    )
    vt._mcp_call = _Router({
        "query": _qmd_query_result([_hit("knowledge-base/notes.md", "Echo gate")]),
        "get": _qmd_get_result(body),
    })

    out = vt._run_knowledge_lookup("echo gate")

    assert "scores incoming audio" in out, "document content must be returned"
    assert "correlation clears the threshold" in out
    # The old implementation returned exactly this and nothing else.
    assert out != "Echo gate in knowledge-base/notes.md"


def test_knowledge_lookup_fetches_content_windowed_around_the_hit_line():
    router = _Router({
        "query": _qmd_query_result([_hit("kb/a.md", "A", line=100)]),
        "get": _qmd_get_result("windowed body text"),
    })
    vt._mcp_call = router

    vt._run_knowledge_lookup("anything")

    get_call = next(c for c in router.calls if c["tool"] == "get")
    assert get_call["args"]["file"] == "kb/a.md"
    assert get_call["args"]["fromLine"] == 100 - vt._GET_LINES_BEFORE
    assert get_call["args"]["maxLines"] == vt._GET_MAX_LINES
    # lineNumbers must be off or every line arrives prefixed "100: ".
    assert get_call["args"]["lineNumbers"] is False


def test_hit_line_near_start_of_file_clamps_to_line_one():
    router = _Router({
        "query": _qmd_query_result([_hit("kb/a.md", "A", line=2)]),
        "get": _qmd_get_result("body"),
    })
    vt._mcp_call = router

    vt._run_knowledge_lookup("q")

    assert next(c for c in router.calls if c["tool"] == "get")["args"]["fromLine"] == 1


def test_mcp_text_reads_resource_block_and_text_block():
    """Guards the exact shape bug that made the first implementation return
    empty excerpts: QMD's get() wraps content in a `resource` block."""
    assert vt._mcp_text(_qmd_get_result("hello")) == "hello"
    assert vt._mcp_text({"content": [{"type": "text", "text": "plain"}]}) == "plain"
    assert vt._mcp_text({}) == ""


# ── collection filtering ─────────────────────────────────────────────────

def test_query_is_filtered_to_the_users_own_collections():
    router = _Router({
        "query": _qmd_query_result([_hit("kb/a.md", "A")]),
        "get": _qmd_get_result("body"),
    })
    vt._mcp_call = router

    vt._run_knowledge_lookup("q")

    first_query = next(c for c in router.calls if c["tool"] == "query")
    assert first_query["args"]["collections"] == vt._KNOWLEDGE_COLLECTIONS
    # skills/agents/security are tooling docs, not the user's notes -- a live
    # query for "echo gate" ranked a pre-commit SKILL.md first at score 1.0.
    for noisy in ("skills", "agents", "security"):
        assert noisy not in vt._KNOWLEDGE_COLLECTIONS


def test_rerank_stays_off_and_intent_is_always_sent():
    """Reranking costs ~85s -- categorically unusable in a voice loop."""
    router = _Router({
        "query": _qmd_query_result([_hit("kb/a.md", "A")]),
        "get": _qmd_get_result("body"),
    })
    vt._mcp_call = router

    vt._run_knowledge_lookup("q")

    args = next(c for c in router.calls if c["tool"] == "query")["args"]
    assert args["rerank"] is False
    assert args["intent"]
    # lex catches exact terms, vec catches misheard/approximate phrasing.
    assert [s["type"] for s in args["searches"]] == ["lex", "vec"]


def test_empty_filtered_result_retries_across_all_collections():
    calls = {"n": 0}

    def query(args):
        calls["n"] += 1
        # First (filtered) call finds nothing; the unfiltered retry finds one.
        if "collections" in args:
            return _qmd_query_result([])
        return _qmd_query_result([_hit("skills/x/SKILL.md", "X")])

    vt._mcp_call = _Router({"query": query, "get": _qmd_get_result("wide body")})

    out = vt._run_knowledge_lookup("q")

    assert calls["n"] == 2, "must retry unfiltered rather than giving up"
    assert "wide body" in out


def test_no_results_anywhere_reports_nothing_found():
    vt._mcp_call = _Router({"query": _qmd_query_result([])})
    assert "didn't find anything" in vt._run_knowledge_lookup("obscure thing")


# ── merge + source labelling ─────────────────────────────────────────────

def test_merged_output_labels_each_source_distinctly(monkeypatch):
    _arm_genesis(monkeypatch, _genesis_payload("we talked it through"))
    vt._mcp_call = _Router({
        "query": _qmd_query_result([_hit("kb/a.md", "Echo gate")]),
        "get": _qmd_get_result("note body here"),
    })

    out = vt._run_knowledge_lookup("echo gate")

    assert "From your notes (Echo gate): note body here" in out
    assert "From a past conversation: we talked it through" in out


def test_genesis_posts_query_and_limit_to_the_search_endpoint(monkeypatch):
    fake = _arm_genesis(monkeypatch, _genesis_payload("body"))
    vt._mcp_call = _Router({"query": _qmd_query_result([])})

    vt._run_knowledge_lookup("wakeword gate")

    assert fake.posts[0]["url"] == "http://localhost:8080/search"
    assert fake.posts[0]["json"] == {"query": "wakeword gate", "limit": vt._KNOWLEDGE_CONVOS}


# ── degradation ──────────────────────────────────────────────────────────

def test_genesis_failure_degrades_to_qmd_only(monkeypatch):
    _arm_genesis(monkeypatch, RuntimeError("genesis container is down"))
    vt._mcp_call = _Router({
        "query": _qmd_query_result([_hit("kb/a.md", "A")]),
        "get": _qmd_get_result("qmd body survives"),
    })

    out = vt._run_knowledge_lookup("q")

    assert "qmd body survives" in out, "a dead Genesis must not take the tool down"
    assert "From a past conversation" not in out


def test_genesis_malformed_payload_degrades_quietly(monkeypatch):
    _arm_genesis(monkeypatch, lambda url, body: _FakeResponse(ValueError("not json")))
    vt._mcp_call = _Router({
        "query": _qmd_query_result([_hit("kb/a.md", "A")]),
        "get": _qmd_get_result("qmd body"),
    })

    assert "qmd body" in vt._run_knowledge_lookup("q")


def test_genesis_http_error_degrades_quietly(monkeypatch):
    _arm_genesis(monkeypatch, lambda url, body: _FakeResponse({}, status=500))
    vt._mcp_call = _Router({
        "query": _qmd_query_result([_hit("kb/a.md", "A")]),
        "get": _qmd_get_result("qmd body"),
    })

    assert "qmd body" in vt._run_knowledge_lookup("q")


def test_genesis_not_consulted_when_unarmed(monkeypatch):
    fake = _FakeHTTP(post=_genesis_payload("leak"))
    monkeypatch.setattr(vt.httpx, "Client", fake)
    vt._mcp_call = _Router({
        "query": _qmd_query_result([_hit("kb/a.md", "A")]),
        "get": _qmd_get_result("body"),
    })

    out = vt._run_knowledge_lookup("q")

    assert fake.posts == [], "unarmed Genesis must not be contacted at all"
    assert "leak" not in out


def test_qmd_failure_still_returns_genesis_results(monkeypatch):
    _arm_genesis(monkeypatch, _genesis_payload("conversation body"))

    def explode(args):
        raise RuntimeError("qmd down")

    vt._mcp_call = _Router({"query": explode})

    assert "conversation body" in vt._run_knowledge_lookup("q")


def test_a_hung_lane_does_not_cost_the_other_lane_its_results(monkeypatch):
    """The fan-out shares one wall clock, so a wedged Genesis must not eat the
    whole tool deadline -- QMD's results still come back."""
    def hang(url, body):
        time.sleep(5)
        return _FakeResponse(_genesis_payload("too late"))

    _arm_genesis(monkeypatch, hang)
    monkeypatch.setattr(vt, "_FANOUT_BUDGET_S", 0.3)

    vt._mcp_call = _Router({
        "query": _qmd_query_result([_hit("kb/a.md", "A")]),
        "get": _qmd_get_result("fast qmd body"),
    })

    started = time.monotonic()
    out = vt._run_knowledge_lookup("q")
    elapsed = time.monotonic() - started

    assert "fast qmd body" in out
    assert elapsed < 2.0, f"fan-out did not honour its budget ({elapsed:.1f}s)"


# ── arming ───────────────────────────────────────────────────────────────

def _stub_arming(monkeypatch, genesis_up):
    monkeypatch.delenv("VOICE_TOOLS", raising=False)
    monkeypatch.setattr(vt, "_GENESIS_URL", "http://localhost:8080")
    # QMD/Hermes probes are irrelevant here; fail them so only CORE arms.
    monkeypatch.setattr(vt, "_probe", lambda url, payload: False)
    monkeypatch.setattr(vt, "_probe_health", lambda url: genesis_up)


def test_genesis_lane_arms_when_health_probe_succeeds(monkeypatch):
    _stub_arming(monkeypatch, True)
    vt.get_tool_defs()
    assert vt._GENESIS_ARMED is True


def test_genesis_lane_disarms_when_container_is_down(monkeypatch):
    """Dropped from the fan-out entirely, rather than paying its timeout on
    every single call."""
    _stub_arming(monkeypatch, False)
    vt.get_tool_defs()
    assert vt._GENESIS_ARMED is False


def test_probe_health_hits_the_health_endpoint(monkeypatch):
    fake = _FakeHTTP(get=200)
    monkeypatch.setattr(vt.httpx, "Client", fake)

    assert vt._probe_health("http://localhost:8080") is True
    assert fake.gets[0]["url"] == "http://localhost:8080/health"


def test_probe_health_is_false_on_error(monkeypatch):
    monkeypatch.setattr(vt.httpx, "Client", _FakeHTTP(get=ConnectionError("down")))
    assert vt._probe_health("http://localhost:8080") is False
    assert vt._probe_health("") is False


def test_no_decision_lookup_tool_is_shipped():
    """Faulkner's only local surface (/api/search on :8086) ANDs every word of
    the query with no ranking, so natural-language decision questions match
    nothing -- 'why did we choose two processes' returns 0 nodes live. Shipping
    it would answer 'I didn't find a recorded decision' to almost everything."""
    assert "decision_lookup" not in vt._DISPATCH
    assert "decision_lookup" not in [t["name"] for t in vt.TOOL_DEFS]


# ── result budget + speakability ─────────────────────────────────────────

def test_knowledge_results_get_a_larger_cap_than_one_line_tools():
    assert vt._RESULT_CHARS["knowledge_lookup"] > vt._MAX_RESULT_CHARS
    # web_search and friends keep the tight default.
    assert "web_search" not in vt._RESULT_CHARS


def test_execute_does_not_truncate_knowledge_results_to_the_short_cap():
    # Each excerpt is capped at _DOC_EXCERPT_CHARS, so it takes both fetched
    # documents to exceed the 600-char default that every other tool uses.
    vt._mcp_call = _Router({
        "query": _qmd_query_result([_hit("kb/a.md", "A"), _hit("kb/b.md", "B")]),
        "get": _qmd_get_result("sentence. " * 200),
    })

    out = vt.execute("knowledge_lookup", {"query": "q"})

    assert len(out) > vt._MAX_RESULT_CHARS
    assert len(out) <= vt._RESULT_CHARS["knowledge_lookup"]


def test_speakable_strips_frontmatter_context_comment_and_markdown():
    raw = (
        "<!-- Context: Curated research notes -->\n"
        "\n"
        "---\n"
        "name: some-doc\n"
        "type: reference\n"
        "---\n"
        "\n"
        "# The Heading\n"
        "\n"
        "The **real** body with `code` in it.\n"
    )

    out = vt._speakable(raw, 500)

    assert out == "The Heading The real body with code in it."
    for artifact in ("<!--", "---", "name:", "**", "`", "#"):
        assert artifact not in out


def test_speakable_caps_length_with_ellipsis():
    out = vt._speakable("word " * 100, 20)
    assert out.endswith("…")
    assert len(out) == 20


def test_lane_never_speaks_qmds_diff_formatted_snippet():
    """QMD's snippet field carries line numbers and @@ hunk markers -- the
    reason the original implementation avoided content at all. get() is clean."""
    router = _Router({
        "query": _qmd_query_result([
            dict(_hit("kb/a.md", "A"), snippet="3: @@ -2,4 @@ (1 before)\n4: junk")
        ]),
        "get": _qmd_get_result("clean body"),
    })
    vt._mcp_call = router

    out = vt._run_knowledge_lookup("q")

    assert "@@" not in out
    assert "clean body" in out


# ── MCP transport ────────────────────────────────────────────────────────

def test_mcp_json_parses_sse_frames():
    """The gateway answers text/event-stream; QMD answers plain JSON."""

    class _Resp:
        headers = {"content-type": "text/event-stream"}
        text = 'event: message\ndata: {"result": {"ok": true}}\n\n'

    assert vt._mcp_json(_Resp()) == {"result": {"ok": True}}


def test_mcp_json_returns_empty_on_garbage():
    class _Resp:
        headers = {"content-type": "application/json"}
        text = "<html>502</html>"

        def json(self):
            raise ValueError("not json")

    assert vt._mcp_json(_Resp()) == {}
