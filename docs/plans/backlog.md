# Voice Agent Cockpit — backlog

Durable "don't forget" list for deferred work. Each item is decision-complete
enough to pick up cold in a future session.

## Rig-side generation cap: `--n-predict` on the brain (deferred 2026-07-17)

**What:** Set an explicit `--n-predict` ceiling on the `llm-gemma12b.service`
llama-server launch flags (currently unset → defaults to `-1`, unbounded up to
`--ctx-size 131072`).

**Why:** The v1.4.0 stream watchdog + `VOICE_MAX_TOKENS` (commit `72a4750`) bound
runaway generations **client-side, in the cockpit only**. A server-side
`--n-predict` bounds worst-case generation length at the source, for **every**
consumer of the `:8084` brain (lurker, Hermes, MKG `coder` backend, ad-hoc
probes) — not just this voice pipeline. It's defense-in-depth for the same
root cause: an unbounded generation on the single-slot server wedges everyone
queued behind it.

**Why it is NOT in this repo:** this is an inference-serving change on the
`llama-cpp-native` rig, not a cockpit patch. It changes model-serving behavior
for all consumers, so per the user's own operating discipline it goes through
the **`local-llm-bench` A/B methodology** (measure output quality + latency
before/after), not a blind flag flip. A too-low `n_predict` would truncate
legitimately long completions for the *other* consumers that DO want length
(unlike voice, which never needs >1024 tokens).

**Where:** the systemd unit for `llm-gemma12b.service` on ai-utility
(192.168.1.79). Project: `/home/platano/project/llama-cpp-native`.

**Suggested value:** start the bench around `n_predict = 2048` (generous for
non-voice consumers, still a hard stop against the degenerate-trickle
pathology) and measure. The voice pipeline's own `VOICE_MAX_TOKENS=1024` stays
regardless — the two are independent belts.

**Pickup prompt (paste into a llama-cpp-native session):**
> Bench and set `--n-predict` on llm-gemma12b.service. Today it's unset
> (unbounded). Goal: a server-side generation ceiling so one runaway
> generation can't wedge the single-slot brain for all consumers. Use
> local-llm-bench discipline — A/B output quality + latency at a candidate
> value (~2048) vs current, on representative prompts from the real consumers
> (voice turn, a coder codegen, a lurker/Hermes agent turn). Context:
> voice-agent-cockpit already caps its own side (VOICE_MAX_TOKENS=1024, stream
> watchdog VOICE_STREAM_MAX_S=120s, commit 72a4750); this is the shared-server
> backstop. See voice-agent-cockpit memory `stream-trickle-defeats-read-timeout`.
