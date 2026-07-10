# Voice pipeline — benchmarks & model choices

Measured on ai-utility (RTX 3080 Ti 12GB) 2026-07-09/10. The pipeline runs **STT + TTS on
CPU** with the **Gemma-12B brain on GPU** (`:8084`) — not a preference, a constraint: the 12B
fills the GPU, leaving **~658 MB free**, so co-resident GPU STT/TTS is impossible. These numbers
are the incumbent baseline and the case for keeping it.

## Incumbent stack (SHIPPED) — Parakeet-TDT STT + pocket TTS + Gemma-12B
Warm, single persistent websocket (how the cockpit connects):
- **STT (Parakeet-TDT, CPU):** ~0.15 s for ~1.8 s audio. **WER ≈ 0%** (4/4 test questions verbatim).
- **First audio out:** ~0.65–0.87 s · **full short reply:** ~1.0–1.5 s · cold first turn ~1.9 s (one-time).
- **TTS A/B (CPU, `patches`/upstream `benchmark_tts.py`):**

  | Handler | Warmup | TTFC (first audio) | RTF | Verdict |
  |---|---|---|---|---|
  | **pocket** (shipped) | 1.96 s | **0.10–0.13 s** | 3.94 | ✅ streaming, sub-140ms first chunk |
  | kokoro | 10.9 s | 3.5 s | 1.68 | ❌ non-streaming → first audio = whole clip |
  | qwen3 | 25.9 s | — | fail | ❌ needs a voice-clone ref WAV it doesn't have |

  pocket's TTFC is ~26× better than kokoro's, and TTFC governs perceived conversational latency.

## NeMo challengers (weekend test) — verdict: NOT adoptable today
- **Nemotron-3.5-ASR-Streaming (0.6B):** blocked on *upstream packaging*, not our box. The ONNX-int4
  path runs but outputs all-blank (mel-preproc mismatch); the nemo-toolkit path needs
  `EncDecRNNTBPEModelWithPrompt`, a class that exists only on NeMo's **unreleased git-main** (latest
  PyPI `nemo-toolkit` 2.7.3 lacks it). No WER/latency obtainable. **Revisit when nemo-toolkit ≥2.8 ships it.**
- **MagpieTTS (357M):** ran — but **RTF 5.73 (≈5.7× slower than real-time) on CPU + 48 s warmup**.
  Rules itself out for a real-time voice agent on CPU-only.
- **Keep Parakeet-TDT + pocket.** Nothing here displaces it *on this box*. Full NVIDIA-speech scout:
  `knowledge-base/research-notes/nvidia-nemo-speech-audex-scout-2026-07-08.md`.

## What a second GPU unlocks ("go wild")
The single binding constraint above is the 658 MB of free VRAM. A second GPU (or a bigger card that
frees the 3080 Ti) flips several deferred items from "no" to "yes":
- **NeMo Nemotron-3.5-ASR-Streaming on GPU** — its whole point is controllable **80 ms–1 s** streaming
  latency; on GPU it's a real Parakeet upgrade (once the packaging ships).
- **MagpieTTS on GPU** — likely real-time there; re-run the A/B vs pocket for quality.
- **On-GPU vision** — the `look` tool's Gemma-12B call is the slow step; dedicated vision VRAM cuts it.
- **Co-resident STT + TTS + 12B** — no more CPU compromise; or headroom for a **bigger brain** (E4B→larger)
  or the **Audex-2B unified S2S** experiment (gated on its noncommercial license).
- Revisit this whole table when the hardware lands.
