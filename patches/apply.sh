#!/usr/bin/env bash
# Re-apply the persona/brain-selector patch pack after a package reinstall.
# Copies the modified $PKG files + brain_control.py into whatever venv the
# speech_to_speech package currently lives in. brains.json and webclient/
# live outside the package and are untouched by reinstalls, so they are not
# copied here. Idempotent: safe to re-run over an already-patched tree.
#
# Since #8 (2026-07-03), the live install is an editable install from
# github.com/huggingface/speech-to-speech main
# ($INSTALL_DIR (default $HOME/speech-to-speech-main), venv at .venv there); $PKG resolves
# straight to the repo's src/speech_to_speech, so a pip reinstall of deps
# does not wipe these edits — only `git checkout` of the repo files would.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${INSTALL_DIR:-$HOME/speech-to-speech-main}"
PKG="$("$INSTALL_DIR/.venv/bin/python3" -c 'import speech_to_speech, os; print(os.path.dirname(speech_to_speech.__file__))')"

if [ -z "$PKG" ] || [ ! -d "$PKG" ]; then
  echo "Could not locate speech_to_speech package directory" >&2
  exit 1
fi

echo "Applying patches into: $PKG"
cp "$SCRIPT_DIR/brain_control.py" "$PKG/brain_control.py"
cp "$SCRIPT_DIR/voice_clone.py" "$PKG/voice_clone.py"
cp "$SCRIPT_DIR/websocket_streamer.py" "$PKG/connections/websocket_streamer.py"
cp "$SCRIPT_DIR/s2s_pipeline.py" "$PKG/s2s_pipeline.py"
cp "$SCRIPT_DIR/lm_output_processor.py" "$PKG/LLM/lm_output_processor.py"
cp "$SCRIPT_DIR/voice_tools.py" "$PKG/voice_tools.py"
cp "$SCRIPT_DIR/turn_stats.py" "$PKG/turn_stats.py"
cp "$SCRIPT_DIR/reflex_lane.py" "$PKG/reflex_lane.py"
cp "$SCRIPT_DIR/hermes_cockpit.py" "$PKG/hermes_cockpit.py"
cp "$SCRIPT_DIR/wakeword_gate.py" "$PKG/wakeword_gate.py"
cp "$SCRIPT_DIR/think_filter.py" "$PKG/think_filter.py"
cp "$SCRIPT_DIR/voice_rules.py" "$PKG/voice_rules.py"
cp "$SCRIPT_DIR/phone_context.py" "$PKG/phone_context.py"
cp "$SCRIPT_DIR/chat_completions_language_model.py" "$PKG/LLM/chat_completions_language_model.py"
echo "Patches applied."
