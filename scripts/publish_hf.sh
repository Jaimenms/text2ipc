#!/usr/bin/env bash
# Export an index as a Hugging Face Inference Endpoints repository and upload it.
#
#   scripts/publish_hf.sh                      # PT, latest built version, default model
#   scripts/publish_hf.sh --lang EN --repo jaimenms/text2ipc-en
#   scripts/publish_hf.sh --version 20270101 --public
#
# Needs `hf auth login` (or HF_TOKEN) once. Re-running is idempotent: unchanged files
# are skipped by the Hub, changed ones become a new commit.
set -euo pipefail
cd "$(dirname "$0")/.."

LANG_CODE=PT; VERSION=latest; MODEL=""; REPO=""; VISIBILITY=--private; MESSAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lang) LANG_CODE=$2; shift 2 ;;
    --version) VERSION=$2; shift 2 ;;
    --model) MODEL=$2; shift 2 ;;
    --repo) REPO=$2; shift 2 ;;
    --public) VISIBILITY=""; shift ;;
    --message) MESSAGE=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
USER_NAME=$(uv run hf auth whoami 2>&1 | sed -n 's/^[Uu]ser=//p' | head -1)
[[ -n "$USER_NAME" ]] || { echo "not logged in: run 'uv run hf auth login'" >&2; exit 1; }
LANG_LOWER=$(echo "$LANG_CODE" | tr '[:upper:]' '[:lower:]')
REPO=${REPO:-$USER_NAME/text2ipc-$LANG_LOWER}
OUT=hf/text2ipc-$LANG_LOWER
MODEL_ARGS=""; [[ -n "$MODEL" ]] && MODEL_ARGS="--model $MODEL"

# shellcheck disable=SC2086
uv run t2ipc hf-export "$OUT" --version "$VERSION" --lang "$LANG_CODE" --repo-id "$REPO" $MODEL_ARGS
BUILT=$(python3 -c "import json;print(json.load(open('$OUT/text2ipc.json'))['version'])")
MESSAGE=${MESSAGE:-"text2ipc IPC $BUILT $LANG_CODE ($(git rev-parse --short HEAD 2>/dev/null || echo uncommitted))"}
uv run hf upload "$REPO" "$OUT" . --repo-type model $VISIBILITY --commit-message "$MESSAGE"
echo "published https://huggingface.co/$REPO"
