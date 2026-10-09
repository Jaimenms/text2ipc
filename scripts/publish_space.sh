#!/usr/bin/env bash
# Export indexes as a static Hugging Face Space (the browser demo) and upload it.
#
#   scripts/publish_space.sh --version 20260101            # PT + EN, e5-base, one IPC version, <user>/text2ipc
#   scripts/publish_space.sh                               # newest built index per language (may differ)
#   scripts/publish_space.sh --lang PT --repo jaimenms/text2ipc-demo --private
#   scripts/publish_space.sh --model st:intfloat/multilingual-e5-small  # 118 MB in the browser, weaker
#
# Needs `hf auth login` (or HF_TOKEN) once. Static Spaces are free; the repo is created
# on first run. Re-running is idempotent: unchanged files are skipped by the Hub.
set -euo pipefail
cd "$(dirname "$0")/.."

LANGS=(); VERSION=latest; MODEL=st:intfloat/multilingual-e5-base; REPO=""; VISIBILITY=--public; MESSAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lang) LANGS+=("$2"); shift 2 ;;
    --version) VERSION=$2; shift 2 ;;
    --model) MODEL=$2; shift 2 ;;
    --repo) REPO=$2; shift 2 ;;
    --private) VISIBILITY=--private; shift ;;
    --message) MESSAGE=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[[ ${#LANGS[@]} -gt 0 ]] || LANGS=(PT EN)
USER_NAME=$(uv run hf auth whoami 2>&1 | sed -n 's/^[Uu]ser=//p' | head -1)
[[ -n "$USER_NAME" ]] || { echo "not logged in: run 'uv run hf auth login'" >&2; exit 1; }
REPO=${REPO:-$USER_NAME/text2ipc}
OUT=space/$(basename "$REPO")
LANG_ARGS=(); for l in "${LANGS[@]}"; do LANG_ARGS+=(--lang "$l"); done

uv run t2ipc web-export "$OUT" --version "$VERSION" --model "$MODEL" --repo-id "$REPO" "${LANG_ARGS[@]}"
BUILT=$(python3 -c "import json;print(' '.join(f\"{e['lang']} {e['version']}\" for e in json.load(open('$OUT/manifest.json'))['indexes']))")
MESSAGE=${MESSAGE:-"text2ipc browser demo: IPC $BUILT ($(git rev-parse --short HEAD 2>/dev/null || echo uncommitted))"}
uv run hf repos create "$REPO" --repo-type space --space-sdk static $VISIBILITY --exist-ok >/dev/null
# --delete "*" keeps the Space an exact mirror of the export (drops template files, removed languages)
uv run hf upload "$REPO" "$OUT" . --repo-type space --delete "*" --commit-message "$MESSAGE"
echo "published https://huggingface.co/spaces/$REPO"
