#!/usr/bin/env bash
# Independent verifier via OpenAI Codex CLI (default gpt-5.6-sol, high reasoning).
# Usage: tools/codex_verify.sh <name> <prompt_file>
# Writes results/verify/<name>.json (structured verdict per tools/verify_schema.json),
#        results/verify/<name>.log (event stream), results/verify/<name>.exit
# Notes: stdin MUST be closed (codex otherwise waits on it). Codex's own bubblewrap
# sandbox needs to run outside any outer sandbox (Claude Code: dangerouslyDisableSandbox).
# Retries once if no verdict file was produced (transient bwrap loopback failures).
set -u
NAME="$1"; PROMPT_FILE="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/results/verify"; mkdir -p "$OUT"
MODEL="${CODEX_MODEL:-gpt-5.6-sol}"
EFFORT="${CODEX_EFFORT:-high}"
rm -f "$OUT/$NAME.json" "$OUT/$NAME.exit"
for attempt in 1 2; do
  codex exec --json --skip-git-repo-check \
    --sandbox danger-full-access -c approval_policy=never \
    -m "$MODEL" -c model_reasoning_effort="$EFFORT" \
    -c "projects.\"$ROOT\".trust_level=\"trusted\"" \
    -C "$ROOT" \
    --output-schema "$ROOT/tools/verify_schema.json" \
    -o "$OUT/$NAME.json" \
    "$(cat "$PROMPT_FILE")" < /dev/null > "$OUT/$NAME.attempt$attempt.log" 2>&1
  rc=$?
  cp "$OUT/$NAME.attempt$attempt.log" "$OUT/$NAME.log"
  if [ -s "$OUT/$NAME.json" ]; then break; fi
  echo "attempt $attempt produced no verdict (rc=$rc), retrying" >&2
done
echo $rc > "$OUT/$NAME.exit"
[ -s "$OUT/$NAME.json" ] && exit 0 || exit 1
