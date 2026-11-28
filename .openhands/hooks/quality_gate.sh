#!/usr/bin/env bash
# Stop hook: refuse to finish while the repo's quality gate is red.
# Runs only while work is in progress. Gating a clean, fully pushed tree would
# re-run the whole suite on every later read-only chat, forever.
set -uo pipefail
cd "${OPENHANDS_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}" || exit 0

dirty=$(git status --porcelain --untracked-files=no 2>/dev/null)
if [ -z "$dirty" ]; then
  # Nothing uncommitted: gate only while commits exist that no remote has.
  unpushed=$(git rev-list --count '@{upstream}..HEAD' 2>/dev/null) \
    || unpushed=$(git rev-list --count origin/main..HEAD 2>/dev/null || echo 0)
  [ "$unpushed" = 0 ] && exit 0
fi

if out=$(bash .openhands/quality.sh 2>&1); then
  exit 0
fi
tail=$(printf '%s\n' "$out" | tail -25 | tr '\n' ' ' | tr '"' "'")
jq -cn --arg r "Quality gate failed. Nothing may be declared finished with red checks. Run bash .openhands/quality.sh yourself and fix it. Output: $tail" '{decision:"deny", reason:$r}'
exit 2
