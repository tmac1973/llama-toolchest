#!/usr/bin/env bash
# Pre-tool-use hook: an agent may push anywhere except main/master.
# Deliberately conservative - a push we cannot understand is denied.
# This guards against mistakes, not against an agent trying to trick it.
set -uo pipefail
set -f                       # the command string is data; never let it glob
input=$(cat)
cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // empty')
case "$cmd" in *push*) ;; *) exit 0 ;; esac

deny() {
  jq -cn --arg r "$1" '{decision:"deny", reason:$r}'
  exit 2
}

# Destination of a ref or refspec: HEAD:refs/heads/x -> x, refs/heads/x -> x.
protected() {
  local d=${1##*:}
  d=${d#refs/heads/}
  [ "$d" = main ] || [ "$d" = master ]
}

# Judge each word on its own. Paths and prose are skipped, so a checkout that
# happens to live under a directory named openhands/ can never act as approval.
seen=0
for tok in $cmd; do
  case "$tok" in -*|git|push|origin|upstream) continue ;; esac
  seen=1
  case "$tok" in *[!A-Za-z0-9_./:-]*) continue ;; esac   # punctuation -> not a ref
  if protected "$tok"; then
    deny "Refusing to push to main. Push an openhands/<issue>-<slug> branch and open a draft pull request instead."
  fi
done

# 'git push' with no refspec writes the current branch, so check that instead.
if [ "$seen" = 0 ]; then
  branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
  case "$branch" in
    main|master)
      deny "This push has no refspec and the current branch is $branch. Switch to an openhands/<issue>-<slug> branch first." ;;
  esac
fi
exit 0
