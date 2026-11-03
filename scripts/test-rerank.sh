#!/usr/bin/env bash
# Test reranking model support via /v1/rerank and the root /rerank.
# Usage: ./scripts/test-rerank.sh [--host HOST] [--port PORT] [model-name]
#
# Sends a query with three documents, one of them the answer, and checks
# that both routes rank that one first. Exits non-zero if not.

set -euo pipefail
source "$(dirname "$0")/lib-test.sh"

if [[ ${#ARGS[@]} -gt 0 ]]; then
    MODEL="${ARGS[0]}"
else
    pick_model rerank
fi
echo ""

# The answer is document 1.
BODY=$(jq -n --arg model "$MODEL" '{
    model: $model,
    query: "Which animal eats bamboo and lives in the mountains of China?",
    documents: [
        "The stock market fell sharply on Tuesday.",
        "The giant panda is a bear native to the mountains of central China that feeds almost entirely on bamboo.",
        "Pandas is a Python library for data analysis."
    ]
}')

FAILED=0
for URL in "${BASE_URL}/rerank" "${MGMT_URL}/rerank"; do
    echo "--- POST ${URL#"${MGMT_URL}"} ---"
    RESPONSE=$(curl -s "$URL" -H "Content-Type: application/json" -d "$BODY")
    if ! echo "$RESPONSE" | jq -e '.results' &>/dev/null; then
        echo "$RESPONSE" | jq . 2>/dev/null || echo "$RESPONSE"
        FAILED=1
        continue
    fi
    echo "$RESPONSE" | jq '{results: [.results[] | {index, relevance_score}], usage}'
    TOP=$(echo "$RESPONSE" | jq '.results | max_by(.relevance_score) | .index')
    if [[ "$TOP" == "1" ]]; then
        echo "OK: the panda document ranks first"
    else
        echo "FAIL: document ${TOP} ranks first, want 1"
        FAILED=1
    fi
    echo ""
done
exit "$FAILED"
