#!/usr/bin/env bash
# Test embedding model support via the OpenAI-compatible embeddings API.
# Usage: ./scripts/test-embeddings.sh [model-name]
#
# Requires an embedding model enabled and available in the router.

set -euo pipefail

HOST="${LLAMACTL_HOST:-localhost}"
PORT="${LLAMACTL_PORT:-3000}"
BASE_URL="http://${HOST}:${PORT}/v1"

MODEL="${1:-}"

if [[ -z "$MODEL" ]]; then
    echo "No model specified. Listing available models..."
    curl -s "${BASE_URL}/models" | python3 -m json.tool 2>/dev/null || curl -s "${BASE_URL}/models"
    echo ""
    echo "Usage: $0 <model-name>"
    echo "Pick an embedding model from the list above."
    exit 1
fi

echo "==> Testing embeddings with model: ${MODEL}"
echo ""

# Single string embedding
echo "--- Single input ---"
RESPONSE=$(curl -s "${BASE_URL}/embeddings" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${MODEL}\", \"input\": \"The quick brown fox jumps over the lazy dog\"}")

# Check for errors
ERROR=$(echo "$RESPONSE" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('error',{}).get('message',''))" 2>/dev/null || true)
if [[ -n "$ERROR" ]]; then
    echo "Error: ${ERROR}"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
    exit 1
fi

DIMS=$(echo "$RESPONSE" | python3 -c "
import sys, json
r = json.load(sys.stdin)
emb = r['data'][0]['embedding']
print(f'Dimensions: {len(emb)}')
print(f'First 5 values: {emb[:5]}')
print(f'Model: {r.get(\"model\", \"unknown\")}')
print(f'Usage: {r.get(\"usage\", {})}')
" 2>/dev/null || echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE")
echo "$DIMS"

# Batch embedding + similarity test
echo ""
echo "--- Batch similarity test ---"
RESPONSE=$(curl -s "${BASE_URL}/embeddings" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${MODEL}\", \"input\": [\"cat\", \"kitten\", \"automobile\"]}")

python3 -c "
import sys, json, math

r = json.load(sys.stdin)
if 'error' in r:
    print(f'Error: {r[\"error\"][\"message\"]}')
    sys.exit(1)

embeddings = [d['embedding'] for d in r['data']]
labels = ['cat', 'kitten', 'automobile']

def cosine_sim(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot / (na * nb) if na and nb else 0

for i in range(len(labels)):
    for j in range(i+1, len(labels)):
        sim = cosine_sim(embeddings[i], embeddings[j])
        print(f'  {labels[i]} <-> {labels[j]}: {sim:.4f}')

print()
print('(cat<->kitten should be much higher than cat<->automobile)')
" <<< "$RESPONSE"
