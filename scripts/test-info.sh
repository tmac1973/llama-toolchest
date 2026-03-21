#!/usr/bin/env bash
# Test model info and PS endpoints.
# Usage: ./scripts/test-info.sh

set -euo pipefail
source "$(dirname "$0")/lib-test.sh"

HOST="${LLAMACTL_HOST:-localhost}"
PORT="${LLAMACTL_PORT:-3000}"
MGMT_URL="http://${HOST}:${PORT}"

echo "--- /api/ps (loaded models) ---"
curl -s "${MGMT_URL}/api/ps" | python3 -c "
import sys, json
r = json.load(sys.stdin)
models = r.get('models', [])
if not models:
    print('No models in router.')
else:
    for m in models:
        status = m.get('status', 'unknown')
        icon = '●' if status == 'loaded' else '○'
        vram = f\"{m.get('vram_est_gb', 0):.1f}GB\" if m.get('vram_est_gb') else ''
        ctx = f\"ctx:{m.get('context_size', '')}\" if m.get('context_size') else ''
        arch = m.get('arch', '')
        extras = ' '.join(filter(None, [arch, vram, ctx]))
        print(f'  {icon} {m[\"name\"]}  {extras}')
" 2>/dev/null || {
    echo "Failed to reach /api/ps"
    exit 1
}

echo ""
echo "--- /api/models/{id}/info ---"
pick_model
echo ""

curl -s "${MGMT_URL}/api/models/${MODEL}/info" | python3 -c "
import sys, json
info = json.load(sys.stdin)
if 'error' in info:
    print(f'Error: {info}')
    sys.exit(1)
print(f'Model:        {info[\"model_id\"]}')
print(f'Architecture: {info.get(\"arch\", \"unknown\")}')
print(f'Quant:        {info.get(\"quant\", \"unknown\")}')
print(f'Context:      {info.get(\"context_length\", \"unknown\")}')
print(f'Size:         {info.get(\"size_bytes\", 0) / 1e9:.1f} GB')
print(f'VRAM Est:     {info.get(\"vram_est_gb\", 0):.1f} GB')
print(f'Capabilities: {', '.join(info.get(\"capabilities\", []))}')
cfg = info.get('config', {})
if cfg:
    print(f'GPU Layers:   {cfg.get(\"gpu_layers\", \"\")}')
    print(f'Context Size: {cfg.get(\"context_size\", \"\")}')
    print(f'Threads:      {cfg.get(\"threads\", \"\")}')
" 2>/dev/null || {
    echo "Failed to fetch model info"
}
