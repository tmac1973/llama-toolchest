#!/usr/bin/env bash
set -euo pipefail

detect_gpu() {
    # NVIDIA: check for nvidia-smi or /dev/nvidia0
    if command -v nvidia-smi &>/dev/null || [ -e /dev/nvidia0 ]; then
        echo "cuda"
        return
    fi

    # AMD: check for /dev/kfd (ROCm kernel driver)
    if [ -e /dev/kfd ]; then
        echo "rocm"
        return
    fi

    # Intel: check for Intel render nodes via sysfs
    if [ -d /sys/class/drm ]; then
        for vendor_file in /sys/class/drm/*/device/vendor; do
            if [ -f "$vendor_file" ] && [ "$(cat "$vendor_file")" = "0x8086" ]; then
                echo "intel"
                return
            fi
        done
    fi

    echo "cpu"
}

GPU="${GPU:-$(detect_gpu)}"

# Allow "setup.sh detect" to just print the detected GPU
if [ "${1:-}" = "detect" ]; then
    echo "$GPU"
    exit 0
fi

COMPOSE_FILE="docker-compose.${GPU}.yml"

if [ ! -f "$COMPOSE_FILE" ]; then
    echo "Error: $COMPOSE_FILE not found (detected GPU: $GPU)"
    echo "Available compose files:"
    ls -1 docker-compose.*.yml 2>/dev/null || echo "  (none)"
    exit 1
fi

echo "GPU backend: $GPU"
echo "Compose file: $COMPOSE_FILE"

case "${1:-up}" in
    up)
        docker compose -f "$COMPOSE_FILE" up -d --build
        ;;
    down)
        docker compose -f "$COMPOSE_FILE" down
        ;;
    rebuild)
        docker compose -f "$COMPOSE_FILE" down
        docker compose -f "$COMPOSE_FILE" build --no-cache
        docker compose -f "$COMPOSE_FILE" up -d
        ;;
    logs)
        docker compose -f "$COMPOSE_FILE" logs -f
        ;;
    *)
        echo "Usage: $0 [up|down|rebuild|logs|detect]"
        echo "  Override GPU: GPU=cuda $0 up"
        exit 1
        ;;
esac
