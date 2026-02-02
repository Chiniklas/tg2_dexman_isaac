#!/usr/bin/env bash
set -euo pipefail

xhost +si:localuser:$(whoami) >/dev/null 2>&1
xhost +si:localuser:root >/dev/null 2>&1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/../docker-compose.yml"
GPU_COMPOSE_FILE="$SCRIPT_DIR/../docker-compose.gpu.yml"

compose_args=("-f" "$COMPOSE_FILE")

if [[ -f "$GPU_COMPOSE_FILE" ]]; then
  runtimes="$(docker info --format '{{json .Runtimes}}' 2>/dev/null || echo '{}')"
  if grep -q '"nvidia"' <<<"$runtimes"; then
    compose_args+=("-f" "$GPU_COMPOSE_FILE")
  else
    printf 'Warning: NVIDIA container runtime not detected; running without GPU acceleration.\n' >&2
  fi
fi

docker compose "${compose_args[@]}" run --rm --service-ports workspace bash
