#!/usr/bin/env bash
# Build the mo-test images. Usage: build.sh [image ...]
#   e.g. build.sh            -> all four
#        build.sh cuda12     -> just the nvidia/cuda based one
set -euo pipefail
cd "$(dirname "$0")"

IMAGES=("$@")
[ ${#IMAGES[@]} -eq 0 ] && IMAGES=(ubuntu20.04 ubuntu22.04 ubuntu24.04 cuda12)

for img in "${IMAGES[@]}"; do
    echo "=== building mo-test:$img ==="
    docker build -t "mo-test:$img" -f "$img/Dockerfile" .
done
echo
echo "done: $(docker images --format '{{.Repository}}:{{.Tag}}' | grep '^mo-test:' | tr '\n' ' ')"
