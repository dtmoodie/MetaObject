#!/usr/bin/env bash
# Run the MetaObject test suite in each test image. The source tree is
# mounted read-only at /src; builds happen in container-local /build.
#
# Usage: test.sh [image ...]
#   e.g. test.sh                    -> all built mo-test images, default compilers
#        test.sh ubuntu24.04        -> one image
# COMPILERS can restrict the toolchains,
# e.g. COMPILERS="g++-13" test.sh cuda12-ubuntu24.04
#
# cuda* images are run with --gpus all (the cuda test binaries execute
# device code).
set -uo pipefail
cd "$(dirname "$0")"

MO_SRC=${MO_SRC:-$(cd .. && pwd)}

IMAGES=("$@")
if [ ${#IMAGES[@]} -eq 0 ]; then
    IMAGES=($(docker images --format '{{.Tag}}' mo-test | sort -r))
    [ ${#IMAGES[@]} -eq 0 ] && { echo "no mo-test images found; run build.sh first" >&2; exit 1; }
fi

overall=0
for img in "${IMAGES[@]}"; do
    echo
    echo "#################### $img ####################"
    gpu=""
    case "$img" in
        cuda*) gpu="--gpus all" ;;
    esac
    # Only pass through env vars that are actually set on the host: a bare
    # `-e VAR` for an unset VAR blanks the image's own ENV (which would
    # silently drop the CUDA_ARCH_BIN default baked into the cuda11 images).
    env_args=()
    for v in COMPILERS CUDA_ARCH_BIN CUDA_ARCH_PTX JOBS; do
        v_val="${!v:-}"
        [ -n "$v_val" ] && env_args+=("-e" "$v=$v_val")
    done
    # shellcheck disable=SC2086
    docker run --rm $gpu "${env_args[@]}" \
        -v "$MO_SRC:/src:ro" \
        "mo-test:$img" || overall=1
done
exit $overall
