#!/usr/bin/env bash
# Build the mo-test images. One parameterized Dockerfile; the per-cell
# arguments (base image, distro package set, python, arch pins) are defined
# below. See README.md for the support matrix and per-cell rationale.
#
# Usage: build.sh [cell ...]
#   e.g. build.sh                          -> all 12 cells
#        build.sh cuda12-ubuntu24.04       -> just the renamed full-coverage cell
#        build.sh ubuntu26.04 cuda13-ubuntu26.04
# WT_VERSION / OPENCV_VERSION can be overridden via the environment
# (overriding the per-cell defaults below).
set -euo pipefail
cd "$(dirname "$0")"

ALL_CELLS=(
    ubuntu20.04 ubuntu22.04 ubuntu24.04 ubuntu26.04
    cuda11-ubuntu20.04 cuda11-ubuntu22.04
    cuda12-ubuntu20.04 cuda12-ubuntu22.04 cuda12-ubuntu24.04
    cuda13-ubuntu22.04 cuda13-ubuntu24.04 cuda13-ubuntu26.04
)

# Skipped cells (NVIDIA does not support the pairing):
#   cuda11 x 24.04/26.04   11.8.0 (last CUDA 11) only ships 20.04/22.04 bases
#   cuda12 x 26.04         no 12.x base for 26.04 (first is 13.3.0)
#   cuda13 x 20.04         CUDA 13 only supports 22.04/24.04/26.04

cell_args() {
    local cell=$1
    if [[ " ${ALL_CELLS[*]} " != *" $cell "* ]]; then
        echo "unknown cell: $cell (known: ${ALL_CELLS[*]})" >&2
        return 1
    fi
    local BASE_IMAGE APP_PKGS WITH_PYTHON CUDA_ARCH_BIN CUDA_ARCH_PTX BOOST COMMON WITH_WT
    BASE_IMAGE="" APP_PKGS="" WITH_PYTHON=OFF CUDA_ARCH_BIN="" CUDA_ARCH_PTX="" WITH_WT=ON
    # Per-distro boost (version-suffixed) packages; the -dev ones without a
    # version suffix track the distro default and come with the cuda cells'
    # OpenCV/python builds.
    case "$cell" in
        *ubuntu20.04) BOOST="libboost1.71-dev libboost-fiber1.71-dev libboost-context1.71-dev" ;;
        *ubuntu22.04) BOOST="libboost1.74-dev libboost-fiber1.74-dev libboost-context1.74-dev" ;;
        *ubuntu24.04) BOOST="libboost1.83-dev libboost-fiber1.83-dev libboost-context1.83-dev" ;;
        *ubuntu26.04) BOOST="libboost1.90-dev libboost-fiber1.90-dev libboost-context1.90-dev" ;;
    esac
    # 26.04 cells: Wt 4.10.2 includes boost/asio/io_service.hpp, which the
    # Asio version shipped with their Boost 1.90 no longer provides (the
    # long-deprecated io_service API was removed; Asio itself is still part
    # of every Boost release). Wt 4.12+ tolerates it, but rather than mix
    # Wt versions across the matrix, skip Wt there for now (see README).
    case "$cell" in *ubuntu26.04) WITH_WT=OFF ;; esac
    COMMON="python3 $BOOST libboost-filesystem-dev libboost-thread-dev libboost-program-options-dev qtbase5-dev"

    case "$cell" in
        # --- CPU-only cells (apt OpenCV, WITH_PYTHON=OFF) ---
        ubuntu20.04)
            BASE_IMAGE=ubuntu:20.04
            APP_PKGS="$COMMON g++-10 clang libopencv-dev"
            ;;
        ubuntu22.04)
            BASE_IMAGE=ubuntu:22.04
            APP_PKGS="$COMMON g++-12 clang libopencv-dev"
            ;;
        ubuntu24.04)
            BASE_IMAGE=ubuntu:24.04
            APP_PKGS="$COMMON g++-12 clang libopencv-dev"
            ;;
        ubuntu26.04)
            BASE_IMAGE=ubuntu:26.04
            APP_PKGS="$COMMON g++-14 clang libopencv-dev"
            ;;
        # --- CUDA cells: WITH_PYTHON=ON, CUDA-enabled OpenCV from source ---
        cuda11-ubuntu20.04)
            # 11.8.0 is the last CUDA 11. Arch pins: nvcc 11.8 tops out at
            # sm_90, so it cannot target the sm_120 test host; the embedded
            # compute_90 PTX is JIT'd by the driver so device tests still run.
            # g++-10 is fine as an nvcc host (official support to GCC 11).
            BASE_IMAGE=nvidia/cuda:11.8.0-devel-ubuntu20.04
            APP_PKGS="$COMMON g++-10 clang python3-dev python3-numpy libjpeg-dev libpng-dev libtiff-dev"
            WITH_PYTHON=ON
            CUDA_ARCH_BIN=90
            CUDA_ARCH_PTX=90
            ;;
        cuda11-ubuntu22.04)
            # No g++-12: nvcc 11.8's host-compiler support ends at GCC 11.
            BASE_IMAGE=nvidia/cuda:11.8.0-devel-ubuntu22.04
            APP_PKGS="$COMMON clang python3-dev python3-numpy libjpeg-dev libpng-dev libtiff-dev"
            WITH_PYTHON=ON
            CUDA_ARCH_BIN=90
            CUDA_ARCH_PTX=90
            ;;
        cuda12-ubuntu20.04)
            # 12.9.2 is the last 12.x that still ships a focal base.
            BASE_IMAGE=nvidia/cuda:12.9.2-devel-ubuntu20.04
            APP_PKGS="$COMMON g++-10 clang python3-dev python3-numpy libjpeg-dev libpng-dev libtiff-dev"
            WITH_PYTHON=ON
            ;;
        cuda12-ubuntu22.04)
            BASE_IMAGE=nvidia/cuda:12.9.2-devel-ubuntu22.04
            APP_PKGS="$COMMON g++-12 clang python3-dev python3-numpy libjpeg-dev libpng-dev libtiff-dev"
            WITH_PYTHON=ON
            ;;
        cuda12-ubuntu24.04)
            # Successor of the old mo-test:cuda12 image (12.8.1 -> 12.9.2,
            # the current latest 12.x).
            BASE_IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04
            APP_PKGS="$COMMON g++-12 clang python3-dev python3-numpy libjpeg-dev libpng-dev libtiff-dev"
            WITH_PYTHON=ON
            ;;
        cuda13-ubuntu22.04)
            # 13.4.2 is the latest CUDA 13.
            BASE_IMAGE=nvidia/cuda:13.4.2-devel-ubuntu22.04
            APP_PKGS="$COMMON g++-12 clang python3-dev python3-numpy libjpeg-dev libpng-dev libtiff-dev"
            WITH_PYTHON=ON
            ;;
        cuda13-ubuntu24.04)
            BASE_IMAGE=nvidia/cuda:13.4.2-devel-ubuntu24.04
            APP_PKGS="$COMMON g++-12 clang python3-dev python3-numpy libjpeg-dev libpng-dev libtiff-dev"
            WITH_PYTHON=ON
            ;;
        cuda13-ubuntu26.04)
            # 26.04 (resolute): first distro of the matrix with CMake 4.x;
            # run_tests.sh exports CMAKE_POLICY_VERSION_MINIMUM for it.
            BASE_IMAGE=nvidia/cuda:13.4.2-devel-ubuntu26.04
            APP_PKGS="$COMMON g++-14 clang python3-dev python3-numpy libjpeg-dev libpng-dev libtiff-dev"
            WITH_PYTHON=ON
            ;;
        *)
            echo "unknown cell: $cell (known: ${ALL_CELLS[*]})" >&2
            return 1
            ;;
    esac

    BUILD_ARGS=(
        --build-arg BASE_IMAGE="$BASE_IMAGE"
        --build-arg APP_PKGS="$APP_PKGS"
        --build-arg WITH_PYTHON="$WITH_PYTHON"
        --build-arg CUDA_ARCH_BIN="$CUDA_ARCH_BIN"
        --build-arg CUDA_ARCH_PTX="$CUDA_ARCH_PTX"
        --build-arg WITH_WT="$WITH_WT"
    )
    if [ -n "${WT_VERSION:-}" ]; then BUILD_ARGS+=(--build-arg WT_VERSION="$WT_VERSION"); fi
    if [ -n "${OPENCV_VERSION:-}" ]; then BUILD_ARGS+=(--build-arg OPENCV_VERSION="$OPENCV_VERSION"); fi
    return 0
}

IMAGES=("$@")
[ ${#IMAGES[@]} -eq 0 ] && IMAGES=("${ALL_CELLS[@]}")

for img in "${IMAGES[@]}"; do
    echo "=== building mo-test:$img ==="
    cell_args "$img"
    docker build "${BUILD_ARGS[@]}" -t "mo-test:$img" -f Dockerfile .
done
echo
echo "done: $(docker images --format '{{.Repository}}:{{.Tag}}' | grep '^mo-test:' | tr '\n' ' ')"
