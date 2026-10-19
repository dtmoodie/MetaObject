#!/usr/bin/env bash
# Configure, build and run the MetaObject test suite once per installed compiler.
#
# Runs inside the mo-test images with the MetaObject source tree mounted
# read-only at /src. Build output goes to /build/<compiler> (container-local,
# discarded on exit).
#
# The test binaries are executed directly rather than via ctest. A full
# `ctest` run from the build root now passes end-to-end (every metaobject
# target and the ct submodule's suites register resolvable paths, as of
# ct e0ddb75), but the binaries are driven directly here: it is the
# canonical CI flow, and the per-binary TEST_TIMEOUT catches a hung test
# (e.g. a stalled GPU kernel) instead of freezing the matrix.
#
# Env:
#   SRCDIR      source tree (default /src)
#   BUILDDIR    build root (default /build)
#   COMPILERS   space-separated cxx drivers to test; default: auto-detect
#   WITH_PYTHON ON/OFF (default OFF). The python module unconditionally
#               includes <opencv2/core/cuda.hpp>, so it only builds against
#               CUDA-enabled OpenCV: only the cuda* images set it ON.
#   WITH_WT     ON/OFF (default ON). The 26.04 images export OFF: Wt 4.10.2
#               cannot be built against the Asio that ships with their
#               Boost 1.90 (see README), so the wtgui module is not built
#               there.
#   CUDA_ARCH_BIN  -gencode architectures for nvcc. When set, it is passed as
#               -DCUDA_ARCH_NAME=Manual (the vendored FindCUDA silently
#               discards -DCUDA_ARCH_BIN in the default Auto mode, which
#               auto-detects the attached GPUs and emits SASS only). Unset
#               keeps Auto: SASS for the attached GPU. The cuda11 images set
#               it to 90 because nvcc 11.8 cannot target sm_120.
#   CUDA_ARCH_PTX  extra PTX architectures (passed with -DCUDA_ARCH_PTX when
#               CUDA_ARCH_BIN is set). The cuda11 images set it to 90 so the
#               embedded compute_90 PTX is JIT'd by the driver on newer GPUs
#               (e.g. sm_120) and the device tests still execute.
#   JOBS           parallelism for make (default: nproc)
#   TEST_TIMEOUT   per-test-binary wall clock in seconds (default 600). A hung
#               binary must fail its cell instead of freezing the matrix
#               (this is what happened when a driver stalled the GPU under
#               test_mo_cuda's spin kernels); the default is generous — the
#               slowest binary finishes in well under a minute.
set -uo pipefail

SRCDIR=${SRCDIR:-/src}
BUILDDIR=${BUILDDIR:-/build}
WITH_PYTHON=${WITH_PYTHON:-OFF}
WITH_WT=${WITH_WT:-ON}
CUDA_ARCH_BIN=${CUDA_ARCH_BIN:-}
CUDA_ARCH_PTX=${CUDA_ARCH_PTX:-}
JOBS=${JOBS:-$(nproc)}
TEST_TIMEOUT=${TEST_TIMEOUT:-600}

# The 26.04 images ship CMake 4.x, which rejects this tree's
# cmake_minimum_required(2.8) (CMake 4 dropped compatibility with < 3.5).
# The env var downgrades the policy floor for the whole configure; CMake 3.x
# ignores it.
export CMAKE_POLICY_VERSION_MINIMUM=3.5

# An explicit arch list selects FindCUDA's Manual mode (see the
# CUDA_ARCH_BIN note above); otherwise the default Auto mode applies.
arch_args=()
if [ -n "$CUDA_ARCH_BIN" ]; then
    arch_args+=(-DCUDA_ARCH_NAME=Manual -DCUDA_ARCH_BIN="$CUDA_ARCH_BIN")
    if [ -n "$CUDA_ARCH_PTX" ]; then
        arch_args+=(-DCUDA_ARCH_PTX="$CUDA_ARCH_PTX")
    fi
fi

if [ -n "${COMPILERS:-}" ]; then
    read -r -a COMPILERS <<<"$COMPILERS"
else
    COMPILERS=()
    for c in g++ g++-9 g++-10 g++-11 g++-12 g++-13 g++-14 g++-15 g++-16 clang++ clang++-10 clang++-14 clang++-16 clang++-18 clang++-19 clang++-20 clang++-21; do
        command -v "$c" >/dev/null 2>&1 && COMPILERS+=("$c")
    done
fi

# Always-present test binaries (built in every configuration).
REQUIRED_TESTS=(test_mo_core test_mo_types test_mo_params test_mo_serialization test_mo_runtime_reflection test_mo_object)
# Configuration-dependent ones: run only when the configuration produced them.
OPTIONAL_TESTS=(test_mo_cuda test_mo_cuda_fiber test_mo_python_init)

overall=0
for cxx in "${COMPILERS[@]}"; do
    # "clang++" contains "g++", so derive clang first
    case "$cxx" in
        clang*) cc="${cxx/++/}" ;;
        *)      cc="${cxx/g++/gcc}" ;;
    esac
    b="$BUILDDIR/$(echo "$cxx" | tr '+/-' '___')"
    echo
    echo "############## compiler: $cxx (cc: $cc) ##############"
    if ! cmake -S "$SRCDIR" -B "$b" \
        -DCMAKE_C_COMPILER="$cc" \
        -DCMAKE_CXX_COMPILER="$cxx" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTS=ON \
        -DBUILD_EXAMPLES=ON \
        -DBUILD_DEPENDENCIES=ON \
        -DWITH_QT=ON \
        -DWITH_OPENCV=ON \
        -DWITH_WT="$WITH_WT" \
        -DWITH_PYTHON="$WITH_PYTHON" \
        "${arch_args[@]}"; then
        echo "!!!! configure failed for $cxx"
        overall=1
        continue
    fi
    if ! make -C "$b" -j"$JOBS"; then
        echo "!!!! build failed for $cxx"
        overall=1
        continue
    fi

    fail=0
    for test in "${REQUIRED_TESTS[@]}"; do
        if [ ! -x "$b/$test" ]; then
            echo "!!!! missing binary $test for $cxx"
            fail=1
            continue
        fi
        if [ "$test" = "test_mo_object" ]; then
            # The recompile.* tests drive RCC's runtime compiler, which is
            # never configured with include/library paths anywhere in the
            # stack (no AddCompilerIncludeDirectory/... calls exist), and
            # also requires a writable source tree (TestBuildFile utime-
            # touches the runtime sources). Upstream Linux CI (travis) has
            # excluded them with this same filter for that reason; match it.
            if ! (cd "$b" && timeout -s KILL "$TEST_TIMEOUT" "./$test" --gtest_filter='object.*:object_factory.*:object_reflection.*:object_initialization.*:serialization.*:signal.*:slot.*'); then
                echo "!!!! $test failed (or exceeded ${TEST_TIMEOUT}s) for $cxx"
                fail=1
            fi
        elif ! (cd "$b" && timeout -s KILL "$TEST_TIMEOUT" "./$test"); then
            echo "!!!! $test failed (or exceeded ${TEST_TIMEOUT}s) for $cxx"
            fail=1
        fi
    done
    for test in "${OPTIONAL_TESTS[@]}"; do
        [ -x "$b/$test" ] || continue
        if ! (cd "$b" && timeout -s KILL "$TEST_TIMEOUT" "./$test"); then
            echo "!!!! $test failed (or exceeded ${TEST_TIMEOUT}s) for $cxx"
            fail=1
        fi
    done

    # Python interop test (cuda* images only: requires the python module,
    # which needs CUDA-enabled OpenCV).
    if [ -f "$b/metaobject_python_test.py" ]; then
        if ! (cd "$b" && timeout -s KILL "$TEST_TIMEOUT" python3 metaobject_python_test.py); then
            echo "!!!! python interop test failed (or exceeded ${TEST_TIMEOUT}s) for $cxx"
            fail=1
        fi
    fi

    if [ "$fail" -ne 0 ]; then
        echo "!!!! $cxx: failures (see above)"
        overall=1
    else
        echo "############## $cxx: all tests passed ##############"
    fi
done
exit $overall
