#!/usr/bin/env bash
# Configure, build and run the MetaObject test suite once per installed compiler.
#
# Runs inside the mo-test images with the MetaObject source tree mounted
# read-only at /src. Build output goes to /build/<compiler> (container-local,
# discarded on exit).
#
# The test binaries are executed directly rather than via ctest, because:
#   - test_mo_core and test_mo_types have no add_test() entry at all
#   - the params test is mis-registered in ctest as "test_mo_cuda"
#     (name collision with the cuda module's test)
#   - the python ctest entry invokes python2 on a file name that does not
#     exist (the configured copy is metaobject_python_test.py)
#
# Env:
#   SRCDIR      source tree (default /src)
#   BUILDDIR    build root (default /build)
#   COMPILERS   space-separated cxx drivers to test; default: auto-detect
#   WITH_PYTHON ON/OFF (default OFF). The python module unconditionally
#               includes <opencv2/core/cuda.hpp>, so it only builds against
#               CUDA-enabled OpenCV: only the cuda12 image sets it ON.
#   CUDA_ARCH_BIN  -gencode architectures for nvcc (default: FindCUDA default)
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
CUDA_ARCH_BIN=${CUDA_ARCH_BIN:-}
JOBS=${JOBS:-$(nproc)}
TEST_TIMEOUT=${TEST_TIMEOUT:-600}

if [ -n "${COMPILERS:-}" ]; then
    read -r -a COMPILERS <<<"$COMPILERS"
else
    COMPILERS=()
    for c in g++ g++-9 g++-10 g++-12 g++-13 clang++ clang++-10 clang++-14 clang++-16 clang++-18; do
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
        -DWITH_WT=ON \
        -DWITH_PYTHON="$WITH_PYTHON" \
        ${CUDA_ARCH_BIN:+-DCUDA_ARCH_BIN="$CUDA_ARCH_BIN"}; then
        echo "!!!! configure failed for $cxx"
        overall=1
        continue
    fi
    if ! make -C "$b" -j"$JOBS"; then
        echo "!!!! build failed for $cxx"
        overall=1
        continue
    fi

    # The python module's generated wrapper (metaobject.py) uses `import imp`,
    # which was removed in Python 3.12. Rewrite it with importlib so the C++
    # module can still be exercised; this is a test-harness shim, not a fix.
    rewrite_python_wrapper() {
        local dir=$1
        [ -f "$dir/metaobject.py" ] || return 0
        if python3 -c "import imp" >/dev/null 2>&1; then
            return 0
        fi
        echo "  (rewriting metaobject.py to use importlib: 'imp' removed in Python 3.12)"
        cat > "$dir/metaobject.py" <<'PYEOF'
import glob
import importlib.machinery
import importlib.util
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_so = glob.glob(os.path.join(_here, "libmetaobject_python*.so"))[0]
_spec = importlib.util.spec_from_file_location(
    __name__, _so, loader=importlib.machinery.ExtensionFileLoader(__name__, _so)
)
_ext = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ext)
sys.modules[__name__] = _ext
PYEOF
    }

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

    # Python interop test (cuda12 image only: requires the python module,
    # which needs CUDA-enabled OpenCV).
    if [ -f "$b/metaobject_python_test.py" ]; then
        rewrite_python_wrapper "$b"
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
