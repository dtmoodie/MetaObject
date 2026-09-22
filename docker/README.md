# MetaObject platform test images

Small, standalone Docker images that configure, build, and run the full
MetaObject test suite (`test_mo_*` binaries + python interop) across the
same distro/toolchain matrix as the `ct` test images
(`../ct/docker/`). The source tree (including the `dependencies/` submodules)
is mounted read-only at runtime, so the images always test the current tree.

## Images

| Tag | Base | Compilers tested | Boost | OpenCV | Python module |
|---|---|---|---|---|---|
| `mo-test:ubuntu20.04` | ubuntu:20.04 | GCC 9, 10; Clang 10 | 1.71 | 4.5 (CPU, apt) | OFF |
| `mo-test:ubuntu22.04` | ubuntu:22.04 | GCC 11, 12; Clang 14 | 1.74 | 4.5 (CPU, apt) | OFF |
| `mo-test:ubuntu24.04` | ubuntu:24.04 | GCC 13, 12; Clang 18 | 1.83 | 4.6 (CPU, apt) | OFF |
| `mo-test:cuda12` | nvidia/cuda:12.8.1-devel-ubuntu24.04 | GCC 13, 12; Clang 18 (+ nvcc 12.8) | 1.83 | 4.14 **with CUDA** (source build + contrib) | ON |

All images install the full optional-dependency set (Boost fiber/context,
OpenCV, Qt5, Wt); `WITH_WT=ON` is passed explicitly since it defaults OFF.
The `cuda12` image is the only one with full coverage: it also enables the
`cuda` module and the `python` module, and its test run uses `--gpus all`.

Wt is not in any current Ubuntu release's apt repos (verified on focal,
jammy and noble), so every image builds Wt 4.10.2 from source in a
build-stage on the same base distro and copies the install into the final
image.

The plugin mechanism (`cmake/metaobject_declare_plugin.cmake`) invokes bare
`python` at configure time to generate `plugin_config2.cpp`, but these
distros only ship `python3`; each image therefore symlinks
`/usr/local/bin/python` → `python3` (`cmake/parse_cmake.py` is Python 3
compatible, so no more is needed).

## Why the python module is OFF everywhere except cuda12

Two module-level hard requirements (not CMake gates) shape the matrix:

1. `modules/metaparams` unconditionally includes OpenCV headers
   (`ct/types/opencv.hpp`) — so every image installs OpenCV.
2. `modules/python/.../PythonSetup.cpp` unconditionally includes
   `<MetaObject/cuda/opencv.hpp>` → `<opencv2/core/cuda.hpp>`, which only
   exists in CUDA-enabled OpenCV. The CMake gate for the python module
   (`MO_HAVE_PYTHON AND MO_HAVE_OPENCV`) does not catch this — with CPU
   OpenCV the module *configures* and then fails to compile.

Additionally, `cmake/dependencies/opencv.cmake` force-sets `WITH_CUDA` from
`OpenCV_CUDA_VERSION` (plain `set`, overriding `-DWITH_CUDA`), so CPU OpenCV
silently disables the cuda module. CUDA coverage therefore only exists in
`cuda12`, where OpenCV is built from source with CUDA — which is also why
`cuda12` uses CUDA 12.8 instead of ct's 12.5: the toolkit must emit SASS
for sm_120 (Blackwell) because the cuda tests execute device code.

## Usage

```bash
./build.sh                  # build all four images (cuda12 takes longest:
./build.sh cuda12           # it builds OpenCV with CUDA inside the image)

./test.sh                   # run the suite in every built image
./test.sh ubuntu24.04       # one image
COMPILERS="g++-13" ./test.sh cuda12   # restrict toolchains
```

Each image auto-detects installed compilers (`g++`, `g++-12`, `g++-13`,
`clang++`, ...) and runs configure+build+tests once per compiler; the image
exits non-zero if any compiler fails, so `test.sh` is CI-friendly.

`COMPILERS`, `CUDA_ARCH_BIN` and `JOBS` are passed through only when set on
the host; otherwise the image defaults apply (in particular the `cuda12`
image's own `CUDA_ARCH_BIN=120;90`).

Tests are run by executing the `test_mo_*` binaries in the build dir
directly rather than via ctest, because ctest coverage is incomplete:
`test_mo_core`/`test_mo_types` have no `add_test()` entry, the params test
is mis-registered as `test_mo_cuda` (colliding with the cuda module's test),
and the python ctest entry invokes `python2` on a file name that does not
exist.

Known harness shim: the generated python wrapper (`metaobject.py`) uses
`import imp`, removed in Python 3.12. On Python ≥ 3.12 `run_tests.sh`
rewrites the wrapper with `importlib` before running the interop test; the
upstream template (`modules/python/metaobject.py.in.unix`) still needs that
fix.

## Prerequisites

- Submodules initialized **recursively** (the `ct` submodule's nested
  `googletest` provides the `gtest` targets):
  `git submodule update --init --recursive`
- For `cuda12`: NVIDIA driver + nvidia-container-toolkit on the host
  (`test.sh` passes `--gpus all`).
