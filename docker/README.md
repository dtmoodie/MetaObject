# MetaObject platform test images

Small, standalone Docker images that configure, build, and run the full
MetaObject test suite (`test_mo_*` binaries + python interop) across the
support matrix below. The source tree (including the `dependencies/`
submodules) is mounted read-only at runtime, so the images always test the
current tree.

All cells are built from one parameterized `Dockerfile`; `build.sh` holds the
cell → arguments matrix (base image, distro package set, python, Wt, arch
pins). Each cell installs the full optional-dependency set (Boost
fiber/context, OpenCV, Qt5, Wt — except the 26.04 cells, below);
`WITH_WT=ON` is passed explicitly since it defaults OFF.

## Support matrix

`mo-test:<cell>`; every `cuda*` cell runs with `--gpus all` and enables the
`cuda` module, the `python` module and a source-built CUDA-enabled OpenCV
(full coverage). The CPU cells keep `WITH_PYTHON=OFF` (see below).

| Cell | Base image | Toolchain (compilers tested) |
|---|---|---|
| `ubuntu20.04` | ubuntu:20.04 | GCC 9 (default) + 10; Clang 10; Boost 1.71; apt OpenCV 4.2; CMake 3.16 |
| `ubuntu22.04` | ubuntu:22.04 | GCC 11 (default) + 12; Clang 14; Boost 1.74; apt OpenCV 4.5; CMake 3.22 |
| `ubuntu24.04` | ubuntu:24.04 | GCC 13 (default) + 12; Clang 18; Boost 1.83; apt OpenCV 4.6; CMake 3.28 |
| `ubuntu26.04` | ubuntu:26.04 | GCC 15 (default) + 14; Clang 21; Boost 1.90; apt OpenCV 4.10; CMake 4.2 (policy shim, below); Python 3.14; Wt skipped (below) |
| `cuda11-ubuntu20.04` | nvidia/cuda:11.8.0-devel-ubuntu20.04 | + nvcc 11.8 (last CUDA 11); GCC 9 + 10; Clang 10; archs pinned 90 (+PTX 90) |
| `cuda11-ubuntu22.04` | nvidia/cuda:11.8.0-devel-ubuntu22.04 | + nvcc 11.8; GCC 11; Clang 14; archs pinned 90 (+PTX 90) — no GCC 12 (nvcc 11.8 host support ends at GCC 11) |
| `cuda12-ubuntu20.04` | nvidia/cuda:12.9.2-devel-ubuntu20.04 | + nvcc 12.9 (last 12.x with a focal base); GCC 9 + 10; Clang 10 |
| `cuda12-ubuntu22.04` | nvidia/cuda:12.9.2-devel-ubuntu22.04 | + nvcc 12.9; GCC 11 + 12; Clang 14 |
| `cuda12-ubuntu24.04` | nvidia/cuda:12.9.2-devel-ubuntu24.04 | + nvcc 12.9 (current latest 12.x); GCC 13 + 12; Clang 18 |
| `cuda13-ubuntu22.04` | nvidia/cuda:13.4.2-devel-ubuntu22.04 | + nvcc 13.4 (current latest 13.x); GCC 11 + 12; Clang 14 |
| `cuda13-ubuntu24.04` | nvidia/cuda:13.4.2-devel-ubuntu24.04 | + nvcc 13.4; GCC 13 + 12; Clang 18 |
| `cuda13-ubuntu26.04` | nvidia/cuda:13.4.2-devel-ubuntu26.04 | + nvcc 13.4; GCC 15 + 14; Clang 21; Python 3.14 + Boost.Python 1.90; Wt skipped (below) |

Skipped cells (NVIDIA does not support the pairing, verified against the
`nvidia/cuda` registry and the CUDA 13.4 OS support table, 2026-10):

- `cuda11` × 24.04 / 26.04 — CUDA 11.8 (the last 11.x) only ships 20.04/22.04 bases.
- `cuda12` × 26.04 — no 12.x base for 26.04; the first CUDA base for 26.04 is 13.3.0.
- `cuda13` × 20.04 — CUDA 13 only supports Ubuntu 22.04 / 24.04 / 26.04.

Renamed: the old `mo-test:cuda12` image (12.8.1 on 24.04) is now
`mo-test:cuda12-ubuntu24.04` on 12.9.2. The CPU cell tags are unchanged.

## Why the python module is OFF on the CPU cells

Two module-level hard requirements (not CMake gates) shape the matrix:

1. `modules/metaparams` unconditionally includes OpenCV headers
   (`ct/types/opencv.hpp`) — so every cell installs OpenCV.
2. `modules/python/.../PythonSetup.cpp` unconditionally includes
   `<MetaObject/cuda/opencv.hpp>` → `<opencv2/core/cuda.hpp>`, which only
   exists in CUDA-enabled OpenCV. The CMake gate for the python module
   (`MO_HAVE_PYTHON AND MO_HAVE_OPENCV`) does not catch this — with CPU
   OpenCV the module *configures* and then fails to compile.

Additionally, `cmake/dependencies/opencv.cmake` force-sets `WITH_CUDA` from
`OpenCV_CUDA_VERSION` (plain `set`, overriding `-DWITH_CUDA`), so CPU OpenCV
silently disables the cuda module. CUDA coverage therefore only exists in the
`cuda*` cells, where OpenCV is built from source with CUDA.

Every `cuda*` cell builds OpenCV **4.14.0 + opencv_contrib 4.14.0** (same
tag, `BUILD_LIST="core;imgproc;cudaarithm;cudev"`). 4.14.0 is the latest 4.x
and still builds against CUDA 11.8 (no 4.x release raised the CUDA minimum),
and the contrib 4.14.0 tag is required for CUDA 13.2+ (the cudev namespace
fix, opencv_contrib#4097 — contrib 4.13.0 does not build with 13.2+). OpenCV
FATALs at configure if the `cudev` module is missing with `WITH_CUDA=ON`,
hence `cudev` in `BUILD_LIST`. The base images' cuDNN (≥ 7.5) satisfies
OpenCV's `WITH_CUDNN=ON`.

## Arch selection (CUDA cells)

`test_mo_cuda` executes device code, so the nvcc archs must cover the
attached GPU (the reference host is an RTX 5090, sm_120/Blackwell):

- **cuda12/13 cells**: no archs pinned. The vendored FindCUDA default
  (`CUDA_ARCH_NAME=Auto`) detects the attached GPU at configure time and
  emits SASS for it (sm_120 under `--gpus all` on the reference host).
- **cuda11 cells**: nvcc 11.8 tops out at sm_90, so Auto-detection on the
  Blackwell host would feed it `compute_120` and break the build. These
  images bake in `CUDA_ARCH_BIN=90 CUDA_ARCH_PTX=90`: SASS sm_90 plus
  embedded compute_90 PTX, which the driver JITs on the newer GPU so the
  device tests still run.

Note: `run_tests.sh` only applies `CUDA_ARCH_BIN` by also passing
`-DCUDA_ARCH_NAME=Manual` — the vendored FindCUDA (`cmake/Cuda.cmake`)
silently `unset`s `CUDA_ARCH_BIN` in the default Auto mode, so a bare
`-DCUDA_ARCH_BIN=...` (as the old images' `ENV` implied) was a no-op.

## Test shims

The harness papers over three upstream gaps; they are test-image shims, not
fixes:

- **CMake 4 policy floor.** The 26.04 cells ship CMake 4.x, which rejects
  this tree's `cmake_minimum_required(2.8)`. `run_tests.sh` exports
  `CMAKE_POLICY_VERSION_MINIMUM=3.5` (ignored by CMake 3.x). (OpenCV 4.14.0
  and Wt 4.10.2 both require ≥ 3.5, so their source builds are unaffected.)
- **`import imp` in the generated python wrapper.** `metaobject.py` uses
  `import imp`, removed in Python 3.12. On Python ≥ 3.12 `run_tests.sh`
  rewrites the wrapper with `importlib` before the interop test; the upstream
  template (`modules/python/metaobject.py.in.unix`) still needs that fix.
- **Bare `python` at configure time.** `cmake/metaobject_declare_plugin.cmake`
  invokes bare `python` (parse_cmake.py) and these distros only ship
  python3; each image symlinks `/usr/local/bin/python` → `python3`.

Wt is not in any Ubuntu release's apt repos (verified on focal, jammy, noble
and resolute), so every cell builds Wt 4.10.2 from source in a build stage
on the same base distro and copies the install into the final image. The
26.04 cells skip Wt (below).

## Wt on the 26.04 cells (skipped)

Wt 4.10.2 (pinned for every cell) builds against `boost::asio` and includes
`boost/asio/io_service.hpp`. Asio is still part of every Boost release
(1.90 and the current 1.92 both ship it, with change notes in the release
page), but the Asio version Ubuntu 26.04 packages dropped the
long-deprecated `io_service` API: on 26.04 `/usr/include/boost/asio.hpp` is
present while `boost/asio/io_service.hpp` is not; on 24.04's Boost 1.83 the
deprecated shim header is still installed, which is why every other cell
builds fine.

The 26.04 cells therefore skip Wt: `build.sh` passes `WITH_WT=OFF`, which
makes the Wt build stage a no-op and is exported to the test runner so it
configures with `-DWITH_WT=OFF` (the `wtgui` module is then not built). The
other ten cells keep Wt 4.10.2.

Background for a future fix: Wt ≥ 4.12 guards the include on
`BOOST_ASIO_VERSION` (modern Asio → `io_context.hpp` plus a local
`io_service` alias), and Wt ≥ 4.11 can also target standalone Asio (the
chriskohlhoff/asio project, where Asio development also continues) via
`-DWT_ASIO_IMPLEMENTATION=standalone`. Wt is actively maintained (4.14.4,
2026-10), and a `WT_VERSION=4.14.4` build was verified to compile cleanly on
26.04 against plain apt `boost::asio` — no standalone Asio download needed.
Bumping the pin for all cells is a one-line `build.sh` change plus a full
rebuild; deferred.

## Usage

```bash
./build.sh                  # build all 12 cells (each cuda cell source-builds
./build.sh cuda12-ubuntu24.04   # CUDA-enabled OpenCV inside the image — the
./build.sh ubuntu26.04 cuda13-ubuntu26.04   # slow step; build selectively)

./test.sh                   # run the suite in every built image
./test.sh ubuntu24.04       # one cell
COMPILERS="g++-13" ./test.sh cuda12-ubuntu24.04   # restrict toolchains
CUDA_ARCH_BIN=120 CUDA_ARCH_PTX=120 ./test.sh cuda11-ubuntu22.04  # override archs
```

Each image auto-detects installed compilers (`g++`, `g++-9` … `g++-16`,
`clang++`, `clang++-10` … `clang++-21`) and runs configure+build+tests once
per compiler; the image exits non-zero if any compiler fails, so `test.sh` is
CI-friendly.

`COMPILERS`, `CUDA_ARCH_BIN`, `CUDA_ARCH_PTX` and `JOBS` are passed through
only when set on the host; otherwise the image defaults apply (in particular
the cuda11 cells' baked-in `CUDA_ARCH_BIN=90`). `WT_VERSION` /
`OPENCV_VERSION` can be overridden at build time via the environment
(`WT_VERSION=4.10.3 ./build.sh …`), overriding the per-cell defaults.

Tests are run by executing the `test_mo_*` binaries in the build dir
directly. A full `ctest` run from the build dir now passes end-to-end as
well (the `ct` submodule's test entries resolve to the build-root binaries
as of ct e0ddb75); the binaries are still driven directly here because
that is the canonical CI flow and the per-binary timeout catches a hung
test instead of freezing the matrix.

## Prerequisites

- Submodules initialized **recursively** (the `ct` submodule's nested
  `googletest` provides the `gtest` targets):
  `git submodule update --init --recursive`
- For the `cuda*` cells: NVIDIA driver + nvidia-container-toolkit on the host
  (`test.sh` passes `--gpus all`). A driver current enough for the newest
  cell (e.g. driver 615 / CUDA 13.4 UMD) runs all of 11/12/13.
