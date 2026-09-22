# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

MetaObject is a reflection layer on top of **RCC** (Runtime Compiled C++): classes marked with macros and inheriting `mo::IMetaObject` become runtime-introspectable — parameters, signals, and slots can be read/changed at runtime, and objects can be hot-recompiled from source (see `examples/console_recompile`).

This is the `dtmoodie/metaobject` fork, actively under development. Recent direction (see `git log`): params-module refactor (`Input`/`Output` → `ISubscriber`/`IPublisher`, `ICoordinateSystem` removed in favor of `SourceId`), the `fiber_refactor` (boost.fiber isolated from nvcc), and **gcc 4.8 / nvcc 6.5 compatibility** — keep new code working on those old toolchains (no C++14+ features in shared headers, reflection info kept out of templates on old gcc).

The parent workspace (`../CLAUDE.md`) describes the full stack (`rcc` → `metaobject` → `aquila` → `eagleyeye`); where the two disagree, this file wins for this repo.

## Build

```bash
git submodule update --init --recursive   # required first (see gotcha below)
mkdir build && cd build
cmake .. && make -j$(nproc)
```

- All executables and shared libs land in the **build root** (`CMAKE_RUNTIME_OUTPUT_DIRECTORY` = `CMAKE_BINARY_DIR`).
- Requires Boost with `system thread fiber filesystem chrono context` (CI builds boost 1.69 from source via `scripts/build_boost.sh`).
- CMake options: `BUILD_TESTS`, `BUILD_DEPENDENCIES`, `BUILD_EXAMPLES` (all ON); `WITH_QT`, `WITH_PYTHON`, `WITH_OPENCV`, `WITH_CUDA` (ON), `WITH_WT` (OFF). `BUILD_DEPENDENCIES=OFF` switches to `find_package(RCC)`.
- The `python` module only builds when **both** Python and OpenCV are found (it binds cv:: types via numpy).
- `cmake_minimum_required(2.8)` with vendored `FindCUDA`/`FindBoost`/`FindWt`/`FindNumPy` in `cmake/` — old-toolchain support is deliberate.
- C++ standard is 11, bumped to 14 on GNU > 4.9 (set in `cmake/metaobject_declare_module.cmake`).

## Tests

Per-module googletest binaries, all in the build root. The `gtest`/`gtest_main` targets come from the `dependencies/ct` subdirectory (ct adds `dependencies/googletest`):

```bash
./test_mo_core
./test_mo_object --gtest_filter='object.*:signal.*:slot.*'
./test_mo_params
./test_mo_serialization
./test_mo_runtime_reflection
./test_mo_types
./test_mo_cuda
```

CI (`.travis.yml` / `.appveyor.yml`) runs the binaries directly rather than `ctest` — that's the canonical flow. `ctest` also works, but note `modules/params/CMakeLists.txt` registers its test as `add_test(NAME test_mo_cuda COMMAND test_mo_params)` (copy-paste bug; name collides with the cuda test). `test_mo_params` is built with AddressSanitizer on GCC. The python ctest entry is doubly broken: `add_test(NAME test_mo_python COMMAND python2 test_mo_python.py)` invokes `python2` (absent on modern distros) against a file name that does not exist (the configured copy is `metaobject_python_test.py`).

`modules/object/tests/plugin/` builds `mo_objectplugin` — a shared object of test classes loaded via the RCC plugin machinery, which is what `test_mo_object` exercises for runtime reflection/recompile.

## Docker test environment

`docker/` holds standalone test images mirroring `ct/docker/` (same configs: `ubuntu20.04`, `ubuntu22.04`, `ubuntu24.04`, `cuda12`). The source tree is mounted read-only at `/src`; each image runs configure+build+tests once per installed compiler and exits non-zero on any failure.

```bash
cd docker
./build.sh                  # build all four images (cuda12 takes longest)
./build.sh cuda12           # or one image
./test.sh                   # run the suite in every built image
COMPILERS="g++-13" ./test.sh cuda12
```

- `test.sh` forwards `COMPILERS`, `CUDA_ARCH_BIN` and `JOBS` to the container only when they are set on the host (a bare `-e VAR` for an unset variable would blank the image's own `ENV`, e.g. cuda12's `CUDA_ARCH_BIN=120;90`); `JOBS` caps `make` parallelism inside the image.
- `cuda12` is the only full-coverage image: it's the only one with `WITH_PYTHON=ON`, because the python module unconditionally includes `<opencv2/core/cuda.hpp>` (see gotchas) and therefore needs CUDA-enabled OpenCV, which it builds from source (4.14, with contrib) inside the image. It also runs with `--gpus all` and passes `CUDA_ARCH_BIN=120;90` (sm_120 = Blackwell/RTX 5090).
- The three ubuntu images keep `WITH_PYTHON=OFF` and use apt OpenCV (CPU); they cover the compiler/toolchain spread (GCC 9/10 + Clang 10, GCC 11/12 + Clang 14, GCC 13/12 + Clang 18).
- The generated `metaobject.py` wrapper uses `import imp`, removed in Python 3.12 — `run_tests.sh` rewrites it with `importlib` before the interop test (harness shim; `modules/python/metaobject.py.in.unix` still needs the real fix).

## Architecture

Each module is `modules/<name>/` with sources under `src/MetaObject/<name>/` (include as `<MetaObject/<name>/...>`), built as shared lib `metaobject_<name>` via the `metaobject_declare_module()` CMake function (`cmake/metaobject_declare_module.cmake`) — that function owns include dirs, exports, install and target folders; new modules get a `modules/` subdir + `add_subdirectory` in the top CMakeLists. Modules (in build order): `core`, `params`, `types`, `cuda`, `metaparams`, `runtime_reflection`, `serialization`, `object`, `python`, `gui` (builds `metaobject_qtgui` and/or `metaobject_wtgui`).

**The user-facing API is a macro DSL** (see `modules/object/tests/plugin/TestObjects.hpp` for canonical examples):

```cpp
struct mo_EXPORTS MyNode : public MetaObject
{
    MO_DERIVE(MyNode, MetaObject)      // base: MO_DERIVE(T, Base1, Base2, ...)
    PARAM(int, threshold, 0)           // TParam — read/write at runtime
    INPUT(cv::Mat, image)              // TSubscriber (was "Input")
    OUTPUT(float, score)               // TPublisher (was "Output")
    MO_SIGNAL(void, on_ready, int)
    MO_SLOT(void, on_ready, int)
    MO_END;                            // emits ct reflection info + RCC registration
};
MO_REGISTER_OBJECT(MyNode);            // outside the class; wraps RCC REGISTERCLASS
```

- **`core`**: `SystemTable` (global singleton registry of types/objects), `TypeTable`, `AsyncStream` — the fiber-based (boost.fiber) async execution primitive; `IAsyncStream` indirection exists specifically so **nvcc never sees boost/fiber headers** (it fails to compile on many platforms — keep `.cu`/cuda sources free of them, cf. `modules/cuda/src/MetaObject/cuda/Event.hpp`). Also: logging (`MO_LOG`, `CompileLogger`), signals (`ISignal`/`ISlot`/`RelayManager`/`Connection`, `ArgumentPack`), `thread/FiberScheduler`, and `detail/allocator` policies (Pool/Stack/RefCount/OpenCV device mem).
- **`object`**: `IMetaObject` (virtual interface: setup, introspection via `getParamInfo/getSignalInfo/getSlotInfo`, `connectByName`, `save/load`) and `MetaObjectFactory` — per-process singleton that creates objects by name, loads `.so` plugins, and drives hot recompilation: `registerTranslationUnit()` (must be called in each TU; deliberately inlined) → `checkCompile()` → `swapObjects()`.
- **`params`**: the data plane. Hierarchy: `IParam` → `IPublisher`/`ISubscriber` → `TParam`/`TPublisher`/`TSubscriber` (see `modules/params/README.md`). Plus `buffers/` (Queue, Map, Circular, NN stream, Stream buffers), `Header` (per-update metadata), `ParamServer`/`NamedParam` (named-parameter registry), `IControlParam`, `AccessToken`.
- **`runtime_reflection`**: runtime type introspection built on the `ct` header-only library. Use `ct::GetName` for type names and the `ct` overload of `std::get<TYPE>`; reflection info is kept out of templates for gcc 4.8.
- **`types`**: cv:: types + `SmallVec` (single general-purpose small vector class).
- **`serialization`**: cereal-based `save`/`load`. Note `serializeParams`/`serializeConnections` on `IMetaObject` are only used for runtime recompilation state, not disk serialization.
- **`python`**: Boost.Python bindings (2 or 3); generates a `metaobject.py` wrapper via `configure_file` of `metaobject.py.in.{unix,win32}`.
- **`gui`**: parameter editors only (Qt5 widgets / Wt), not a general app GUI.
- `metaobject_config.hpp` is generated (`metaobject_config.hpp.in`) into `build/include` — `MO_HAVE_*` flags control optional-feature code paths.

## Gotchas

- **Optional dependencies are coupled in ways the CMake gates don't express.** `modules/metaparams` unconditionally includes OpenCV headers, and `modules/python/.../PythonSetup.cpp` unconditionally includes `<MetaObject/cuda/opencv.hpp>` → `<opencv2/core/cuda.hpp>`, which only exists in CUDA-enabled OpenCV. So `WITH_PYTHON=ON` with CPU OpenCV configures fine and then fails to compile. Worse, `cmake/dependencies/opencv.cmake` force-sets `WITH_CUDA` from `OpenCV_CUDA_VERSION` (plain `set`, overriding `-DWITH_CUDA`), so CPU OpenCV silently disables the cuda module too.
- **Submodules resolve against the GitHub origin, not the local workspace.** `.gitmodules` uses relative URLs (`../rcc`, `../ct`, `../cereal`, `../spdlog`), which resolve against `origin` = `github.com/dtmoodie/...`. `git submodule update --init` therefore clones dtmoodie's forks from GitHub — the sibling `rcc/`/`ct/` dirs in this workspace are *not* what gets pulled. To build against the local checkouts, override with `git config submodule.dependencies/<name>.url <abs path>` before init. A fresh clone has all four submodule dirs empty until init.
- Style drift from refactors to respect: include guards (not `pragma once`), references over pointers-to-required-values, `MO_`-prefixed macros over the old bare names.
- `modules/core` defines `BOOST_NO_AUTO_PTR` publicly — don't use `boost::auto_ptr`.
- The `ct` submodule's own repo contains WIP Python-testing files in the workspace checkout; ignore them.
