# MetaObject test image: CUDA 12.8 on Ubuntu 24.04 — the full-coverage image.
#   nvcc 12.8 with GCC 13 (default) / 12, Clang 18, Boost 1.83 (incl.
#   fiber/context), OpenCV 4.14 built with CUDA (contrib, cuDNN/cuBLAS/
#   cuFFT), Wt 4.10.2 (source), Qt 5.15,
#   Python 3.12 + Boost.Python + numpy, CMake 3.28.
#
# CUDA 12.8 (rather than the 12.5 used by ct's cuda12 image) is the first
# toolkit that emits SASS for sm_120 (Blackwell, e.g. RTX 5090); the test
# binaries execute device code, so a native arch is wanted.
#
# OpenCV is built from source with CUDA because:
#   - modules/metaparams unconditionally includes OpenCV headers
#   - modules/python unconditionally includes <opencv2/core/cuda.hpp>,
#     which only exists in CUDA-enabled OpenCV
# apt OpenCV is therefore not installed (it would also force WITH_CUDA=OFF
# via cmake/dependencies/opencv.cmake).
#
# Wt is not in any current Ubuntu release's apt repos (checked focal,
# jammy and noble), so it is built from source in a stage on the same base
# distro and copied into the final image.
#
# The devel base image ships an apt repository for CUDA; it is removed
# because apt would otherwise see it and can pull cuda-toolkit-*-config
# packages that repoint /usr/local/cuda at a toolkit that is not installed.
# The full 12.8 toolkit is already present in the image.
ARG WT_VERSION=4.10.2
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS wt-build
ARG WT_VERSION
ENV DEBIAN_FRONTEND=noninteractive
RUN rm -f /etc/apt/sources.list.d/cuda.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        wget \
        ca-certificates \
        libpng-dev \
        libjpeg-dev \
        libboost-filesystem-dev \
        libboost-thread-dev \
        libboost-program-options-dev \
    && cd /tmp \
    && wget -q "https://github.com/emweb/wt/archive/refs/tags/${WT_VERSION}.tar.gz" \
    && tar xf "${WT_VERSION}.tar.gz" \
    && cd "wt-${WT_VERSION}" \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    && make -C build -j"$(nproc)" \
    && make -C build install \
    && ldconfig \
    && rm -rf "/tmp/wt-${WT_VERSION}" "/tmp/${WT_VERSION}.tar.gz"

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
# Test configuration: python module ON (possible here: CUDA-enabled OpenCV),
# and CUDA archs for the vendored FindCUDA (sm_120 = Blackwell, sm_90 = Hopper).
ENV WITH_PYTHON=ON
ENV CUDA_ARCH_BIN=120;90

RUN rm -f /etc/apt/sources.list.d/cuda.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        ca-certificates \
        g++-12 \
        clang \
        python3 python3-dev python3-numpy \
        libboost1.83-dev \
        libboost-fiber1.83-dev \
        libboost-context1.83-dev \
        libboost-filesystem-dev \
        libboost-thread-dev \
        libboost-program-options-dev \
        qtbase5-dev \
        libjpeg-dev libpng-dev libtiff-dev \
    && BOOST_PY_DEV=$(apt-cache search '^libboost-python[0-9].*-dev$' | awk '{print $1}' | sort -V | tail -1) \
    && echo "installing $BOOST_PY_DEV" \
    && apt-get install -y --no-install-recommends "$BOOST_PY_DEV" \
    && rm -rf /var/lib/apt/lists/*

# Wt from the build stage (headers + shared libs into /usr/local).
COPY --from=wt-build /usr/local/lib /usr/local/lib
COPY --from=wt-build /usr/local/include /usr/local/include
RUN ldconfig

# Pin /usr/local/cuda to the installed toolkit so the vendored FindCUDA
# finds nvcc no matter what the alternatives symlinks say.
RUN ln -sfn /usr/local/cuda-12.8 /usr/local/cuda \
    && /usr/local/cuda/bin/nvcc --version

# CUDA-enabled OpenCV (core + imgproc + cudaarithm, matching what
# metaobject/tests consume). CUDA builds also need the header-only 'cudev'
# module from the matching opencv_contrib release (core requires it when
# WITH_CUDA=ON), so both trees are fetched; BUILD_LIST keeps the rest of
# contrib out.
#
# Note the arch order is 90;120, not the natural descending 120;90:
# OpenCV's "deprecated compute capability" guard is a regex test
# (`" ${CUDA_ARCH_BIN} ${CUDA_ARCH_PTX}" MATCHES " 1.0"`), and regex `.`
# matches any character — " 120" matches " 1.0", so a list that STARTS
# with 120 (Blackwell) dies with "1.0 compute capability is not
# supported". The bug (still present in 4.11.0) only fires for the first
# list element, so 90;120 works. Do not reorder.
ARG OPENCV_VERSION=4.14.0
RUN cd /tmp \
    && wget -q "https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz" \
    && wget -q -O "opencv_contrib-${OPENCV_VERSION}.tar.gz" \
        "https://github.com/opencv/opencv_contrib/archive/refs/tags/${OPENCV_VERSION}.tar.gz" \
    && tar xf "${OPENCV_VERSION}.tar.gz" \
    && tar xf "opencv_contrib-${OPENCV_VERSION}.tar.gz" \
    && cd "opencv-${OPENCV_VERSION}" \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_LIST="core;imgproc;cudaarithm;cudev" \
        -DOPENCV_EXTRA_MODULES_PATH="/tmp/opencv_contrib-${OPENCV_VERSION}/modules" \
        -DWITH_CUDA=ON \
        -DWITH_CUDNN=ON -DWITH_CUBLAS=ON -DWITH_CUFFT=ON \
        -DCUDA_ARCH_BIN="90;120" \
        -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
        -DCMAKE_VERBOSE_MAKEFILE=on \
        -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF \
    && make -C build -j$(nproc) \
    && make -C build install \
    && ldconfig \
    && rm -rf "/tmp/opencv-${OPENCV_VERSION}" "/tmp/opencv_contrib-${OPENCV_VERSION}" \
               "/tmp/${OPENCV_VERSION}.tar.gz" "/tmp/opencv_contrib-${OPENCV_VERSION}.tar.gz"

# cmake/metaobject_declare_plugin.cmake invokes bare `python` at configure
# time (it runs parse_cmake.py to generate plugin_config2.cpp); noble only
# ships python3. parse_cmake.py is Python 3 compatible, so a symlink is
# enough. (Placed after the OpenCV build so it does not invalidate that
# cached layer.)
RUN ln -sf /usr/bin/python3 /usr/local/bin/python

COPY run_tests.sh /opt/mo-test/run_tests.sh
RUN chmod +x /opt/mo-test/run_tests.sh

ENTRYPOINT ["/opt/mo-test/run_tests.sh"]
