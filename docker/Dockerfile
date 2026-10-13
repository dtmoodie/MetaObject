# MetaObject test images — one parameterized Dockerfile for the whole support
# matrix (Ubuntu 20.04/22.04/24.04/26.04 x CUDA 11/12/13). build.sh passes the
# per-cell arguments; see README.md for the matrix and per-cell notes.
#
# Build args (per cell, see docker/build.sh):
#   BASE_IMAGE     ubuntu:<ver> or nvidia/cuda:<ver>-devel-ubuntu<ver>
#   CUDA_MAJOR     empty (CPU-only cell) | 11 | 12 | 13
#   WITH_PYTHON    ON for cuda cells (possible there: CUDA-enabled OpenCV), OFF
#                  for CPU cells (the python module unconditionally includes
#                  <opencv2/core/cuda.hpp>, which only exists in CUDA-enabled
#                  OpenCV)
#   APP_PKGS       space-separated per-cell apt list (compilers, boost, python,
#                  qtbase5-dev; libopencv-dev on CPU cells)
#   CUDA_ARCH_BIN/CUDA_ARCH_PTX  arch defaults baked in as ENV; empty keeps the
#                  vendored FindCUDA's Auto mode (SASS for the attached GPU).
#                  The cuda11 cells pin 90/90 (nvcc 11.8 cannot target sm_120;
#                  the embedded PTX is JIT'd by the driver on newer GPUs).
#   WT_VERSION     Wt is not in any Ubuntu release's apt repos (verified on
#                  focal, jammy, noble and resolute), so it is built from
#                  source in a stage on the same base distro and copied into
#                  the final image.
#   WITH_WT        ON (default): build Wt in the wt-build stage. OFF: the
#                  stage does nothing (leaving empty /usr/local dirs so the
#                  COPY --from below is a no-op) and the value is exported
#                  so run_tests.sh configures with -DWITH_WT=OFF. The 26.04
#                  cells use OFF: Wt 4.10.2 includes boost/asio/io_service.hpp,
#                  which the Asio version shipped with their Boost 1.90 no
#                  longer provides (see README).
#   OPENCV_VERSION CUDA cells build OpenCV from source with CUDA (the python
#                  module needs <opencv2/core/cuda.hpp> and WITH_CUDA is
#                  force-derived from OpenCV_CUDA_VERSION, so apt/CPU OpenCV
#                  would silently disable the cuda module). The matching
#                  opencv_contrib release is required: OpenCV FATALs if the
#                  'cudev' module is missing when WITH_CUDA=ON, and the CUDA
#                  13.2 cudev fix (contrib #4097) only exists from 4.14.0.
ARG BASE_IMAGE=ubuntu:24.04
ARG CUDA_MAJOR=
ARG WT_VERSION=4.10.2
ARG WITH_WT=ON
ARG OPENCV_VERSION=4.14.0

FROM ${BASE_IMAGE} AS wt-build
ARG WT_VERSION
ARG WITH_WT
ARG CUDA_MAJOR
ENV DEBIAN_FRONTEND=noninteractive
# The nvidia/cuda devel base images ship an apt repository for CUDA; it is
# removed so apt cannot pull cuda-toolkit-*-config packages that repoint
# /usr/local/cuda at a toolkit that is not installed.
# WITH_WT=OFF (the 26.04 cells) skips the build entirely; the empty dirs
# keep the COPY --from in the final stage valid.
RUN if [ "$WITH_WT" = "OFF" ]; then \
      echo "Wt build skipped (WITH_WT=OFF)" \
      && mkdir -p /usr/local/lib /usr/local/include; \
    else \
      if [ -n "$CUDA_MAJOR" ]; then rm -f /etc/apt/sources.list.d/cuda.list; fi \
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
      && rm -rf "/tmp/wt-${WT_VERSION}" "/tmp/${WT_VERSION}.tar.gz"; \
    fi

FROM ${BASE_IMAGE}
ARG CUDA_MAJOR
ARG WITH_PYTHON=OFF
ARG WITH_WT=ON
ARG APP_PKGS=""
ARG CUDA_ARCH_BIN=
ARG CUDA_ARCH_PTX=
ARG OPENCV_VERSION

ENV DEBIAN_FRONTEND=noninteractive
ENV WITH_PYTHON=${WITH_PYTHON}
ENV WITH_WT=${WITH_WT}
ENV CUDA_ARCH_BIN=${CUDA_ARCH_BIN}
ENV CUDA_ARCH_PTX=${CUDA_ARCH_PTX}

# python3 is installed in every cell: cmake/metaobject_declare_plugin.cmake
# invokes bare `python` at configure time (parse_cmake.py) and only ships as
# python3 on these distros, so the symlink below needs it.
RUN if [ -n "$CUDA_MAJOR" ]; then rm -f /etc/apt/sources.list.d/cuda.list; fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        $APP_PKGS \
    && if [ -n "$CUDA_MAJOR" ]; then \
         BOOST_PY_DEV=$(apt-cache search '^libboost-python[0-9].*-dev$' | awk '{print $1}' | sort -V | tail -1) \
         && [ -n "$BOOST_PY_DEV" ] \
         && echo "installing $BOOST_PY_DEV" \
         && apt-get install -y --no-install-recommends "$BOOST_PY_DEV"; \
       fi \
    && rm -rf /var/lib/apt/lists/*

# Wt from the build stage (headers + shared libs into /usr/local); a no-op
# for the WITH_WT=OFF cells.
COPY --from=wt-build /usr/local/lib /usr/local/lib
COPY --from=wt-build /usr/local/include /usr/local/include
RUN ldconfig

# Pin /usr/local/cuda to the installed toolkit so the vendored FindCUDA finds
# nvcc no matter what the alternatives symlinks say.
RUN if [ -n "$CUDA_MAJOR" ]; then \
      CUDA_DIR=$(ls -d /usr/local/cuda-*/ | sed 's:/$::' | head -n1) \
      && ln -sfn "$CUDA_DIR" /usr/local/cuda \
      && "$CUDA_DIR/bin/nvcc" --version; fi

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
# list element, so 90;120 works. Do not reorder. (CUDA 11 cells use 90;86:
# 120 does not exist in the 11.x toolkit.)
RUN if [ -n "$CUDA_MAJOR" ]; then \
      if [ "$CUDA_MAJOR" = "11" ]; then OPENCV_ARCHS="90;86"; else OPENCV_ARCHS="90;120"; fi \
      && cd /tmp \
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
          -DCUDA_ARCH_BIN="$OPENCV_ARCHS" \
          -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
          -DCMAKE_VERBOSE_MAKEFILE=on \
          -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF \
      && make -C build -j"$(nproc)" \
      && make -C build install \
      && ldconfig \
      && rm -rf "/tmp/opencv-${OPENCV_VERSION}" "/tmp/opencv_contrib-${OPENCV_VERSION}" \
                 "/tmp/${OPENCV_VERSION}.tar.gz" "/tmp/opencv_contrib-${OPENCV_VERSION}.tar.gz"; fi

# cmake/metaobject_declare_plugin.cmake invokes bare `python` at configure
# time (it runs parse_cmake.py to generate plugin_config2.cpp); these distros
# only ship python3. parse_cmake.py is Python 3 compatible, so a symlink is
# enough. (Placed after the OpenCV build so it does not invalidate that
# cached layer.)
RUN ln -sf /usr/bin/python3 /usr/local/bin/python

COPY run_tests.sh /opt/mo-test/run_tests.sh
RUN chmod +x /opt/mo-test/run_tests.sh

ENTRYPOINT ["/opt/mo-test/run_tests.sh"]
