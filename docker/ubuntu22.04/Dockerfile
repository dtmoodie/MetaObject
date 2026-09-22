# MetaObject test image: Ubuntu 22.04
#   GCC 11 (default) / 12, Clang 14, Boost 1.74 (incl. fiber/context),
#   OpenCV 4.5 (CPU), Wt 4.10.2 (source), Qt 5.15, CMake 3.22.
#
# WITH_PYTHON stays OFF: the python module unconditionally includes
# <opencv2/core/cuda.hpp> (modules/python/.../PythonSetup.cpp), so it only
# builds against CUDA-enabled OpenCV — that is what the cuda12 image covers.
#
# Wt is not in any current Ubuntu release's apt repos (checked focal,
# jammy and noble), so it is built from source in a stage on the same base
# distro and copied into the final image.
ARG WT_VERSION=4.10.2
FROM ubuntu:22.04 AS wt-build
ARG WT_VERSION
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
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

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        g++-12 \
        clang \
        libboost1.74-dev \
        libboost-fiber1.74-dev \
        libboost-context1.74-dev \
        libboost-filesystem-dev \
        libboost-thread-dev \
        libboost-program-options-dev \
        libopencv-dev \
        qtbase5-dev \
    && rm -rf /var/lib/apt/lists/*

# Wt from the build stage (headers + shared libs into /usr/local).
COPY --from=wt-build /usr/local/lib /usr/local/lib
COPY --from=wt-build /usr/local/include /usr/local/include
RUN ldconfig

# cmake/metaobject_declare_plugin.cmake invokes bare `python` at configure
# time (it runs parse_cmake.py to generate plugin_config2.cpp); these distros
# only ship python3. parse_cmake.py is Python 3 compatible, so a symlink
# is enough.
RUN ln -sf /usr/bin/python3 /usr/local/bin/python

COPY run_tests.sh /opt/mo-test/run_tests.sh
RUN chmod +x /opt/mo-test/run_tests.sh

ENTRYPOINT ["/opt/mo-test/run_tests.sh"]
