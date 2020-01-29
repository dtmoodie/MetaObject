sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub

CUDA_REPO_PKG=cuda-repo-ubuntu1604_9.2.148-1_amd64.deb
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/$CUDA_REPO_PKG
dpkg -i $CUDA_REPO_PKG
rm $CUDA_REPO_PKG

apt-get -y update

# install packages
CUDA_PKG_VERSION=9-2
apt-get install -y --no-install-recommends --allow-downgrades\
    cuda-core-$CUDA_PKG_VERSION \
    cuda-cudart-dev-$CUDA_PKG_VERSION \
    cuda-cublas-dev-$CUDA_PKG_VERSION \
    cuda-curand-dev-$CUDA_PKG_VERSION \
    cuda-compiler-$CUDA_PKG_VERSION \
    cuda-nvcc-$CUDA_PKG_VERSION \
    gcc-5=5.4.0-6ubuntu1~16.04.11 \
    gcc-5-base:amd64=5.4.0-6ubuntu1~16.04.11 \
    libgcc-5-dev:amd64=5.4.0-6ubuntu1~16.04.11 \
    cpp-5=5.4.0-6ubuntu1~16.04.11 \
    libasan2=5.4.0-6ubuntu1~16.04.11 \
    libmpx0=5.4.0-6ubuntu1~16.04.11 \
    python-numpy
# manually create CUDA symlink
ln -s /usr/local/cuda-* /usr/local/cuda
