#include "common.hpp"
#include "MemoryBlock.hpp"

#include <MetaObject/logging/logging.hpp>

#include <cuda_runtime_api.h>

namespace mo
{

void* CUDA::allocate(const uint64_t size)
{

    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    MO_ASSERT_FMT(err == cudaSuccess, "Unable to allocate {} on device due to {}", size, err);
    return ptr;
}

void CUDA::deallocate(void* data)
{
    cudaError_t err = cudaFree(data);
    MO_ASSERT_FMT(err == cudaSuccess, "Unable to free device memory due to {}", err);
}


void* HOST::allocate(const uint64_t size)
{

    void* ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, size);
    MO_ASSERT_FMT(err == cudaSuccess, "Unable to allocate {} on device due to {}", size, err);
}

void HOST::deallocate(void* data)
{
    cudaError_t err = cudaFreeHost(data);
    MO_ASSERT_FMT(err == cudaSuccess, "Unable to free host pinned memory due to {}", err);
}

template class Memory<CUDA>;
template class MemoryBlock<CUDA>;
template class Memory<HOST>;
template class MemoryBlock<HOST>;

}
