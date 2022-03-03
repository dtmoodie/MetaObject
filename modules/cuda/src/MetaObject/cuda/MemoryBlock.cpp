#include "MemoryBlock.hpp"
#include "common.hpp"

#include <MetaObject/logging/logging.hpp>

#include <cuda_runtime_api.h>

namespace mo
{
    namespace cuda
    {
        uint8_t* CUDA::allocate(const uint64_t size, const int32_t)
        {
            void* ptr = nullptr;
            cudaError_t err = cudaMalloc(&ptr, size);
            MO_ASSERT_FMT(err == cudaSuccess, "Unable to allocate {} on device due to {}", size, err);
            return static_cast<uint8_t*>(ptr);
        }

        void CUDA::deallocate(void* data, const uint64_t)
        {
            cudaError_t err = cudaFree(data);
            if(err != cudaErrorCudartUnloading)
            {
                MO_ASSERT_FMT(err == cudaSuccess, "Unable to free device memory due to {}", err);
            }
        }

        uint8_t* HOST::allocate(const uint64_t size, const int32_t)
        {

            void* ptr = nullptr;
            cudaError_t err = cudaMallocHost(&ptr, size);
            MO_ASSERT_FMT(err == cudaSuccess, "Unable to allocate {} on device due to {}", size, err);
            return static_cast<uint8_t*>(ptr);
        }

        void HOST::deallocate(void* data, const uint64_t)
        {
            cudaError_t err = cudaFreeHost(data);
            if(err != cudaErrorCudartUnloading)
            {
                MO_ASSERT_FMT(err == cudaSuccess, "Unable to free host pinned memory due to {}", err);
            }
        }
    } // namespace cuda
    template struct MemoryBlock<cuda::CUDA>;
    template struct MemoryBlock<cuda::HOST>;
} // namespace mo
