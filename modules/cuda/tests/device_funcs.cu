#include "device_funcs.hpp"
#include <cuda_runtime_api.h>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

namespace cuda_tests
{
    uint32_t divUp(const uint32_t total, const uint32_t threads)
    {
        return (total + threads - 1) / threads;
    }

    __global__ void waitKernel(volatile const bool* quit)
    {
        while (!*quit)
        {
        }
    }

    __global__ void setKernel(volatile bool* quit)
    {
        *quit = true;
    }

    __global__ void mulKernel(const uint32_t N, float* data, const float scale)
    {
        const uint32_t thread_index = threadIdx.x + blockIdx.x * blockDim.x;
        if (thread_index < N)
        {
            data[thread_index] *= scale;
        }
    }

    __global__ void setKernel(float* data, float val, int N)
    {
        auto tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid < N)
        {
            data[tid] = val;
        }
    }

    __global__ void squareKernel(float* data, int N)
    {
        auto tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid < N)
        {
            data[tid] = data[tid] * data[tid];
        }
    }

    __global__ void sigKernel(volatile bool* quit)
    {
        *quit = true;
    }

    void wait(volatile const bool* quit, cudaStream_t stream)
    {
        waitKernel<<<1, 1, 0, stream>>>(quit);
    }

    void set(volatile bool* quit, cudaStream_t stream)
    {
        setKernel<<<1, 1, 0, stream>>>(quit);
    }

    void set(float* data, float val, int N, cudaStream_t stream)
    {
        const size_t num_threads = 256;
        const auto blocks = divUp(N, num_threads);
        setKernel<<<blocks, num_threads, 0, stream>>>(data, val, N);
    }

    void fill(float* begin, float* end, float val, cudaStream_t stream)
    {
        thrust::fill(thrust::device_ptr<float>(begin), thrust::device_ptr<float>(end), val);
    }

    void multiply(const uint32_t N, float* data, const float scale, cudaStream_t stream)
    {
        const uint32_t num_threads = 256;
        const uint32_t num_blocks = divUp(N, num_threads);

        mulKernel<<<num_blocks, num_threads, 0, stream>>>(N, data, scale);
    }

    void square(float* data, uint32_t N, cudaStream_t stream)
    {
        const uint32_t num_threads = 256;
        const uint32_t num_blocks = divUp(N, num_threads);
        squareKernel<<<num_blocks, num_threads, 0, stream>>>(data, N);
    }
}
