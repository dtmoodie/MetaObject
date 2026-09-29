#ifndef MO_CUDA_TEST_DEVICE_FUNCS_HPP
#define MO_CUDA_TEST_DEVICE_FUNCS_HPP
#include <cstdint>
struct CUstream_st;
using cudaStream_t = CUstream_st*;
namespace cuda_tests
{
    // Keep `stream` busy for a bounded time (~1 us per iteration) with a
    // memory-less kernel, so that work enqueued behind it cannot complete
    // while it runs. See device_funcs.cu for why this replaces the old
    // spin-on-a-device-flag wait/set kernels.
    void hold(uint32_t iterations, cudaStream_t stream);

    void set(float* data, float val, int N, cudaStream_t stream);

    void fill(float* begin, float* end, float val, cudaStream_t stream);

    void multiply(const uint32_t N, float* data, const float scale, cudaStream_t stream);

    void square(float* data, uint32_t N, cudaStream_t stream);
}// cuda_tests

#endif // MO_CUDA_TEST_DEVICE_FUNCS_HPP
