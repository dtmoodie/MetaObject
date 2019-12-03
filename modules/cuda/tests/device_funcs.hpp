#ifndef MO_CUDA_TEST_DEVICE_FUNCS_HPP
#define MO_CUDA_TEST_DEVICE_FUNCS_HPP
#include <cstdint>
struct CUstream_st;
using cudaStream_t = CUstream_st*;
namespace cuda_tests
{
    void wait(volatile const bool* quit, cudaStream_t stream);

    void set(volatile bool* quit, cudaStream_t stream);

    void set(float* data, float val, int N, cudaStream_t stream);

    void fill(float* begin, float* end, float val, cudaStream_t stream);

    void multiply(const uint32_t N, float* data, const float scale, cudaStream_t stream);

    void square(float* data, uint32_t N, cudaStream_t stream);
}// cuda_tests

#endif // MO_CUDA_TEST_DEVICE_FUNCS_HPP
