#ifndef MO_CUDA_COMMON_HPP
#define MO_CUDA_COMMON_HPP
#include <cuda_runtime_api.h>
#include <spdlog/fmt/bundled/ostream.h>

namespace mo
{

}

namespace std
{
    ostream& operator <<(ostream& os, cudaError);
}

#endif //MO_CUDA_COMMON_HPP
