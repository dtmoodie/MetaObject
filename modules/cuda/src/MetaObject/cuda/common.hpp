#ifndef MO_CUDA_COMMON_HPP
#define MO_CUDA_COMMON_HPP
#include <MetaObject/logging/logging.hpp>

#include <cuda_runtime_api.h>
#include <ostream>

namespace fmt
{
    std::string to_string(cudaError err);
}

namespace std
{
    ostream& operator<<(ostream& os, cudaError);
}

#endif // MO_CUDA_COMMON_HPP
