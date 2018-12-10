#ifndef MO_CUDA_COMMON_HPP
#define MO_CUDA_COMMON_HPP
#include <MetaObject/logging/logging.hpp>

#include <cuda_runtime_api.h>
#include <ostream>

#define CUDA_ERROR_CHECK(EXPR)                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret = (EXPR);                                                                                             \
        if (ret != cudaSuccess)                                                                                        \
            THROW(error, #EXPR " failed! Due to: {}", ret);                                                            \
    } while (0)

namespace fmt
{
    std::string to_string(const cudaError err);
}

namespace std
{
    ostream& operator<<(ostream& os, cudaError);
}

#endif // MO_CUDA_COMMON_HPP
