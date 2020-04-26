#include "common.hpp"

namespace mo
{
}

#define PRINT_CUDA_ERR(ERR)                                                                                            \
    case ERR:                                                                                                          \
        return #ERR

namespace fmt
{
    std::string to_string(const cudaError err)
    {
        return cudaGetErrorString(err);
    }
} // namespace fmt

namespace std
{
    ostream& operator<<(ostream& os, cudaError err)
    {
        os << fmt::to_string(err);
        return os;
    }
} // namespace std
