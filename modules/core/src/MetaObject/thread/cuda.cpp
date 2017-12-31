#include <MetaObject/thread/cuda.hpp>

thread_local bool is_cuda_thread = false;
namespace mo
{
    void setCudaThread() { is_cuda_thread = true; }
    bool isCudaThread() { return is_cuda_thread; }
}
