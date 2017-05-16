#include <MetaObject/thread/Cuda.hpp>

thread_local bool is_cuda_thread = false;
namespace mo
{
void SetCudaThread()
{
    is_cuda_thread = true;
}
bool IsCudaThread()
{
    return is_cuda_thread;
}
}
