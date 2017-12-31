#include "MetaObject/core/Context.hpp"
#include "CudaContext.hpp"
#include "CvContext.hpp"
#include "MetaObject/core/detail/Allocator.hpp"
#include "MetaObject/core/detail/Allocator.hpp"
#include "MetaObject/core/detail/HelperMacros.hpp"
#include "MetaObject/logging/logging.hpp"
#include "MetaObject/logging/profiling.hpp"
#include "MetaObject/thread/ThreadRegistry.hpp"
#include "MetaObject/thread/ThreadRegistry.hpp"
#include "boost/lexical_cast.hpp"
#include <boost/thread/tss.hpp>
using namespace mo;

thread_local std::shared_ptr<Context> thread_set_context = nullptr;

std::shared_ptr<Context> mo::Context::create(const std::string& name, int priority)
{
    Context* ctx = nullptr;
#ifdef HAVE_OPENCV
    ctx = new CvContext(priority);
#else
#ifdef HAVE_CUDA
    ctx = new CudaContext(priority);
#else
    ctx = new Context();
#endif
#endif
    ctx->setName(name);
    std::shared_ptr<Context> output(ctx);
    if (thread_set_context == nullptr)
        thread_set_context = output;
    return output;
}

Context::Context()
{
    thread_id = getThisThread();
    allocator = Allocator::getThreadSpecificAllocator();
}

std::shared_ptr<Context> Context::getDefaultThreadContext()
{
    return thread_set_context;
}

void Context::setDefaultThreadContext(const std::shared_ptr<Context>& ctx)
{
    thread_set_context = ctx;
}

void Context::setName(const std::string& name)
{
    if (name.size())
    {
        allocator->setName(name);
        mo::setThreadName(name.c_str());
    }
    else
    {
        allocator->setName("Thread " + boost::lexical_cast<std::string>(thread_id) + " allocator");
    }
    this->name = name;
}

Context::~Context()
{
#ifdef _DEBUG
    MO_LOG(info) << "Context [" << name << "] destroyed";
#endif
}

cv::cuda::Stream& Context::getStream()
{
    THROW(warning) << "Not a gpu context";
    return *static_cast<cv::cuda::Stream*>(nullptr);
}
cudaStream_t Context::getCudaStream() const
{
    return nullptr;
}
void Context::setStream(const cv::cuda::Stream& stream)
{
    (void)stream;
}
void Context::setStream(cudaStream_t stream)
{
    (void)stream;
}
