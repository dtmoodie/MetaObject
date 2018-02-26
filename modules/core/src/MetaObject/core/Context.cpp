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

thread_local Context* current_context = nullptr;

Context* Context::getCurrent()
{
    return current_context;
}

void Context::setCurrent(Context* ctx)
{
    current_context = ctx;
}


std::shared_ptr<Context> mo::Context::create(const std::string& name, int priority)
{
    std::shared_ptr<Context> ctx;
#ifdef HAVE_OPENCV
    ctx = std::make_shared<CvContext>(priority);
#else
#ifdef HAVE_CUDA
    ctx = std::make_shared<CudaContext>(priority);
#else
    ctx = std::make_shared<Context>();
#endif
#endif
    ctx->setName(name);
    setCurrent(ctx.get());
    return ctx;
}

Context::Context()
{
    thread_id = getThisThread();
    allocator = Allocator::getDefaultAllocator();
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

void Context::setStream(const cv::cuda::Stream& /*stream*/)
{
}

void Context::setStream(cudaStream_t /*stream*/)
{
}
