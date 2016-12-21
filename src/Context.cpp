#include "MetaObject/Context.hpp"
#include "MetaObject/Thread/ThreadRegistry.hpp"
#include "MetaObject/Detail/Allocator.hpp"
#include "MetaObject/Logging/Profiling.hpp"
#include "boost/lexical_cast.hpp"
#include <boost/thread/tss.hpp>

using namespace mo;
boost::thread_specific_ptr<Context> thread_specific_context;

thread_local Context* thread_set_context = nullptr;

Context* Context::GetDefaultThreadContext()
{
    if(thread_set_context)
        return thread_set_context;

    if(thread_specific_context.get() == nullptr)
    {
        thread_specific_context.reset(new Context());
    }
    return thread_specific_context.get();
}

void Context::SetDefaultThreadContext(Context*  ctx)
{
    thread_set_context = ctx;
}

Context::Context(const std::string& name)
{
    thread_id = GetThisThread();
    allocator = Allocator::GetThreadSpecificAllocator();
    cv::Mat::setDefaultAllocator(allocator, true);
    cv::cuda::GpuMat::setDefaultAllocator(allocator, true);
    if(name.size())
    {
        allocator->SetName(name);
        mo::SetThreadName(name.c_str());

    }else
    {
        allocator->SetName("Thread " + boost::lexical_cast<std::string>(thread_id) + " allocator");
    }
    this->name = name;
}
cv::cuda::Stream &Context::GetStream()
{
    return stream;
}

void Context::SetStream(cv::cuda::Stream stream)
{
    mo::SetStreamName(name.c_str(), stream);
}
