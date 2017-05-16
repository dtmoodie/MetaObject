#include "MetaObject/core/Context.hpp"
#include "MetaObject/thread/ThreadRegistry.hpp"
#include "MetaObject/core/detail/Allocator.hpp"
#include "MetaObject/logging/Profiling.hpp"
#include "MetaObject/core/detail/HelperMacros.hpp"
#include "MetaObject/core/detail/Allocator.hpp"
#include "MetaObject/thread/ThreadRegistry.hpp"

#include "boost/lexical_cast.hpp"
#include <boost/thread/tss.hpp>
using namespace mo;
boost::thread_specific_ptr<Context> thread_specific_context;

thread_local Context* thread_set_context = nullptr;

Context::Context(const std::string& name) {
    thread_id = GetThisThread();
    allocator = Allocator::GetThreadSpecificAllocator();
    GpuThreadAllocatorSetter<cv::cuda::GpuMat>::Set(allocator);
    CpuThreadAllocatorSetter<cv::Mat>::Set(allocator);
    if(name.size())
        setName(name);
    if(!thread_set_context)
        thread_set_context = this;
}

Context* Context::GetDefaultThreadContext() {
    if(thread_set_context)
        return thread_set_context;

    if(thread_specific_context.get() == nullptr) {
        thread_specific_context.reset(new Context());
    }
    return thread_specific_context.get();
}

void Context::SetDefaultThreadContext(Context*  ctx) {
    thread_set_context = ctx;
}

void Context::setName(const std::string& name) {
    if(name.size()) {
        allocator->setName(name);
        mo::SetThreadName(name.c_str());

    } else {
        allocator->setName("Thread " + boost::lexical_cast<std::string>(thread_id) + " allocator");
    }
    this->name = name;
}

Context::~Context() {
    //stream.waitForCompletion();
}

cv::cuda::Stream &Context::GetStream() {
    return stream;
}

void Context::SetStream(cv::cuda::Stream stream) {
    mo::SetStreamName(name.c_str(), stream);
}
