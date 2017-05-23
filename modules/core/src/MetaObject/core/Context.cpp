#include "MetaObject/core/Context.hpp"
#include "MetaObject/logging/Log.hpp"
#include "MetaObject/thread/ThreadRegistry.hpp"
#include "MetaObject/core/detail/Allocator.hpp"
#include "MetaObject/logging/Profiling.hpp"
#include "MetaObject/core/detail/HelperMacros.hpp"
#include "MetaObject/core/detail/Allocator.hpp"
#include "MetaObject/thread/ThreadRegistry.hpp"
#include "CvContext.hpp"
#include "CudaContext.hpp"
#include "boost/lexical_cast.hpp"
#include <boost/thread/tss.hpp>
using namespace mo;
boost::thread_specific_ptr<Context> thread_specific_context;

thread_local Context* thread_set_context = nullptr;
static Context*     create(const std::string& name){
    Context* ctx = nullptr;
#ifdef HAVE_OPENCV
    ctx = new CvContext();
#else
#ifdef HAVE_CUDA
    ctx = new CudaContext();
#else
    ctx = new Context();
#endif
#endif
    ctx->setName(name);
    if(!thread_set_context)
        thread_set_context = ctx;
    return ctx;
}

Context::Context() {
    thread_id = getThisThread();
    allocator = Allocator::getThreadSpecificAllocator();
    GpuThreadAllocatorSetter<cv::cuda::GpuMat>::Set(allocator);
    CpuThreadAllocatorSetter<cv::Mat>::Set(allocator);
}

Context* Context::getDefaultThreadContext() {
    if(thread_set_context)
        return thread_set_context;

    if(thread_specific_context.get() == nullptr) {
        thread_specific_context.reset(new Context());
    }
    return thread_specific_context.get();
}

void Context::setDefaultThreadContext(Context*  ctx) {
    thread_set_context = ctx;
}

void Context::setName(const std::string& name) {
    if(name.size()) {
        allocator->setName(name);
        mo::setThreadName(name.c_str());
    } else {
        allocator->setName("Thread " + boost::lexical_cast<std::string>(thread_id) + " allocator");
    }
    this->name = name;
}

Context::~Context() {
    //stream.waitForCompletion();
}

cv::cuda::Stream& Context::getStream() {
    THROW(warning) << "Not a gpu context";
    return *static_cast<cv::cuda::Stream*>(nullptr);
}

void Context::setStream(const cv::cuda::Stream& stream) {}
