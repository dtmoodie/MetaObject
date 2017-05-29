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

thread_local std::shared_ptr<Context> thread_set_context = nullptr;

std::shared_ptr<Context> mo::Context::create(const std::string& name){
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
    std::shared_ptr<Context> output(ctx);
    if(thread_set_context == nullptr)
        thread_set_context = output;
    return output;
}

Context::Context() {
    thread_id = getThisThread();
    allocator = Allocator::getThreadSpecificAllocator();
    GpuThreadAllocatorSetter<cv::cuda::GpuMat>::Set(allocator);
    CpuThreadAllocatorSetter<cv::Mat>::Set(allocator);
}

std::shared_ptr<Context> Context::getDefaultThreadContext() {
    return thread_set_context;
}

void Context::setDefaultThreadContext(const std::shared_ptr<Context>& ctx) {
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
}

cv::cuda::Stream& Context::getStream() {
    THROW(warning) << "Not a gpu context";
    return *static_cast<cv::cuda::Stream*>(nullptr);
}
cudaStream_t Context::getCudaStream() const{
    return nullptr;
}
void Context::setStream(const cv::cuda::Stream& stream) {}
