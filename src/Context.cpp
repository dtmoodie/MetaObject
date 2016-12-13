#include "MetaObject/Context.hpp"
#include "MetaObject/Thread/ThreadRegistry.hpp"
#include "MetaObject/Detail/Allocator.hpp"
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

Context::Context()
{
    thread_id = GetThisThread();
    allocator = Allocator::GetThreadSpecificAllocator();
}
