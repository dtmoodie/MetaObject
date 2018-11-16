#include "MetaObject/core/Context.hpp"
#include "MetaObject/core/SystemTable.hpp"
#include "MetaObject/core/detail/Allocator.hpp"
#include "MetaObject/core/detail/Allocator.hpp"
#include "MetaObject/core/detail/HelperMacros.hpp"

#include "MetaObject/core/detail/allocator_policies/Combined.hpp"
#include "MetaObject/core/detail/allocator_policies/Lock.hpp"
#include "MetaObject/core/detail/allocator_policies/Pool.hpp"
#include "MetaObject/core/detail/allocator_policies/RefCount.hpp"
#include "MetaObject/core/detail/allocator_policies/Stack.hpp"
#include "MetaObject/core/detail/allocator_policies/Usage.hpp"

#include "MetaObject/logging/logging.hpp"
#include "MetaObject/logging/profiling.hpp"
#include <MetaObject/thread/Thread.hpp>
#include <MetaObject/thread/ThreadRegistry.hpp>

#include "ContextConstructor.hpp"

#include "boost/lexical_cast.hpp"
#include <boost/thread/tss.hpp>
using namespace mo;

static thread_local IContext* current_context = nullptr;

IContext* IContext::getCurrent()
{
    return current_context;
}

void IContext::setCurrent(IContext* ctx)
{
    current_context = ctx;
}

IContext::~IContext()
{
}

ContextFactory* ContextFactory::instance()
{
    return singleton<ContextFactory>();
}

void ContextFactory::registerConstructor(ContextConstructor* ctr)
{
    m_ctrs.push_back(ctr);
}

ContextFactory::Ptr
ContextFactory::create(const std::string& name, int device_id, int device_priority, int thread_priority)
{
    ContextConstructor* best_ctr = nullptr;
    int highest_priority = 0;
    for (auto ctr : m_ctrs)
    {
        auto p = ctr->priority(device_id, device_priority, thread_priority);
        if (p > highest_priority)
        {
            highest_priority = p;
            best_ctr = ctr;
        }
    }
    if (best_ctr)
    {
        return best_ctr->create(name, device_id, device_priority, thread_priority);
    }
}

Context::Ptr Context::create(const std::string& name, int device_id, int cuda_priority, int thread_priority)
{
    std::shared_ptr<Context> ctx = ContextFactory::instance()->create(name, device_id, cuda_priority, thread_priority);
    setCurrent(ctx.get());
    return ctx;
}

using Allocator_t = mo::CombinedPolicy<LockPolicy<PoolPolicy<CPU>>, LockPolicy<StackPolicy<CPU>>>;

Context::Context()
{
    m_thread_id = getThisThread();
    // TODO allocator rework

    m_allocator = Allocator_t::create();

    m_device_id = -1;
}

Context::Context(TypeInfo type)
{
    m_thread_id = getThisThread();
    m_allocator = Allocator_t::create();
    m_device_id = -1;
    m_context_type = type;
}

void Context::setName(const std::string& name)
{
    if (name.size())
    {
        m_allocator->setName(name);
        mo::setThreadName(name.c_str());
    }
    else
    {
        m_allocator->setName("Thread " + boost::lexical_cast<std::string>(m_thread_id) + " allocator");
    }
    m_name = name;
}

void Context::pushWork(std::function<void(void)>&& work)
{
    MO_ASSERT(m_work_handler);
    m_work_handler(std::move(work));
}

std::string Context::name() const
{
    return m_name;
}

Context::~Context()
{
#ifdef _DEBUG
    MO_LOG(info, "Context [{}] destroyed", m_name);
#endif
}

size_t Context::processId() const
{
    return m_process_id;
}

std::shared_ptr<Allocator> Context::allocator() const
{
    return m_allocator;
}

size_t Context::threadId() const
{
    return m_thread_id;
}

bool Context::isDeviceContext() const
{
    return m_device_id >= 0;
}

void Context::setDeviceId(int id)
{
    m_device_id = id;
}

void Context::setAllocator(const std::shared_ptr<Allocator>& allocator)
{
    m_allocator = allocator;
}

void Context::setWorkHandler(std::function<void(std::function<void(void)>)>&& work_handler)
{
    m_work_handler = std::move(work_handler);
}

struct CPUContextConstructor : public ContextConstructor
{
};
