#include "MetaObject/core/AsyncStream.hpp"

#include "MetaObject/core/SystemTable.hpp"
#include "MetaObject/core/detail/Allocator.hpp"
#include "MetaObject/core/detail/Allocator.hpp"
#include "MetaObject/core/detail/HelperMacros.hpp"

#include "MetaObject/logging/logging.hpp"
#include "MetaObject/logging/profiling.hpp"
#include "MetaObject/thread/FiberProperties.hpp"
#include <MetaObject/thread/Thread.hpp>
#include <MetaObject/thread/ThreadRegistry.hpp>

#include "AsyncStreamConstructor.hpp"

#include "boost/lexical_cast.hpp"
#include <boost/thread/tss.hpp>
using namespace mo;

IAsyncStream::~IAsyncStream()
{
}

AsyncStreamFactory* AsyncStreamFactory::instance(SystemTable* table)
{
    return singleton<AsyncStreamFactory>(table);
}

AsyncStreamFactory* AsyncStreamFactory::instance()
{
    return singleton<AsyncStreamFactory>();
}

void AsyncStreamFactory::registerConstructor(AsyncStreamConstructor* ctr)
{
    m_ctrs.push_back(ctr);
}

AsyncStreamFactory::Ptr_t AsyncStreamFactory::create(const std::string& name,
                                                     const int32_t device_id,
                                                     const PriorityLevels device_priority,
                                                     const PriorityLevels thread_priority)
{
    AsyncStreamConstructor* best_ctr = nullptr;
    uint32_t highest_priority = 0;
    for (auto ctr : m_ctrs)
    {
        auto p = ctr->priority(device_id);
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
    return {};
}

AsyncStream::AsyncStream()
{
    m_thread_id = getThisThread();
    m_allocator = PerModuleInterface::GetInstance()->GetSystemTable()->createAllocator();

    m_device_id = -1;
}

AsyncStream::AsyncStream(TypeInfo type)
{
    m_thread_id = getThisThread();
    m_allocator = PerModuleInterface::GetInstance()->GetSystemTable()->createAllocator();
    m_device_id = -1;
    m_type = type;
}

AsyncStream::~AsyncStream()
{
#ifdef _DEBUG
    MO_LOG(info, "Context [{}] destroyed", m_name);
#endif
}

void AsyncStream::setName(const std::string& name)
{
    if (name.size())
    {
        if (m_allocator)
        {
            m_allocator->setName(name);
        }
        mo::setThreadName(name.c_str());
    }
    else
    {
        if (m_allocator)
        {
            m_allocator->setName("Thread " + boost::lexical_cast<std::string>(m_thread_id) + " allocator");
        }
    }
    m_name = name;
}

void AsyncStream::pushWork(std::function<void(void)>&& work, const PriorityLevels priority)
{
    boost::fibers::fiber fiber(work);
    FiberProperty& prop = fiber.properties<FiberProperty>();
    prop.setPriority(priority == NONE ? m_host_priority : priority);
    fiber.detach();
}

void AsyncStream::pushEvent(std::function<void(void)>&& event, const uint64_t event_id, const PriorityLevels priority)
{
    boost::fibers::fiber fiber(event);
    FiberProperty& prop = fiber.properties<FiberProperty>();
    prop.setAll(priority == NONE ? m_host_priority : priority, event_id, false);
    fiber.detach();
}

std::string AsyncStream::name() const
{
    return m_name;
}

uint64_t AsyncStream::processId() const
{
    return m_process_id;
}

std::shared_ptr<Allocator> AsyncStream::hostAllocator() const
{
    return m_allocator;
}

void AsyncStream::setHostPriority(const PriorityLevels p)
{
    m_host_priority = p;
}

uint64_t AsyncStream::threadId() const
{
    return m_thread_id;
}

bool AsyncStream::isDeviceContext() const
{
    return false;
}

TypeInfo AsyncStream::interface() const
{
    return m_type;
}

uint64_t AsyncStream::streamId() const
{
    return m_stream_id;
}

struct CPUAsyncStreamConstructor : public AsyncStreamConstructor
{
    CPUAsyncStreamConstructor()
    {
        SystemTable::staticDispatchToSystemTable(
            [this](SystemTable* table) { AsyncStreamFactory::instance(table)->registerConstructor(this); });
    }

    uint32_t priority(const int32_t) override
    {
        return 1U;
    }

    Ptr_t create(const std::string& name, const int32_t, PriorityLevels, PriorityLevels thread_priority) override
    {
        auto stream = std::make_shared<AsyncStream>();
        stream->setName(name);
        stream->setHostPriority(thread_priority);
        return stream;
    }
};

CPUAsyncStreamConstructor g_ctr;
