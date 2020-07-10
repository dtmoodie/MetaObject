#include "MetaObject/core/AsyncStream.hpp"
#include "MetaObject/core/SystemTable.hpp"
#include "MetaObject/core/detail/Allocator.hpp"
#include "MetaObject/core/detail/HelperMacros.hpp"

#include "MetaObject/logging/logging.hpp"
#include "MetaObject/logging/profiling.hpp"
#include "MetaObject/thread/FiberProperties.hpp"
#include <MetaObject/thread/FiberScheduler.hpp>
#include <MetaObject/thread/Thread.hpp>
#include <MetaObject/thread/ThreadInfo.hpp>

#include "AsyncStreamConstructor.hpp"
#include "AsyncStreamFactory.hpp"

#include <boost/fiber/fiber.hpp>
#include <boost/fiber/operations.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/thread/tss.hpp>

namespace mo
{
    AsyncStream::AsyncStream(AllocatorPtr_t alloc)
    {
        m_thread_id = getThisThreadId();
        m_allocator = std::move(alloc);
        m_device_id = -1;
    }

    AsyncStream::~AsyncStream() = default;

    void AsyncStream::setName(const std::string& name)
    {
        if (!name.empty())
        {
            if (m_allocator)
            {
                m_allocator->setName(name);
            }
            mo::setThisThreadName(name.c_str());
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
        boost::fibers::fiber fiber(std::move(work));
        auto& prop = fiber.properties<FiberProperty>();
        prop.setPriority(priority == NONE ? m_host_priority : priority);
        prop.setStream(this->shared_from_this());
        fiber.detach();
    }

    void
    AsyncStream::pushEvent(std::function<void(void)>&& event, const uint64_t event_id, const PriorityLevels priority)
    {
        boost::fibers::fiber fiber([event]() {
            if (boost::this_fiber::properties<FiberProperty>().enabled())
            {
                event();
            }
        });

        auto& prop = fiber.properties<FiberProperty>();
        prop.setAll(priority == NONE ? m_host_priority : priority, event_id, false);
        prop.setStream(this->shared_from_this());
        fiber.detach();
    }

    void AsyncStream::synchronize()
    {
        const auto sched = PriorityScheduler::current();
        auto size = sched->size();
        while (size != 1)
        {
            boost::this_fiber::yield();
            size = sched->size();
        }
    }

    void AsyncStream::hostToHost(ct::TArrayView<void> dst, ct::TArrayView<const void> src)
    {
        const size_t src_size = src.size();
        const size_t dst_size = dst.size();
        MO_ASSERT(src_size == dst_size);
        if (src_size == 0)
        {
            return;
        }
        void* dst_data = dst.data();
        const void* src_data = src.data();
        std::memcpy(dst_data, src_data, src_size);
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

    PriorityLevels AsyncStream::hostPriority() const
    {
        return m_host_priority;
    }

    uint64_t AsyncStream::threadId() const
    {
        return m_thread_id;
    }

    bool AsyncStream::isDeviceStream() const
    {
        return false;
    }

    uint64_t AsyncStream::streamId() const
    {
        return m_stream_id;
    }

    namespace
    {
        struct CPUAsyncStreamConstructor : public AsyncStreamConstructor
        {
            CPUAsyncStreamConstructor()
            {
                SystemTable::dispatchToSystemTable(
                    [this](SystemTable* table) { AsyncStreamFactory::instance(table)->registerConstructor(this); });
            }

            uint32_t priority(int32_t) override
            {
                return 1U;
            }

            Ptr_t create(const std::string& name, int32_t, PriorityLevels, PriorityLevels thread_priority) override
            {
                auto stream = std::make_shared<AsyncStream>();
                stream->setName(name);
                stream->setHostPriority(thread_priority);
                return stream;
            }
        };

        CPUAsyncStreamConstructor g_ctr;
    } // namespace
} // namespace mo
