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

#include <boost/fiber/barrier.hpp>
#include <boost/fiber/condition_variable.hpp>
#include <boost/fiber/fiber.hpp>
#include <boost/fiber/mutex.hpp>
#include <boost/fiber/operations.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/thread/tss.hpp>

namespace mo
{

    void AsyncStream::workerLoop(AsyncStream::Ptr_t ptr)
    {
        mo::IAsyncStream::setCurrent(ptr);
        while (ptr->m_continue)
        {
            std::function<void(IAsyncStream*)> work;
            {
                std::lock_guard<boost::fibers::mutex> lock(ptr->m_mtx);
                if (!ptr->m_work_queue.empty())
                {
                    work = std::move(std::get<0>(ptr->m_work_queue.front()));
                    ptr->m_work_queue.pop_front();
                }
            }
            if (work)
            {
                work(ptr.get());
            }
            else
            {
                boost::this_fiber::sleep_for(std::chrono::milliseconds(10));
            }
        }
        (void)ptr;
    }

    AsyncStream::AsyncStream(AllocatorPtr_t alloc)
    {
        m_thread_id = getThisThreadId();
        m_allocator = std::move(alloc);
        m_device_id = -1;
        m_continue = true;
    }

    AsyncStream::~AsyncStream()
    {
        this->AsyncStream::synchronize();
        m_continue = false;
        m_worker_fiber.join();
    }

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

    void AsyncStream::pushWork(std::function<void(IAsyncStream*)>&& work)
    {
        std::lock_guard<boost::fibers::mutex> lock(m_mtx);
        m_work_queue.push_back(std::make_tuple(std::move(work), 0));
    }

    void AsyncStream::pushEvent(std::function<void(IAsyncStream*)>&& event, const uint64_t event_id)
    {
        std::lock_guard<boost::fibers::mutex> lock(m_mtx);
        if (event_id != 0)
        {
            std::remove_if(m_work_queue.begin(),
                           m_work_queue.end(),
                           [event_id](const std::tuple<IAsyncStream::Work_f, uint64_t>& val) {
                               return std::get<1>(val) == event_id;
                           });
        }
        m_work_queue.push_back(std::make_tuple(std::move(event), event_id));
    }

    void AsyncStream::synchronize()
    {
        mo::ScopedProfile profile("mo::AsyncStream::synchronize");
        boost::fibers::barrier barrier(2);
        {
            if (m_work_queue.empty())
            {
                return;
            }
            std::lock_guard<boost::fibers::mutex> lock(m_mtx);
            m_work_queue.push_back(std::make_tuple([&barrier](IAsyncStream* strm) { barrier.wait(); }, 0));
        }
        barrier.wait();
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

    size_t AsyncStream::size() const
    {
        return m_work_queue.size();
    }

    void AsyncStream::initialize()
    {
        auto ptr = this->shared_from_this();
        m_worker_fiber = boost::fibers::fiber(&AsyncStream::workerLoop, ptr);
        m_worker_fiber.properties<FiberProperty>().setStream(ptr);
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
                auto stream = std::make_shared<AsyncStream>(Allocator::getDefault());
                stream->setName(name);
                stream->setHostPriority(thread_priority);
                return stream;
            }
        };

        CPUAsyncStreamConstructor g_ctr;
    } // namespace
} // namespace mo
