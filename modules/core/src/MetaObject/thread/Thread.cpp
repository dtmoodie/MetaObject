#include "MetaObject/thread/Thread.hpp"
#include "FiberScheduler.hpp"
#include "ThreadPool.hpp"

#include "MetaObject/core/AsyncStream.hpp"
#include "MetaObject/core/AsyncStreamFactory.hpp"

#include "MetaObject/core/detail/Allocator.hpp"
#include "MetaObject/core/detail/Time.hpp"
#include "MetaObject/logging/profiling.hpp"
#include "MetaObject/signals/TSignalRelay.hpp"
#include "MetaObject/signals/TSlot.hpp"
#include "MetaObject/thread/ThreadRegistry.hpp"
#include <MetaObject/thread/ThreadInfo.hpp>

#include <boost/fiber/condition_variable.hpp>
#include <boost/fiber/operations.hpp>
#include <boost/fiber/recursive_timed_mutex.hpp>

#include <future>

namespace mo
{

    void sleepFor(std::chrono::nanoseconds ns)
    {
        boost::this_fiber::sleep_for(ns);
    }

    void sleepUntil(std::chrono::high_resolution_clock::time_point tp)
    {
        boost::this_fiber::sleep_until(tp);
    }

    void yield()
    {
        boost::this_fiber::yield();
    }

    void initThread()
    {
        thread_local bool initialized = false;
        if (!initialized)
        {
            auto table = SystemTable::instance();
            MO_ASSERT(table);
            initThread(*table);
            initialized = true;
        }
    }

    void initThread(SystemTable& table)
    {
        thread_local bool initialized = false;
        if (!initialized)
        {
            std::shared_ptr<mo::ThreadPool> pool = table.getSingleton<mo::ThreadPool>();
            if (mo::PriorityScheduler::current() == nullptr)
            {
                boost::fibers::use_scheduling_algorithm<mo::PriorityScheduler>(pool);
            }
        }
    }

    void Thread::setExitCallback(std::function<void(void)>&& f)
    {
        Lock_t lock(m_mtx);
        m_on_exit = std::move(f);
    }

    void Thread::setName(const std::string& name)
    {
        m_name = name;
        setThreadName(m_thread, name);
    }

    ThreadPool* Thread::pool() const
    {
        return m_pool;
    }

    IAsyncStreamPtr_t Thread::asyncStream(const Duration timeout) const
    {
        Lock_t lock(m_mtx);
        if (!m_stream)
        {
            m_cv.wait_for(lock, timeout);
        }
        return m_stream;
    }

    Thread::Thread(ThreadPool* pool)
    {
        if (pool == nullptr)
        {
            pool = SystemTable::instance()->getSingleton<ThreadPool>().get();
        }
        m_pool = pool;
        m_continue = true;
        m_thread = boost::thread(&Thread::main, this);
    }

    Thread::~Thread()
    {
        PROFILE_FUNCTION

        m_continue = false;
        m_cv.notify_all();
        m_scheduler_wakeup_cv->notify_all();

        m_thread.try_join_for(boost::chrono::seconds(10));
    }

    struct ThreadExit
    {
        ~ThreadExit()
        {
            on_exit();
        }

        std::function<void(void)> on_exit;
    };

    void Thread::main()
    {
        mo::setThisThreadName("Worker");
        boost::fibers::use_scheduling_algorithm<PriorityScheduler>(m_pool->shared_from_this(), &m_scheduler_wakeup_cv);
        auto stream = mo::AsyncStreamFactory::instance()->create(
            "WorkerStream", 0, PriorityLevels::MEDIUM, PriorityLevels::MEDIUM);
        {
            // TODO fiber
            Lock_t lock(m_mtx);
            m_stream = stream;
            lock.unlock();
            m_cv.notify_all();
        }

        ThreadExit on_exit{[this]() {
            if (m_on_exit)
            {
                m_on_exit();
            }
        }};

        while (m_continue)
        {
            Mutex_t::Lock_t lock(m_mtx);
            m_cv.wait_for(lock, std::chrono::milliseconds(1));
        }
        stream.reset();
        m_stream.reset();
    }

    size_t Thread::threadId() const
    {
        return getThreadId(m_thread);
    }

    const std::string& Thread::threadName() const
    {
        return m_name;
    }

    bool Thread::isOnThread() const
    {
        return threadId() == getThisThread();
    }

} // namespace mo
