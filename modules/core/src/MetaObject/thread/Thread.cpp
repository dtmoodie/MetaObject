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

#include <cstdio>
#include <cstdlib>
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
        auto table = SystemTable::instance();
        MO_ASSERT(table);
        std::shared_ptr<mo::ThreadPool> pool = table->getSingleton<mo::ThreadPool>();
        boost::fibers::use_scheduling_algorithm<mo::PriorityScheduler>(pool);
    }

    void initThread(SystemTable& table)
    {
        std::shared_ptr<mo::ThreadPool> pool = table.getSingleton<mo::ThreadPool>();
        boost::fibers::use_scheduling_algorithm<mo::PriorityScheduler>(pool);
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
        m_pool = pool;
        m_thread = boost::thread(&Thread::main, this);
    }

    Thread::~Thread()
    {
        PROFILE_FUNCTION

        // Wait (bounded) for the worker to install its scheduler and
        // publish the wakeup cv. Normal operation reaches that point in
        // microseconds; the bound only covers a worker that was starved or
        // failed before publishing.
        {
            std::unique_lock<std::mutex> lk(m_ready_mtx);
            m_ready_cv.wait_for(lk, std::chrono::seconds(10), [this]() { return m_scheduler_ready; });
        }

        // Signal shutdown. The flag is set under the same (fiber) mutex the
        // worker's wait() holds-then-releases, and the worker's wait uses a
        // predicate on the flag, so a shutdown signal arriving before the
        // wait starts cannot be lost. The notify wakes the worker's parked
        // fiber; boost.fiber re-schedules that fiber on the WORKER's own
        // scheduler (context::schedule -> schedule_from_remote), so the
        // worker's main() epilogue runs on the worker thread and the thread
        // actually exits.
        //
        // Note: the wakeup-cv pointer published during startup is
        // deliberately NOT dereferenced here. The worker's scheduler is
        // destroyed at its thread exit, which can race with this thread;
        // the cv notify above is sufficient (it already wakes the
        // scheduler).
        {
            Lock_t lock(m_mtx);
            m_shutdown.store(true, std::memory_order_release);
        }
        m_cv.notify_all();

        if (m_thread.joinable() && !m_thread.timed_join(boost::posix_time::time_duration(0, 0, 30)))
        {
            // The worker did not stop. Destroying the members below (m_cv,
            // m_stream, m_mtx, ...) while the worker thread may still be
            // using them would be use-after-free; fail loudly instead of
            // silently corrupting the heap.
            std::fprintf(stderr, "mo::Thread: worker thread did not stop within 30s; aborting to avoid use-after-free\n");
            std::abort();
        }
        // Join succeeded: the worker thread has exited, its scheduler has
        // retired itself from the pool and been destroyed, and the exit
        // callback has fired. Members may now be destroyed safely.
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
        auto stream = mo::AsyncStreamFactory::instance()->create();
        {
            // TODO fiber
            Lock_t lock(m_mtx);
            m_stream = stream;
            lock.unlock();
            m_cv.notify_all();
        }
        boost::fibers::use_scheduling_algorithm<PriorityScheduler>(m_pool->shared_from_this(), &m_scheduler_wakeup_cv);
        {
            // Publish the wakeup-cv pointer (written by the scheduler
            // constructor above) to ~Thread under the handshake lock.
            std::lock_guard<std::mutex> lk(m_ready_mtx);
            m_scheduler_ready = true;
        }
        m_ready_cv.notify_all();

        ThreadExit on_exit{[this]() {
            if (m_on_exit)
            {
                m_on_exit();
            }
        }};

        // Park on THIS thread until shutdown.
        //
        // The wait is predicate-based so a shutdown signal that arrives
        // before the wait starts (Thread destroyed right after creation)
        // is not lost. It is fiber-aware (mo's ConditionVariable wraps
        // boost::fibers::condition_variable_any): when ~Thread() on
        // another thread calls notify, boost.fiber routes the notification
        // to THIS thread's own scheduler (context::schedule ->
        // schedule_from_remote), so the epilogue below runs on this
        // thread, this thread actually exits, and ~Thread's join succeeds
        // before its members are destroyed.
        Mutex_t::Lock_t lock(m_mtx);
        m_cv.wait(lock, [this]() { return m_shutdown.load(std::memory_order_acquire); });

        stream.reset();
        m_stream.reset();

        // Retire this thread's scheduler from the pool before it is
        // destroyed at thread exit, so other schedulers' pick_next()/
        // steal() can no longer reach it through a pool entry.
        if (nullptr != m_pool)
        {
            m_pool->retireScheduler(PriorityScheduler::current());
        }
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