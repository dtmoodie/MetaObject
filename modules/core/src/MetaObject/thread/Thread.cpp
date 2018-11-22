#include "MetaObject/thread/Thread.hpp"
#include "FiberScheduler.hpp"
#include "MetaObject/core/AsyncStream.hpp"
#include "MetaObject/core/detail/Allocator.hpp"
#include "MetaObject/core/detail/Time.hpp"
#include "MetaObject/logging/profiling.hpp"
#include "MetaObject/signals/TSignalRelay.hpp"
#include "MetaObject/signals/TSlot.hpp"
#include "MetaObject/thread/ThreadRegistry.hpp"
#include "MetaObject/thread/boost_thread.hpp"

#include <boost/fiber/all.hpp>

#include <future>

using namespace mo;

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
    m_thread = boost::thread(&Thread::main, this);
    m_pool = pool;
}

Thread::~Thread()
{
    PROFILE_FUNCTION

    MO_LOG(info, "Waiting for {} to join", m_name);
    m_thread.interrupt();
    if (!m_thread.timed_join(boost::posix_time::time_duration(0, 0, 10)))
    {
        MO_LOG(warn, "{} did not join after waiting 10 seconds");
    }
    MO_LOG(info, "{} shutdown complete");
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

    boost::fibers::use_scheduling_algorithm<PriorityScheduler>();

    ThreadExit on_exit{[this]() {
        if (m_on_exit)
        {
            m_on_exit();
        }
    }};

    std::set<uint64_t> event_ids;

    std::vector<std::function<void(void)>> work_to_process;
    work_to_process.resize(100);

    while (!boost::this_thread::interruption_requested())
    {
        try
        {
            // TODO fiber rework
        }
        catch (...)
        {
        }
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
