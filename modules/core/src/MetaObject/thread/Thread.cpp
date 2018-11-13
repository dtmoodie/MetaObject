#include "MetaObject/thread/Thread.hpp"
#include "MetaObject/core/Context.hpp"
#include "MetaObject/core/detail/Allocator.hpp"
#include "MetaObject/core/detail/Time.hpp"
#include "MetaObject/logging/profiling.hpp"
#include "MetaObject/signals/TSignalRelay.hpp"
#include "MetaObject/signals/TSlot.hpp"
#include "MetaObject/thread/ThreadRegistry.hpp"
#include "MetaObject/thread/boost_thread.hpp"
#include <opencv2/core.hpp>
#ifdef HAVE_CUDA
#include <cuda_runtime_api.h>
#endif
#include <future>

using namespace mo;

EventToken::EventToken(std::function<void(void)>&& event, const uint64_t id)
    : event(std::move(event))
    , event_id(id)
{
}

void Thread::pushEventQueue(EventToken&& event)
{
    m_event_queue.enqueue(std::move(event));
    m_cv.notify_all();
}

// Work can be stolen and can exist on any thread
void Thread::pushWork(std::function<void(void)>&& f)
{
    m_work_queue.enqueue(f);
    m_cv.notify_all();
}

void Thread::setExitCallback(std::function<void(void)>&& f)
{
    Lock lock(m_mtx);
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

ContextPtr_t Thread::context(const Duration timeout)
{
    Lock lock(m_mtx);
    if (!m_ctx)
    {
        m_cv.wait_for(lock, boost::chrono::nanoseconds(timeout.count()));
    }
    return m_ctx;
}

Thread::Thread(ThreadPool* pool)
{
    m_thread = boost::thread(&Thread::main, this);
    m_pool = pool;
}

Thread::~Thread()
{
    PROFILE_FUNCTION

    MO_LOG(info) << "Waiting for " << m_name << " to join";
    m_thread.interrupt();
    if (!m_thread.timed_join(boost::posix_time::time_duration(0, 0, 10)))
    {
        MO_LOG(warning) << m_name << " did not join after waiting 10 seconds";
    }
    MO_LOG(info) << m_name << " shutdown complete";
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
    auto ctx = mo::Context::create();
    {
        ctx->setEventHandle([this](EventToken&& event) { this->pushEventQueue(std::move(event)); });
        ctx->setWorkHandler([this](std::function<void(void)>&& work) { this->pushWork(std::move(work)); });
        Lock lock(m_mtx);
        m_ctx = ctx;
        lock.unlock();
        m_cv.notify_all();
    }

    ThreadExit on_exit{[this]() {
        if (m_on_exit)
        {
            m_on_exit();
        }
    }};

    std::vector<EventToken> events_to_process;

    events_to_process.resize(100);
    std::set<uint64_t> event_ids;

    std::vector<std::function<void(void)>> work_to_process;
    work_to_process.resize(100);

    while (!boost::this_thread::interruption_requested())
    {
        try
        {
            const size_t num_events = m_event_queue.try_dequeue_bulk(events_to_process.begin(), 100);

            for (size_t i = 0; i < num_events; ++i)
            {
                if (events_to_process[i].event_id != 0)
                {
                    event_ids.insert(events_to_process[i].event_id);
                }
            }

            for (size_t i = 0; i < num_events; ++i)
            {
                bool process = true;
                const auto id = events_to_process[i].event_id;
                if (id != 0)
                {
                    auto itr = event_ids.find(id);
                    if (itr == event_ids.end())
                    {
                        process = false;
                    }
                    else
                    {
                        event_ids.erase(itr);
                    }
                }
                else
                {
                    process = true;
                }
                if (process)
                {
                    events_to_process[i].event();
                }
            }
            const size_t num_work = m_work_queue.try_dequeue_bulk(work_to_process.begin(), 100);
            for (size_t i = 0; i < num_work; ++i)
            {
                work_to_process[i]();
            }
        }
        catch (...)
        {
        }
    }
    ctx.reset();
    m_ctx.reset();
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
