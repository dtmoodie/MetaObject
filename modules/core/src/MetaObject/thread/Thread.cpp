#include "MetaObject/thread/Thread.hpp"
#include "MetaObject/core/Context.hpp"
#include "MetaObject/core/detail/Allocator.hpp"
#include "MetaObject/core/detail/Time.hpp"
#include "MetaObject/logging/profiling.hpp"
#include "MetaObject/signals/TSignalRelay.hpp"
#include "MetaObject/signals/TSlot.hpp"
#include "MetaObject/thread/ThreadRegistry.hpp"
#include "MetaObject/thread/boost_thread.hpp"
#include <future>

using namespace mo;

void Thread::pushEventQueue(const std::function<void(void)>& f)
{
    _event_queue.enqueue(f);
    _cv.notify_all();
}
// Work can be stolen and can exist on any thread
void Thread::pushWork(const std::function<void(void)>& f)
{
    _work_queue.enqueue(f);
    _cv.notify_all();
}

void Thread::start()
{
    _run = true;
    _cv.notify_all();
}

void Thread::stop()
{
    if (_run == false && _paused == true)
        return;
    _run = false;
    _thread.interrupt();
    int wait_count = 0;
    while (!_paused)
    {
        boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
        ++wait_count;
        if (wait_count % 1000 == 0)
            MO_LOG(warning) << "Waited " << wait_count / 1000 << " second for " << this->_name << " to stop";
    }
    //_thread.join();
    MO_LOG(info) << _name << " has stopped";
}

void Thread::setExitCallback(const std::function<void(void)>& f)
{
    _on_exit = f;
}

void Thread::setStartCallback(const std::function<void(void)>& f)
{
    _on_start = f;
}

void Thread::setName(const std::string& name)
{
    this->_name = name;
    setThreadName(_thread, name);
}

std::shared_ptr<Connection> Thread::setInnerLoop(TSlot<int(void)>* slot)

{
    if (_run && mo::getThisThread() != getId())
    {

        std::promise<std::shared_ptr<Connection>> promise;
        std::future<std::shared_ptr<Connection>> future = promise.get_future();
        this->_event_queue.enqueue([slot, &promise, this]() { promise.set_value(slot->connect(_inner_loop)); });
        return future.get();
    }
    else
    {
        return slot->connect(_inner_loop);
    }
}

ThreadPool* Thread::getPool() const
{
    return _pool;
}

ContextPtr_t Thread::getContext()
{
    boost::unique_lock<boost::recursive_timed_mutex> lock(_mtx);
    if (!_ctx)
    {
        _cv.wait_for(lock, boost::chrono::seconds(5));
    }
    return _ctx;
}

Thread::Thread()
{
    _pool = nullptr;
    _inner_loop.reset(new mo::TSignalRelay<int(void)>());
    _quit = false;
    _thread = boost::thread(&Thread::main, this);
    _paused = false;
}

Thread::Thread(ThreadPool* pool)
{
    _inner_loop.reset(new mo::TSignalRelay<int(void)>());
    _pool = pool;
    _quit = false;
    _thread = boost::thread(&Thread::main, this);
}

Thread::~Thread()
{
    PROFILE_FUNCTION
    _quit = true;
    _run = false;
    MO_LOG(info) << "Waiting for " << this->_name << " to join";
    _thread.interrupt();
    if (!_thread.timed_join(boost::posix_time::time_duration(0, 0, 10)))
    {
        MO_LOG(warning) << this->_name << " did not join after waiting 10 seconds";
    }
    MO_LOG(info) << this->_name << " shutdown complete";
}

struct mo::Thread::ThreadSanitizer
{
    ThreadSanitizer(volatile bool& paused_flag, boost::condition_variable_any& cv, mo::Thread& thread)
        : _paused_flag(paused_flag), _cv(cv), m_thread(thread)
    {
    }

    ~ThreadSanitizer()
    {
        MO_LOG(info) << m_thread._name << " exiting";
        m_thread.getContext()->getStream().waitForCompletion();
        mo::ThreadSpecificQueue::run();
        _paused_flag = true;
        _cv.notify_all();
        mo::ThreadSpecificQueue::cleanup();
        std::function<void(void)> f;
        while (m_thread._work_queue.try_dequeue(f))
        {
        }
        while (m_thread._event_queue.try_dequeue(f))
        {
        }
    }
    volatile bool& _paused_flag;
    boost::condition_variable_any& _cv;
    mo::Thread& m_thread;
};

void Thread::main()
{
    ThreadSanitizer allocator_deleter(_paused, _cv, *this);
    (void)allocator_deleter;
    auto ctx = mo::Context::create();
    {
        boost::unique_lock<boost::recursive_timed_mutex> lock(_mtx);
        _ctx = ctx;
        lock.unlock();
        _cv.notify_all();
    }
    mo::Context::setDefaultThreadContext(_ctx);
    if (_on_start)
    {
        _on_start();
    }

    while (!_quit)
    {
        // Execute any events
        try
        {
            std::function<void(void)> f;
            if (_work_queue.try_dequeue(f))
            {
                f();
            }
            if (_event_queue.try_dequeue(f))
            {
                f();
            }
            mo::ThreadSpecificQueue::runOnce();

            int delay = 0;
            if (_inner_loop->hasSlots() && _run)
            {
                _paused = false;
                delay = (*_inner_loop)();
            }
            if (delay)
            {
                const auto start_time = mo::getCurrentTime();
                auto delta = mo::Time_t(mo::getCurrentTime() - start_time);
                bool processed_work = false;
                while (delta < mo::Time_t(mo::ms * delay))
                {
                    if (_work_queue.try_dequeue(f))
                    {
                        processed_work = true;
                        f();
                    }
                    if (_event_queue.try_dequeue(f))
                    {
                        processed_work = true;
                        f();
                    }
                    if (mo::ThreadSpecificQueue::runOnce())
                    {
                        processed_work = true;
                    }

                    delta = mo::Time_t(mo::getCurrentTime() - start_time);
                    if (!processed_work)
                    {
                        boost::this_thread::sleep_for(
                            boost::chrono::milliseconds(delay) -
                            boost::chrono::milliseconds(
                                std::chrono::duration_cast<std::chrono::milliseconds>(mo::getCurrentTime() - start_time)
                                    .count()));
                        break;
                    }
                }
                auto size = mo::ThreadSpecificQueue::size();
                if (size)
                {
                    MO_LOG(debug) << size << " events unprocessed on thread " << _name << " [" << getThreadId(_thread)
                                  << "]";
                }

                if (size > 100)
                {
                    mo::ThreadSpecificQueue::run();
                }
            }
            if (!_run)
            {
                _paused = true;
                _cv.notify_all();
            }
        }
        catch (cv::Exception& /*e*/)
        {
        }
        catch (boost::thread_interrupted& /*e*/)
        {
        }
        catch (...)
        {
        }
    }

    ctx.reset();
    _ctx.reset();
}

size_t Thread::getId() const
{
    return getThreadId(_thread);
}

const std::string& Thread::getThreadName() const
{
    return _name;
}

bool Thread::isOnThread() const
{
    return getId() == getThisThread();
}
