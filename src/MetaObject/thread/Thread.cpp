#include "MetaObject/Thread/Thread.hpp"
#include "MetaObject/Signals/TypedSignalRelay.hpp"
#include "MetaObject/Context.hpp"
#include "MetaObject/Signals/TypedSlot.hpp"
#include "MetaObject/Thread/BoostThread.h"
#include "MetaObject/Thread/ThreadRegistry.hpp"
#include "MetaObject/Logging/Profiling.hpp"
#include "MetaObject/Detail/Allocator.hpp"

using namespace mo;


void Thread::PushEventQueue(const std::function<void(void)>& f)
{
    boost::recursive_mutex::scoped_lock lock(_mtx);
    _event_queue.push(f);
    _cv.notify_all();
}
// Work can be stolen and can exist on any thread
void Thread::PushWork(const std::function<void(void)>& f)
{
    boost::recursive_mutex::scoped_lock lock(_mtx);
    _work_queue.push(f);
    _cv.notify_all();
}
void Thread::Start()
{
    _run = true;
    _cv.notify_all();
}
void Thread::Stop()
{
    _run = false;
    _thread.interrupt();
    int wait_count = 0;
    while(!_paused)
    {
        boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
        ++wait_count;
        if(wait_count % 1000 == 0)
            LOG(warning) << "Waited 1000 seconds for " << this->_name << " to stop";
    }
}
void Thread::SetExitCallback(const std::function<void(void)>& f)
{
    _on_exit = f;
}
void Thread::SetStartCallback(const std::function<void(void)>& f)
{
    _on_start = f;
}
std::shared_ptr<Connection> Thread::SetInnerLoop(TypedSlot<int(void)>* slot)
{
    return slot->Connect(_inner_loop);
}
ThreadPool* Thread::GetPool() const
{
    return _pool;
}
Context* Thread::GetContext()
{
    boost::recursive_mutex::scoped_lock lock(_mtx);
    if(!_ctx)
    {
        _cv.wait_for(lock, boost::chrono::seconds(5));
    }
    return _ctx;
}

Thread::Thread()
{
    _pool = nullptr;
    _ctx = nullptr;
    _inner_loop.reset(new mo::TypedSignalRelay<int(void)>());
    _quit = false;
    _thread = boost::thread(&Thread::Main, this);
    _paused = false;

}
Thread::Thread(ThreadPool* pool)
{
    _inner_loop.reset(new mo::TypedSignalRelay<int(void)>());
    _pool = pool;
    _ctx = nullptr;
    _quit = false;
    _thread = boost::thread(&Thread::Main, this);

}

Thread::~Thread()
{
    PROFILE_FUNCTION
    _quit = true;
    _run = false;
    LOG(info) << "Shutting down " << this->_name << " thread";
    _thread.interrupt();
    if(!_thread.timed_join(boost::posix_time::time_duration(0,0,10)))
    {
        LOG(warning) << this->_name << " did not join after waiting 10 seconds";
        while(!_thread.timed_join(boost::posix_time::time_duration(0,0,10)))
        {
            _thread.interrupt();
        }
    }
}
void Thread::HandleEvents(int ms)
{
    boost::posix_time::ptime start = boost::posix_time::microsec_clock::universal_time();
    {
        boost::recursive_mutex::scoped_lock lock(_mtx);
        while(mo::ThreadSpecificQueue::RunOnce())
        {
            if(boost::posix_time::time_duration(
                boost::posix_time::microsec_clock::universal_time() - start).total_milliseconds() >= ms)
            {
                return;
            }
        }
        while (_work_queue.size())
        {
            _work_queue.back()();
            _work_queue.pop();
            if(boost::posix_time::time_duration(boost::posix_time::microsec_clock::universal_time()
                                                - start).total_milliseconds() >= ms)
            {
                return;
            }
        }
        while(_event_queue.size())
        {
            _event_queue.back()();
            _event_queue.pop();
            if(boost::posix_time::time_duration(
                boost::posix_time::microsec_clock::universal_time()- start).total_milliseconds() >= ms)
            {
                return;
            }
        }
    }
    boost::posix_time::time_duration elapsed = boost::posix_time::microsec_clock::universal_time() - start;
    if(elapsed.total_milliseconds() < ms)
    {
        boost::this_thread::sleep_for(boost::chrono::milliseconds(ms - elapsed.total_milliseconds()));
    }
}
struct ThreadAllocatorDeleter
{
    ~ThreadAllocatorDeleter()
    {
        mo::Allocator::CleanupThreadSpecificAllocator();
    }
};

void Thread::Main()
{
    ThreadAllocatorDeleter allocator_deleter;
    (void)allocator_deleter;
    mo::Context ctx;
    {
        boost::recursive_mutex::scoped_lock lock(_mtx);
        _ctx = &ctx;
        lock.unlock();
        _cv.notify_all();
    }

    mo::Context::SetDefaultThreadContext(_ctx);
    if(_on_start)
        _on_start();

    while(!_quit)
    {
        try
        {
            boost::recursive_mutex::scoped_lock lock(_mtx);
            PROFILE_RANGE(events);
            while (_work_queue.size())
            {
                _work_queue.back()();
                _work_queue.pop();
            }
            while(_event_queue.size())
            {
                _event_queue.back()();
                _event_queue.pop();
            }
            mo::ThreadSpecificQueue::RunOnce();
        }catch(boost::thread_interrupted& e)
        {

        }catch(...)
        {

        }

        if(_run)
        {
            try
            {
                if (_inner_loop->HasSlots())
                {
                    PROFILE_RANGE(inner_loop);
                    int delay = (*_inner_loop)();
                    if (delay)
                    {
                        // process events
                        PROFILE_RANGE(events);
                        HandleEvents(delay);
                    }
                }else
                {
                    HandleEvents(10);
                }
            }catch(boost::thread_interrupted& e)
            {
                if(_quit)
                {
                    break;
                }
            }catch(...)
            {
                boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
                LOG(warning) << "Unknown exception thrown in inner loop, in thread " << _name;
            }
        }else
        {
            while(!_run && ! boost::this_thread::interruption_requested())
            {

                try
                {
                    PROFILE_RANGE(events);
                    _paused = true;
                    boost::recursive_mutex::scoped_lock lock(_mtx);
                    HandleEvents(10);
                    _cv.wait_for(lock, boost::chrono::milliseconds(10));
                    {
                        while (_work_queue.size())
                        {
                            _work_queue.back()();
                            _work_queue.pop();
                        }
                        while(_event_queue.size())
                        {
                            _event_queue.back()();
                            _event_queue.pop();
                        }
                        mo::ThreadSpecificQueue::RunOnce();
                    }
                }catch(boost::thread_interrupted& e)
                {
                    if(_quit)
                    {
                        break;
                    }
                }catch(std::exception& e)
                {

                }catch(...)
                {

                }
            }
            if(boost::this_thread::interruption_requested() && _quit)
                break;
            if(_on_start)
                _on_start();
            _paused = false;
        }
    }
    _paused = true;
    if(_on_exit)
        _on_exit();
    if(_quit == false)
        LOG(warning)  << _name << " exiting without quit being called";
    LOG(info) << _name << " Thread exiting";

}
size_t Thread::GetId() const
{
    return GetThreadId(_thread);
}
bool Thread::IsOnThread() const
{
    return GetId() == GetThisThread();
}
