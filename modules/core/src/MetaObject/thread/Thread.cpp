#include "MetaObject/thread/Thread.hpp"
#include "MetaObject/signals/TSignalRelay.hpp"
#include "MetaObject/core/Context.hpp"
#include "MetaObject/signals/TSlot.hpp"
#include "MetaObject/thread/BoostThread.hpp"
#include "MetaObject/thread/ThreadRegistry.hpp"
#include "MetaObject/logging/Profiling.hpp"
#include "MetaObject/core/detail/Allocator.hpp"
#include "MetaObject/core/detail/Time.hpp"
using namespace mo;


void Thread::pushEventQueue(const std::function<void(void)>& f){
    boost::unique_lock<boost::recursive_timed_mutex> lock(_mtx);
    _event_queue.enqueue(f);
    lock.unlock();
    _cv.notify_all();
    
}
// Work can be stolen and can exist on any thread
void Thread::pushWork(const std::function<void(void)>& f){
    boost::unique_lock<boost::recursive_timed_mutex> lock(_mtx);
    _work_queue.enqueue(f);
    lock.unlock();
    _cv.notify_all();
}

void Thread::start(){
    _run = true;
    _cv.notify_all();
}

void Thread::stop(){
    if(_run == false && _paused == true)
        return;
    _run = false;
    _thread.interrupt();
    int wait_count = 0;
    while(!_paused){
        boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
        ++wait_count;
        if(wait_count % 1000 == 0)
            LOG(warning) << "Waited " << wait_count / 1000 << " second for " << this->_name << " to stop";
    }
    //_thread.join();
    LOG(info) << _name << " has stopped";
}
void Thread::setExitCallback(const std::function<void(void)>& f){
    _on_exit = f;
}
void Thread::setStartCallback(const std::function<void(void)>& f){
    _on_start = f;
}
void Thread::setName(const std::string& name){
    this->_name = name;
    setThreadName(_thread, name);
}

std::shared_ptr<Connection> Thread::setInnerLoop(TSlot<int(void)>* slot){
    return slot->connect(_inner_loop);
}
ThreadPool* Thread::getPool() const{
    return _pool;
}

ContextPtr_t Thread::getContext(){
    boost::unique_lock<boost::recursive_timed_mutex> lock(_mtx);
    if(!_ctx){
        _cv.wait_for(lock, boost::chrono::seconds(5));
    }
    return _ctx;
}

Thread::Thread(){
    _pool = nullptr;
    _inner_loop.reset(new mo::TSignalRelay<int(void)>());
    _quit = false;
    _thread = boost::thread(&Thread::main, this);
    _paused = false;
}

Thread::Thread(ThreadPool* pool){
    _inner_loop.reset(new mo::TSignalRelay<int(void)>());
    _pool = pool;
    _quit = false;
    _thread = boost::thread(&Thread::main, this);
}

Thread::~Thread(){
    PROFILE_FUNCTION
    _quit = true;
    _run = false;
    LOG(info) << "Waiting for " << this->_name << " to join";
    _thread.interrupt();
    if(!_thread.timed_join(boost::posix_time::time_duration(0,0,10))){
        LOG(warning) << this->_name << " did not join after waiting 10 seconds";
    }
    LOG(info) << this->_name << " shutdown complete";
}

void Thread::handleEvents(int ms){
    /*boost::posix_time::ptime start = boost::posix_time::microsec_clock::universal_time();
    {
        boost::unique_lock<boost::recursive_timed_mutex> lock(_mtx);
        while(mo::ThreadSpecificQueue::runOnce())
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
    }*/
}
struct mo::Thread::ThreadSanitizer
{
    ThreadSanitizer(volatile bool& paused_flag, boost::condition_variable_any& cv, mo::Thread& thread):
        _paused_flag(paused_flag),
        _cv(cv),
        m_thread(thread){
    }
    ~ThreadSanitizer(){
        LOG(info) << m_thread._name << " exiting";
        mo::Allocator::cleanupThreadSpecificAllocator();
        _paused_flag = true;
        _cv.notify_all();
    }
    volatile bool& _paused_flag;
    boost::condition_variable_any& _cv;
    mo::Thread& m_thread;
};

void Thread::main(){
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
    if(_on_start)
        _on_start();
    while(!_quit){
        // Execute any events
        try{
            std::function<void(void)> f;
            if(_work_queue.try_dequeue(f)){
                f();
            }
            if(_event_queue.try_dequeue(f)){
                f();
            }
            mo::ThreadSpecificQueue::runOnce();

            int delay = 0;
            if(_inner_loop->HasSlots() && _run){
                _paused = false;
                delay = (*_inner_loop)();
            }
            if(delay){
                auto start_time = mo::getCurrentTime();
                bool processed_work = false;
                while(mo::Time_t(mo::getCurrentTime() - start_time) < mo::Time_t(mo::ms * delay)){
                    if(_work_queue.try_dequeue(f)){
                        processed_work = true;
                        f();
                    }
                    if(_event_queue.try_dequeue(f)){
                        processed_work = true;
                        f();
                    }
                    if(mo::ThreadSpecificQueue::runOnce())
                        processed_work = true;
                    if(!processed_work){
                        
                        //boost::this_thread::sleep_for();
                        break;
                    }
                        
                }
                auto size = mo::ThreadSpecificQueue::size();
                if(size)
                    LOG(trace) << size << " events unprocessed on thread " << _name << " [" << getThreadId(_thread) << "]";
            }
            if(!_run){
                _paused = true;
                _cv.notify_all();
            }
        }catch(cv::Exception& e){
            (void)e;
        }catch(boost::thread_interrupted& e){
            (void)e;
        }catch(...){
        }
    }
}

size_t Thread::getId() const{
    return getThreadId(_thread);
}

bool Thread::isOnThread() const{
    return getId() == getThisThread();
}
