#include "MetaObject/thread/Thread.hpp"
#include "MetaObject/signals/TSignalRelay.hpp"
#include "MetaObject/core/Context.hpp"
#include "MetaObject/signals/TSlot.hpp"
#include "MetaObject/thread/BoostThread.hpp"
#include "MetaObject/thread/ThreadRegistry.hpp"
#include "MetaObject/logging/Profiling.hpp"
#include "MetaObject/core/detail/Allocator.hpp"

using namespace mo;


void Thread::pushEventQueue(const std::function<void(void)>& f){
    boost::unique_lock<boost::recursive_timed_mutex> lock(_mtx);
    _event_queue.push(f);
    lock.unlock();
    _cv.notify_all();
}
// Work can be stolen and can exist on any thread
void Thread::pushWork(const std::function<void(void)>& f){
    boost::unique_lock<boost::recursive_timed_mutex> lock(_mtx);
    _work_queue.push(f);
    lock.unlock();
    _cv.notify_all();
}

void Thread::start(){
    _run = true;
    _cv.notify_all();
}

void Thread::stop(){
    _run = false;
    _thread.interrupt();
    int wait_count = 0;
    while(!_paused && _thread.joinable()){
        boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
        ++wait_count;
        if(wait_count % 1000 == 0)
            LOG(warning) << "Waited " << wait_count / 1000 << " second for " << this->_name << " to stop";
    }
}
void Thread::setExitCallback(const std::function<void(void)>& f){
    _on_exit = f;
}
void Thread::setStartCallback(const std::function<void(void)>& f){
    _on_start = f;
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
    LOG(info) << "Shutting down " << this->_name << " thread";
    _thread.interrupt();
    if(!_thread.timed_join(boost::posix_time::time_duration(0,0,10))){
        LOG(warning) << this->_name << " did not join after waiting 10 seconds";
    }
}

void Thread::handleEvents(int ms){
    boost::posix_time::ptime start = boost::posix_time::microsec_clock::universal_time();
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
    }
}
struct ThreadSanitizer
{
    ThreadSanitizer(volatile bool& paused_flag, boost::condition_variable_any& cv):
        _paused_flag(paused_flag),
        _cv(cv){
    }
    ~ThreadSanitizer(){
        mo::Allocator::cleanupThreadSpecificAllocator();
        _paused_flag = true;
        _cv.notify_all();
    }
    volatile bool& _paused_flag;
    boost::condition_variable_any& _cv;
};

void Thread::main(){
    ThreadSanitizer allocator_deleter(_paused, _cv);
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
            boost::unique_lock<boost::recursive_timed_mutex> lock(_mtx, boost::chrono::milliseconds(1));
            if(lock.owns_lock()){
                if(_work_queue.size()){
                    _work_queue.back()();
                    _work_queue.pop();
                }
                if(_event_queue.size()){
                    _event_queue.back()();
                    _event_queue.pop();
                }
                lock.unlock();
                mo::ThreadSpecificQueue::runOnce();
            }

            int delay = 0;
            if(_inner_loop->HasSlots() && _run){
                delay = (*_inner_loop)();
            }
            if(delay){
                while (_work_queue.size()){
                    _work_queue.back()();
                    _work_queue.pop();
                }
                while (_event_queue.size()){
                    _event_queue.back()();
                    _event_queue.pop();
                }
            }
            if(!_run){
                _paused = true;
                _cv.notify_all();
            }
        }catch(cv::Exception& e){
        }catch(boost::thread_interrupted& e){
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
