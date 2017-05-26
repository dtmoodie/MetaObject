#include "MetaObject/thread/ThreadHandle.hpp"
#include "MetaObject/thread/ThreadPool.hpp"
#include "MetaObject/thread/Thread.hpp"
#include "MetaObject/logging/Profiling.hpp"
using namespace mo;
ThreadHandle::ThreadHandle(Thread* thread, int* ref_count)
{
    _ref_count = ref_count;
    _thread = thread;
    increment();
}
ThreadHandle::~ThreadHandle()
{
    decrement();
}
ThreadHandle::ThreadHandle()
{
    _ref_count = nullptr;
    _thread = nullptr;
}
ThreadHandle::ThreadHandle(const ThreadHandle& other)
{
    _ref_count = other._ref_count;
    _thread = other._thread;
    increment();
}
ThreadHandle::ThreadHandle(ThreadHandle&& other)
{
    _ref_count = other._ref_count;
    _thread = other._thread;
    other._ref_count = nullptr;
    other._thread = nullptr;
}
ThreadHandle& ThreadHandle::operator=(ThreadHandle&& other)
{
    _thread = other._thread;
    this->_ref_count = other._ref_count;
    other._ref_count = nullptr;
    other._thread = nullptr;
    return *this;
}
ThreadHandle& ThreadHandle::operator=(const ThreadHandle& other){
    decrement();
    this->_ref_count = other._ref_count;
    this->_thread = other._thread;
    increment();
    return *this;
}
ContextPtr_t ThreadHandle::getContext(){
    if(_thread){
        return _thread->getContext();
    }
    return ContextPtr_t();
}
void ThreadHandle::pushEventQueue(const std::function<void(void)>& f)
{
    if(_thread)
    {
        _thread->pushEventQueue(f);
    }
}
// Work can be stolen and can exist on any thread
void ThreadHandle::pushWork(const std::function<void(void)>& f)
{
    if(_thread)
    {
        _thread->pushWork(f);
    }
}
void ThreadHandle::start()
{
    if (_thread)
    {
        _thread->start();
    }
}
void ThreadHandle::stop()
{
    if(_thread)
    {
        _thread->stop();
    }
}
void ThreadHandle::setExitCallback(const std::function<void(void)>& f)
{
    if(_thread)
    {
        _thread->setExitCallback(f);
    }
}
void ThreadHandle::setStartCallback(const std::function<void(void)>& f)
{
    if(_thread)
    {
        _thread->setStartCallback(f);
    }
}

std::shared_ptr<Connection> ThreadHandle::setInnerLoop(TSlot<int(void)>* slot)
{
    if(_thread)
    {
        return _thread->setInnerLoop(slot);
    }
    return std::shared_ptr<Connection>();
}
void ThreadHandle::decrement()
{
    if (_ref_count)
    {
        --(*_ref_count);
        if (*_ref_count <= 0 && _thread)
     {
            ThreadPool* pool = _thread->getPool();
            if (pool)
            {
                pool->ReturnThread(_thread);
            }
            else
            {
                delete _thread;
            }
        }
    }
}
void ThreadHandle::increment()
{
    if(_ref_count)
    {
        ++(*_ref_count);
    }
}
size_t ThreadHandle::getId() const
{
    return _thread->getId();
}

bool ThreadHandle::isOnThread() const
{
    return _thread->isOnThread();
}
bool ThreadHandle::getIsRunning() const
{
    if(_thread == nullptr)
        return false;
    return _thread->_run && !_thread->_paused;
}
void ThreadHandle::setThreadName(const std::string& name)
{
    if(_thread)
    {
        if(_thread->isOnThread())
        {
            mo::setThreadName(name.c_str());
            mo::setStreamName(name.c_str(), _thread->getContext()->getCudaStream());
            _thread->_name = name;
            _thread->getContext()->setName(name);
        }else
        {
            Thread* thread = _thread;
            std::string name_ = name;
            _thread->pushEventQueue([name_, thread, this]()
            {
                mo::setThreadName(name_.c_str());
                mo::setStreamName(name_.c_str(), thread->getContext()->getCudaStream());
                _thread->_name = name_;
                _thread->getContext()->setName(name_);
            });
        }
    }
}
