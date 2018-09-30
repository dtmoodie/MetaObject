#include "MetaObject/thread/ThreadPool.hpp"
#include "MetaObject/thread/Thread.hpp"
using namespace mo;
ThreadPool::PooledThread::~PooledThread()
{
    delete this->thread;
}

ThreadPool* ThreadPool::instance()
{
    static ThreadPool* g_inst = nullptr;
    if (g_inst == nullptr)
    {
        g_inst = new ThreadPool();
    }
    return g_inst;
}

ThreadHandle ThreadPool::requestThread()
{
    for (auto& thread : _threads)
    {
        if (thread.available)
        {
            thread.ref_count = 0;
            return ThreadHandle(thread.thread, &thread.ref_count);
        }
    }
    _threads.emplace_back(false, new Thread(this));
    return ThreadHandle(_threads.back().thread, &_threads.back().ref_count);
}

void ThreadPool::cleanup()
{
    _threads.clear();
}

void ThreadPool::returnThread(Thread* thread_)
{
    for (auto& thread : _threads)
    {
        if (thread.thread == thread_)
        {
            thread.available = true;
            thread_->stop();
        }
    }
}
