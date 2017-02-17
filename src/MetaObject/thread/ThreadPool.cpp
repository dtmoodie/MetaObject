#include "MetaObject/Thread/ThreadPool.hpp"

using namespace mo;

ThreadPool* ThreadPool::Instance()
{
    static ThreadPool* g_inst = nullptr;
    if(g_inst == nullptr)
    {
        g_inst = new ThreadPool();
    }
    return g_inst;
}

ThreadHandle ThreadPool::RequestThread()
{
    for(auto& thread : _threads)
    {
        if(thread.available)
        {
            thread.ref_count = 0;
            return ThreadHandle(thread.thread, &thread.ref_count);
        }
    }
    _threads.emplace_back(false, new Thread(this));
    return ThreadHandle(_threads.back().thread, &_threads.back().ref_count);
}

void ThreadPool::Cleanup()
{
    for(auto& thread : _threads)
    {
        delete thread.thread;
    }
    _threads.clear();
}

void ThreadPool::ReturnThread(Thread* thread_)
{
    for(auto& thread : _threads)
    {
        if(thread.thread == thread_)
        {
            thread.available = true;
            thread_->Stop();
        }
    }
}