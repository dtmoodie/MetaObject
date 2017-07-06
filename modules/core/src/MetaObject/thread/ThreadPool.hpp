#pragma once
#include "Thread.hpp"
#include "ThreadHandle.hpp"
#include <MetaObject/detail/Export.hpp>
namespace mo {
class Thread;
class MO_EXPORTS ThreadPool {
public:
    static ThreadPool* instance();
    ThreadHandle       requestThread();
    void               cleanup();

protected:
    friend class ThreadHandle;
    void returnThread(Thread* thread);

private:
    struct PooledThread {
        PooledThread(bool available_, Thread* thread_)
            : available(available_)
            , thread(thread_) {}
        ~PooledThread();
        bool    available = true;
        int     ref_count = 0;
        Thread* thread;
    };
    std::list<PooledThread> _threads;
};
}
