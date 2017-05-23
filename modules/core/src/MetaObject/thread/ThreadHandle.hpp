#pragma once

#include <MetaObject/detail/Export.hpp>
#include <memory>
#include <functional>

namespace mo
{
    class Thread;
    class ThreadPool;
    class Context;
    class ISlot;
    class Connection;
    template<class T> class TSlot;
    class MO_EXPORTS ThreadHandle
    {
    public:
        ThreadHandle();
        ThreadHandle(const ThreadHandle& other);
        ThreadHandle(ThreadHandle&& other);

        ~ThreadHandle();
        ThreadHandle& operator=(ThreadHandle&& other);
        ThreadHandle& operator=(const ThreadHandle& other);

        Context* getContext();
        size_t getId() const;
        bool isOnThread() const;
        void pushEventQueue(const std::function<void(void)>& f);
        // Work can be stolen and can exist on any thread
        void pushWork(const std::function<void(void)>& f);
        void start();
        void stop();
        bool getIsRunning() const;
        void setExitCallback(const std::function<void(void)>& f);
        void setStartCallback(const std::function<void(void)>& f);
        void setThreadName(const std::string& name);
        std::shared_ptr<Connection> setInnerLoop(TSlot<int(void)>* slot);
    protected:
        friend class ThreadPool;
        ThreadHandle(Thread* thread, int* ref_count);
        Thread* _thread;
        int* _ref_count;
        void decrement();
        void increment();
    };
}
