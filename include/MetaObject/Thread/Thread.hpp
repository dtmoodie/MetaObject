#pragma once
#include <MetaObject/Detail/Export.hpp>
#include <MetaObject/Signals/TypedSignalRelay.hpp>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>
#include <functional>
#include <queue>

namespace mo
{
    class ThreadPool;
    class Context;
    class ThreadHandle;
    class ISlot;
    class MO_EXPORTS Thread
    {
    public:
        // Events have to be handled by this thread
        void PushEventQueue(const std::function<void(void)>& f);
        // Work can be stolen and can exist on any thread
        void PushWork(const std::function<void(void)>& f);
        void Start();
        void Stop();
        size_t GetId() const;
        bool IsOnThread() const;

        void SetExitCallback(const std::function<void(void)>& f);
        void SetStartCallback(const std::function<void(void)>& f);
        //void SetInnerLoop(const std::function<int(void)>& f);
        std::shared_ptr<Connection> SetInnerLoop(TypedSlot<int(void)>* slot);
        ThreadPool* GetPool() const;
        Context* GetContext();
    protected:
        friend class ThreadPool;
        friend class ThreadHandle;
        
        Thread();
        Thread(ThreadPool* pool);
        ~Thread();
        void Main();
        void HandleEvents(int ms);

        Thread& operator=(const Thread&) = delete;
        Thread(const Thread&)            = delete;

        boost::thread                        _thread;
        std::shared_ptr<mo::TypedSignalRelay<int(void)>> _inner_loop;

        std::function<void(void)>             _on_start;
        std::function<void(void)>             _on_exit;
        Context*                              _ctx;
        ThreadPool*                           _pool;
        boost::condition_variable_any         _cv;
        boost::recursive_timed_mutex                _mtx;
        // if _run == true, execute the main inner loop
        volatile bool                         _run;
        // if _quit == true, cleanup and exit the thread
        volatile bool                         _quit;
        // Set by work thread, if true then it is not executing the inner loop
        volatile bool                         _paused;
        // Set by the thread handle, set this flag to skip executing the event loop because inner loop needs to run again asap
        volatile bool                         _run_inner_loop;
        std::queue<std::function<void(void)>> _work_queue;
        std::queue<std::function<void(void)>> _event_queue;
        std::string                           _name;
    };
}
