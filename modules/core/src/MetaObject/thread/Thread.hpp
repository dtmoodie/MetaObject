#ifndef MO_THREAD_HPP
#define MO_THREAD_HPP

#include <MetaObject/core/detail/ConcurrentQueue.hpp>
#include <MetaObject/core/detail/forward.hpp>
#include <MetaObject/detail/Export.hpp>
#include <MetaObject/signals/TSignalRelay.hpp>

#include <MetaObject/thread/ConditionVariable.hpp>
#include <MetaObject/thread/Mutex.hpp>

#include <boost/thread.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>

namespace mo
{
    class ThreadPool;
    class Context;
    class ThreadHandle;
    class ISlot;

    void MO_EXPORTS initThread();
    void MO_EXPORTS initThread(SystemTable& table);

    void MO_EXPORTS sleepFor(std::chrono::nanoseconds);
    void MO_EXPORTS sleepUntil(std::chrono::high_resolution_clock::time_point);
    void MO_EXPORTS yield();

    class MO_EXPORTS Thread
    {
      public:
        Thread(ThreadPool* pool);
        Thread(const Thread&) = delete;
        Thread& operator=(const Thread&) = delete;
        ~Thread();

        size_t threadId() const;
        bool isOnThread() const;
        const std::string& threadName() const;

        void setExitCallback(std::function<void(void)>&& f);

        void setName(const std::string& name);

        ThreadPool* pool() const;
        IAsyncStreamPtr_t asyncStream(Duration timeout = 5 * second) const;

      private:
        void main();

        boost::thread m_thread;

        mutable Mutex_t m_mtx;
        mutable ConditionVariable m_cv;

        std::function<void(void)> m_on_exit;

        IAsyncStreamPtr_t m_stream;
        ThreadPool* m_pool = nullptr;

        std::string m_name;

        // Set by the worker (Thread::main) once its scheduler is installed
        // and the wakeup cv pointer below is written. ~Thread() waits on
        // this handshake (bounded) before signaling shutdown, so the
        // publish of m_scheduler_wakeup_cv is synchronized and the worker
        // is known to have started.
        std::mutex m_ready_mtx;
        std::condition_variable m_ready_cv;
        bool m_scheduler_ready = false;

        // Written by the worker's scheduler constructor
        // (use_scheduling_algorithm); published under m_ready_mtx.
        std::condition_variable* m_scheduler_wakeup_cv = nullptr;

        // Set by ~Thread(); the worker's main() parks in a wait() whose
        // predicate checks this (see Thread::main). Atomic: it is set from
        // the owner thread and read from the worker thread.
        std::atomic<bool> m_shutdown{false};
    };
} // namespace mo
#endif // MO_THREAD_HPP