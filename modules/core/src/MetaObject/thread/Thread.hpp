#pragma once

#include <MetaObject/core/detail/ConcurrentQueue.hpp>
#include <MetaObject/core/detail/Forward.hpp>
#include <MetaObject/detail/Export.hpp>
#include <MetaObject/signals/TSignalRelay.hpp>

#include <boost/fiber/condition_variable.hpp>
#include <boost/fiber/recursive_timed_mutex.hpp>
#include <boost/thread.hpp>

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
        IAsyncStreamPtr_t asyncStream(const Duration timeout = 5 * second) const;

      private:
        void main();

        boost::thread m_thread;

        mutable Mutex_t m_mtx;
        mutable boost::fibers::condition_variable_any m_cv;

        std::function<void(void)> m_on_exit;

        IAsyncStreamPtr_t m_stream;
        ThreadPool* m_pool = nullptr;

        std::string m_name;
        std::condition_variable* m_scheduler_wakeup_cv = nullptr;
    };
}
