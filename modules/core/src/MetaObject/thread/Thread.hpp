#pragma once

#include <MetaObject/core/detail/ConcurrentQueue.hpp>
#include <MetaObject/core/detail/Forward.hpp>
#include <MetaObject/detail/Export.hpp>
#include <MetaObject/signals/TSignalRelay.hpp>

#include <boost/thread.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>

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
        ContextPtr_t context(const Duration timeout = 5 * second);

      private:
        void main();

        boost::thread m_thread;

        Mutex_t m_mtx;
        std::function<void(void)> m_on_exit;

        ContextPtr_t m_ctx;
        ThreadPool* m_pool = nullptr;

        boost::condition_variable_any m_cv;

        moodycamel::ConcurrentQueue<std::function<void(void)>> m_work_queue;

        std::string m_name;
    };
}
