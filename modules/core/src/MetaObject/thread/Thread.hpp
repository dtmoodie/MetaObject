#ifndef MO_THREAD_HPP
#define MO_THREAD_HPP

#include <MetaObject/core/detail/ConcurrentQueue.hpp>
#include <MetaObject/core/detail/forward.hpp>
#include <MetaObject/detail/Export.hpp>
#include <MetaObject/signals/TSignalRelay.hpp>

#include <MetaObject/thread/ConditionVariable.hpp>
#include <MetaObject/thread/Mutex.hpp>

#include <ct/reflect.hpp>
#include <ct/reflect_macros.hpp>

#include <boost/thread.hpp>

#include <chrono>
#include <functional>
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
        Thread(ThreadPool* pool = nullptr);
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
        std::condition_variable* m_scheduler_wakeup_cv = nullptr;
    };
} // namespace mo

namespace ct
{
    REFLECT_BEGIN(mo::Thread)
        static mo::IAsyncStreamPtr_t getStream(const mo::Thread& thread)
        {
            return thread.asyncStream();
        }
        MEMBER_FUNCTION(stream, &getStream)
        PROPERTY(name, &mo::Thread::threadName, &mo::Thread::setName)
    REFLECT_END;
} // namespace ct
#endif // MO_THREAD_HPP
