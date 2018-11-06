#pragma once
#include "Thread.hpp"
#include "ThreadHandle.hpp"
#include <MetaObject/detail/Export.hpp>
namespace mo
{
    class Thread;
    class MO_EXPORTS ThreadPool
    {
      public:
        static ThreadPool* instance();
        std::shared_ptr<Thread> requestThread();
        void cleanup();

      private:
        std::mutex m_mtx;
        void returnThread(const std::shared_ptr<Thread>& thread);

        std::list<std::shared_ptr<Thread>> m_threads;
    };
}
