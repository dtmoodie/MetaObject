#pragma once
#include "Thread.hpp"
#include "ThreadHandle.hpp"
#include <MetaObject/detail/Export.hpp>
namespace mo
{
    class Thread;
    struct PriorityScheduler;
    class MO_EXPORTS ThreadPool : public std::enable_shared_from_this<ThreadPool>
    {
      public:
        ~ThreadPool();
        std::shared_ptr<Thread> requestThread();
        void cleanup();

        void addScheduler(PriorityScheduler*);
        void removeScheduler(PriorityScheduler*);
        std::vector<PriorityScheduler*> getSchedulers() const;

      private:
        mutable std::mutex m_mtx;
        void returnThread(const std::shared_ptr<Thread>& thread);

        std::list<std::shared_ptr<Thread>> m_free_threads;
        std::list<std::shared_ptr<Thread>> m_running_threads;
        std::set<PriorityScheduler*> m_schedulers;
    };
} // namespace mo
