#pragma once
#include "Thread.hpp"
#include "ThreadHandle.hpp"
#include <MetaObject/detail/Export.hpp>
namespace mo
{
    class Thread;
    struct PriorityScheduler;
    struct SchedulerEntry; // defined in FiberScheduler.hpp (needs boost fiber types)
    class MO_EXPORTS ThreadPool : public std::enable_shared_from_this<ThreadPool>
    {
      public:
        ~ThreadPool();
        std::shared_ptr<Thread> requestThread();
        void cleanup();

        void addScheduler(PriorityScheduler*);
        void removeScheduler(PriorityScheduler*);
        // Retire the scheduler (mark its entry dead and remove it from the
        // pool). Called by a worker at the end of Thread::main() — before
        // its thread exits and destroys the scheduler — and again by
        // ~PriorityScheduler; safe to call twice.
        void retireScheduler(PriorityScheduler*);
        std::vector<std::shared_ptr<SchedulerEntry>> getSchedulers() const;

      private:
        mutable std::mutex m_mtx;
        void returnThread(const std::shared_ptr<Thread>& thread);

        std::list<std::shared_ptr<Thread>> m_free_threads;
        std::list<std::shared_ptr<Thread>> m_running_threads;
        // Entries (not raw pointers): they outlive the schedulers they
        // guard, so a list copied out of getSchedulers() can never be
        // dereferenced after the owning thread has exited.
        std::set<std::shared_ptr<SchedulerEntry>> m_schedulers;
    };
}
