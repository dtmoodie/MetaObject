#ifndef MO_THREAD_FIBER_SCHEDULER_HPP
#define MO_THREAD_FIBER_SCHEDULER_HPP
#include "fiber_include.hpp"

#include "FiberProperties.hpp"

#include "detail/ContextQueue.hpp"

#include <MetaObject/thread/fiber_include.hpp>

#include <boost/fiber/algo/algorithm.hpp>
#include <boost/fiber/scheduler.hpp>

namespace mo
{
    class ThreadPool;

    struct MO_EXPORTS PriorityScheduler : public boost::fibers::algo::algorithm_with_properties<FiberProperty>
    {
        static PriorityScheduler* current();

        PriorityScheduler(std::weak_ptr<ThreadPool> pool, uint64_t work_threshold = 100);
        PriorityScheduler(std::weak_ptr<ThreadPool> pool, std::condition_variable** wakeup_cv);
        ~PriorityScheduler() override;

        void awakened(boost::fibers::context* ctx, FiberProperty& props) noexcept override;

        boost::fibers::context* pick_next() noexcept override;

        bool has_ready_fibers() const noexcept override;

        void property_change(boost::fibers::context* ctx, FiberProperty& props) noexcept override;

        void suspend_until(std::chrono::steady_clock::time_point const& time_point) noexcept override;

        void notify() noexcept override;

        boost::fibers::context* steal();

        void releaseAssistant();

        size_t size() const;

      private:
        using Queue = boost::fibers::scheduler::ready_queue_type;

        mutable boost::fibers::detail::spinlock m_work_spinlock;
        Queue m_work_queue;

        std::mutex m_mtx;
        std::condition_variable m_cv;
        bool m_flag = false;
        std::weak_ptr<ThreadPool> m_pool;
        uint64_t m_work_threshold;
        std::shared_ptr<Thread> m_assistant;
        // Every scheduler (main-thread and pool workers) runs the worker
        // attach/detach discipline: a context is attached to the
        // scheduler's worker-queue when picked (pick_next) and detached
        // when re-enqueued (awakened). boost.fiber's worker_queue_ is
        // unsynchronized, so this keeps each worker_queue_ single-threaded
        // (only its owning thread links/unlinks it: fiber construction
        // on the owning thread attaches and is immediately followed by
        // the same-thread detach in awakened), which the cross-thread
        // steal path relies on (steal never unlinks ready-queue contexts
        // - see steal/awakened in FiberScheduler.cpp).
        bool m_is_worker = true;
        static void setCurrent(PriorityScheduler*);
    };

    // Pool-owned lifetime guard for a scheduler.
    //
    // pick_next() copies the pool's scheduler list under the pool lock and
    // dereferences the entries AFTER unlocking; a worker's scheduler is
    // destroyed when its thread exits (thread-local boost scheduler
    // teardown), so a raw PriorityScheduler* copied that way is a
    // use-after-free hazard (check-then-use TOCTOU). The pool keeps one of
    // these per scheduler: the entry outlives the scheduler (the pool is a
    // process-lifetime singleton) and every access checks liveness under
    // its own mutex. A worker retires its own entry at the end of
    // Thread::main(), i.e. before its thread (and thus its scheduler) is
    // destroyed.
    struct MO_EXPORTS SchedulerEntry
    {
        explicit SchedulerEntry(PriorityScheduler* s)
            : m_sched(s)
        {
        }

        // The scheduler, or nullptr once retired.
        PriorityScheduler* scheduler()
        {
            std::lock_guard<std::mutex> lk(m_mtx);
            return m_sched;
        }

        // Steal work from the scheduler if it is still alive.
        boost::fibers::context* steal()
        {
            std::lock_guard<std::mutex> lk(m_mtx);
            return nullptr != m_sched ? m_sched->steal() : nullptr;
        }

        void releaseAssistant()
        {
            std::lock_guard<std::mutex> lk(m_mtx);
            if (nullptr != m_sched)
            {
                m_sched->releaseAssistant();
            }
        }

        // Idempotent: called by the retiring worker and (again) by
        // ~PriorityScheduler at thread exit.
        void retire()
        {
            std::lock_guard<std::mutex> lk(m_mtx);
            m_sched = nullptr;
        }

      private:
        std::mutex m_mtx;
        PriorityScheduler* m_sched = nullptr;
    };
} // namespace mo

#endif // MO_THREAD_FIBER_SCHEDULER_HPP
