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
        bool m_is_worker = false;
        static void setCurrent(PriorityScheduler*);
    };
} // namespace mo

#endif // MO_THREAD_FIBER_SCHEDULER_HPP
