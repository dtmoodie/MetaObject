#ifndef MO_THREAD_FIBER_SCHEDULER_HPP
#define MO_THREAD_FIBER_SCHEDULER_HPP
#include "FiberProperties.hpp"
#include "detail/ContextQueue.hpp"

#include <boost/fiber/all.hpp>

namespace mo
{
    class ThreadPool;

    struct PriorityScheduler : public boost::fibers::algo::algorithm_with_properties<FiberProperty>
    {
        PriorityScheduler(ThreadPool* pool, const uint64_t work_threshold = 100, const bool suspend = false);

        void awakened(boost::fibers::context* ctx, FiberProperty& props) noexcept override;

        boost::fibers::context* pick_next() noexcept override;

        bool has_ready_fibers() const noexcept override;

        void property_change(boost::fibers::context* ctx, FiberProperty& props) noexcept override;

        void suspend_until(std::chrono::steady_clock::time_point const& time_point) noexcept override;

        void notify() noexcept override;

        boost::fibers::context* steal();

      private:
        using Queue = ContextQueue;
        Queue m_work_queue;
        Queue m_event_queue;

        std::mutex m_mtx;
        std::condition_variable m_cv;
        bool m_flag = false;
        ThreadPool* m_pool;
        uint64_t m_work_threshold;
        std::shared_ptr<Thread> m_assistant;
        bool m_suspend;
    };
}

#endif // MO_THREAD_FIBER_SCHEDULER_HPP
