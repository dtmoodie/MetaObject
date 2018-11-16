#ifndef MO_THREAD_FIBER_SCHEDULER_HPP
#define MO_THREAD_FIBER_SCHEDULER_HPP
#include "FiberProperties.hpp"

#include <boost/fiber/algo/round_robin.hpp>
#include <boost/fiber/all.hpp>

namespace mo
{
    struct PriorityScheduler : public boost::fibers::algo::algorithm_with_properties<FiberProperty>
    {
        PriorityScheduler();

        void awakened(boost::fibers::context* ctx, FiberProperty& props) noexcept override;

        boost::fibers::context* pick_next() noexcept override;

        bool has_ready_fibers() const noexcept override;

        void property_change(boost::fibers::context* ctx, FiberProperty& props) noexcept override;

        void suspend_until(std::chrono::steady_clock::time_point const& time_point) noexcept override;

        void notify() noexcept override;

      private:
        using Queue = boost::fibers::scheduler::ready_queue_type;

        Queue m_queue;
        std::mutex m_mtx;
        std::condition_variable m_cv;
        bool m_flag = false;
    };
}

#endif // MO_THREAD_FIBER_SCHEDULER_HPP
