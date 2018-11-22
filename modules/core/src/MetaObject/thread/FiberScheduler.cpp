#include "FiberScheduler.hpp"
#include <MetaObject/logging/logging.hpp>

namespace mo
{

    PriorityScheduler::PriorityScheduler()
    {
        MO_LOG(info, "Instantiating scheduler");
    }

    void PriorityScheduler::awakened(boost::fibers::context* ctx, FiberProperty& props) noexcept
    {
        const auto ctx_priority = props.getPriority();

        Queue::iterator i(std::find_if(m_queue.begin(), m_queue.end(), [ctx_priority, this](boost::fibers::context& c) {
            return properties(&c).getPriority() < ctx_priority;
        }));

        m_queue.insert(i, *ctx);
    }

    boost::fibers::context* PriorityScheduler::pick_next() noexcept
    {
        if (m_queue.empty())
        {
            return nullptr;
        }
        boost::fibers::context* ctx(&m_queue.front());
        m_queue.pop_front();
        return ctx;
    }

    bool PriorityScheduler::has_ready_fibers() const noexcept
    {
        return !m_queue.empty();
    }

    void PriorityScheduler::property_change(boost::fibers::context* ctx, FiberProperty& props) noexcept
    {
        if (!ctx->ready_is_linked())
        {
            return;
        }
        ctx->ready_unlink();
        awakened(ctx, props);
    }

    void PriorityScheduler::suspend_until(std::chrono::steady_clock::time_point const& time_point) noexcept
    {
        if ((std::chrono::steady_clock::time_point::max)() == time_point)
        {
            std::unique_lock<std::mutex> lk(m_mtx);
            m_cv.wait(lk, [this]() { return m_flag; });
            m_flag = false;
        }
        else
        {
            std::unique_lock<std::mutex> lk(m_mtx);
            m_cv.wait_until(lk, time_point, [this]() { return m_flag; });
            m_flag = false;
        }
    }

    void PriorityScheduler::notify() noexcept
    {
        std::unique_lock<std::mutex> lk(m_mtx);
        m_flag = true;
        lk.unlock();
        m_cv.notify_all();
    }
}
