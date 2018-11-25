#include "FiberScheduler.hpp"
#include <MetaObject/logging/logging.hpp>
#include <MetaObject/thread/ThreadPool.hpp>

#include <boost/context/detail/prefetch.hpp>
namespace mo
{

    PriorityScheduler::PriorityScheduler(std::shared_ptr<ThreadPool> pool, const uint64_t wt)
        : m_pool(pool)
        , m_work_threshold(wt)
    {
        m_pool->addScheduler(this);
        MO_LOG(info, "Instantiating scheduler");
    }

    PriorityScheduler::PriorityScheduler(std::shared_ptr<ThreadPool> pool, std::condition_variable** wakeup_cv)
        : m_pool(pool)
        , m_work_threshold(std::numeric_limits<uint64_t>::max())
        , m_is_worker(true)
    {
        *wakeup_cv = &m_cv;
        m_pool->addScheduler(this);
        MO_LOG(info, "Instantiating worker scheduler");
    }

    PriorityScheduler::~PriorityScheduler()
    {
        m_assistant.reset();
        m_pool->removeScheduler(this);
    }

    void PriorityScheduler::awakened(boost::fibers::context* ctx, FiberProperty& props) noexcept
    {
        const auto ctx_priority = props.getPriority();

        if (m_is_worker)
        {
            if (!ctx->is_context(boost::fibers::type::pinned_context))
            {
                ctx->detach();
            }
        }

        boost::fibers::detail::spinlock_lock lk{m_work_spinlock};

        auto itr =
            std::find_if(m_work_queue.begin(), m_work_queue.end(), [ctx_priority, this](boost::fibers::context& c) {
                return properties(&c).getPriority() < ctx_priority;
            });

        m_work_queue.insert(itr, *ctx);
        const uint64_t size = m_work_queue.size();

        if (size > m_work_threshold && nullptr == m_assistant)
        {
            m_assistant = m_pool->requestThread();
        }
    }

    boost::fibers::context* PriorityScheduler::steal()
    {
        boost::fibers::detail::spinlock_lock lk{m_work_spinlock};
        if (m_work_queue.empty())
        {
            return nullptr;
        }

        boost::fibers::context* victim = &m_work_queue.back();
        if (victim->is_context(boost::fibers::type::pinned_context))
        {
            return nullptr;
        }
        m_work_queue.pop_back();
        victim->detach();
        return victim;
    }

    boost::fibers::context* PriorityScheduler::pick_next() noexcept
    {
        boost::fibers::context* victim = nullptr;
        {
            boost::fibers::detail::spinlock_lock lk{m_work_spinlock};
            if (!m_work_queue.empty())
            {
                victim = &m_work_queue.front();
                m_work_queue.pop_front();
            }
        }

        if (nullptr != victim)
        {
            if (m_is_worker)
            {
                boost::context::detail::prefetch_range(victim, sizeof(boost::fibers::context));
                if (!victim->is_context(boost::fibers::type::pinned_context))
                {
                    boost::fibers::context::active()->attach(victim);
                }
            }
        }
        else
        {
            std::uint32_t id = 0;
            const auto schedulers = m_pool->getSchedulers();
            if (schedulers.size() == 1)
            {
                return victim;
            }
            const auto id_ = std::find(schedulers.begin(), schedulers.end(), this) - schedulers.begin();
            std::size_t count = 0, size = schedulers.size();
            static thread_local std::minstd_rand generator{std::random_device{}()};
            std::uniform_int_distribution<std::uint32_t> distribution{0, static_cast<std::uint32_t>(size - 1)};
            do
            {
                do
                {
                    ++count;
                    // random selection of one logical cpu
                    // that belongs to the local NUMA node
                    id = distribution(generator);
                    // prevent stealing from own scheduler
                } while (id == id_);
                // steal context from other scheduler
                victim = schedulers[id]->steal();
            } while (nullptr == victim && count < size);
            if (nullptr != victim)
            {
                boost::context::detail::prefetch_range(victim, sizeof(boost::fibers::context));
                BOOST_ASSERT(!victim->is_context(boost::fibers::type::pinned_context));
                boost::fibers::context::active()->attach(victim);
            }
        }

        return victim;
    }

    bool PriorityScheduler::has_ready_fibers() const noexcept
    {
        boost::fibers::detail::spinlock_lock lk1{m_work_spinlock};

        return !m_work_queue.empty();
    }

    void PriorityScheduler::property_change(boost::fibers::context* ctx, FiberProperty& props) noexcept
    {
        if (!ctx->ready_is_linked())
        {
            return;
        }
        {
            boost::fibers::detail::spinlock_lock lk1{m_work_spinlock};
            ctx->ready_unlink();
        }
        awakened(ctx, props);
    }

    void PriorityScheduler::suspend_until(std::chrono::steady_clock::time_point const& time_point) noexcept
    {
        if (m_is_worker)
        {
            if ((std::chrono::steady_clock::time_point::max)() == time_point)
            {
                std::unique_lock<std::mutex> lk{m_mtx};
                m_cv.wait(lk, [this]() { return m_flag; });
                m_flag = false;
            }
            else
            {
                std::unique_lock<std::mutex> lk{m_mtx};
                m_cv.wait_until(lk, time_point, [this]() { return m_flag; });
                m_flag = false;
            }
        }
    }

    void PriorityScheduler::notify() noexcept
    {
        if (m_is_worker)
        {
            std::unique_lock<std::mutex> lk(m_mtx);
            m_flag = true;
            lk.unlock();
            m_cv.notify_all();
        }
    }
}
