#include "FiberScheduler.hpp"
#include <MetaObject/logging/logging.hpp>
#include <MetaObject/thread/ThreadPool.hpp>

#include <boost/context/detail/prefetch.hpp>
namespace mo
{

    WorkQueue::WorkQueue(PriorityLevels priority, PriorityScheduler* scheduler)
    {
        m_scheduler = scheduler;
        // can't do this from the ctr since attachQueue needs to be able to call shared_from_this
        // scheduler->attachQueue(*this, priority);
    }

    void WorkQueue::pushBack(boost::fibers::context& ctx)
    {
        boost::fibers::detail::spinlock_lock lk(m_work_spinlock);
        m_work_queue.push_back(ctx);
    }

    boost::fibers::context* WorkQueue::front()
    {
        boost::fibers::detail::spinlock_lock lk(m_work_spinlock);
        return &m_work_queue.front();
    }

    void WorkQueue::popFront()
    {
        boost::fibers::detail::spinlock_lock lk(m_work_spinlock);
        m_work_queue.pop_front();
    }

    size_t WorkQueue::size() const
    {
        boost::fibers::detail::spinlock_lock lk(m_work_spinlock);
        return m_work_queue.size();
    }

    bool WorkQueue::empty() const
    {
        boost::fibers::detail::spinlock_lock lk(m_work_spinlock);
        return m_work_queue.empty();
    }

    void WorkQueue::disable(const uint64_t id)
    {
        boost::fibers::detail::spinlock_lock lk(m_work_spinlock);
        for (auto itr = m_work_queue.begin(); itr != m_work_queue.end(); ++itr)
        {
            FiberProperty* prop = dynamic_cast<FiberProperty*>(itr->get_properties());
            if (prop)
            {
                if (prop->getId() == id)
                {
                    prop->disable();
                }
            }
        }
    }

    PriorityScheduler* WorkQueue::getScheduler() const
    {
        return m_scheduler;
    }

    void WorkQueue::setScheduler(PriorityScheduler* scheduler)
    {
        m_scheduler = scheduler;
    }

    struct WorkQueueImpl : WorkQueue
    {
        WorkQueueImpl(PriorityLevels priority, PriorityScheduler* scheduler = PriorityScheduler::current())
            : WorkQueue(priority, scheduler)
        {
        }
    };

    std::shared_ptr<WorkQueue> WorkQueue::create(PriorityLevels priority)
    {
        auto output = std::make_shared<WorkQueueImpl>(priority);
        return output;
    };

    std::shared_ptr<WorkQueue> WorkQueue::create(PriorityLevels priority, PriorityScheduler* scheduler)
    {
        auto output = std::make_shared<WorkQueueImpl>(priority, scheduler);
        return output;
    };

    static thread_local PriorityScheduler* g_current = nullptr;

    PriorityScheduler* PriorityScheduler::current()
    {
        return g_current;
    }

    void PriorityScheduler::setCurrent(PriorityScheduler* sched)
    {
        g_current = sched;
    }

    PriorityScheduler::PriorityScheduler(std::weak_ptr<ThreadPool> pool, const uint64_t wt)
        : m_pool(pool)
        , m_work_threshold(wt)
    {
        setCurrent(this);
        auto locked = pool.lock();
        if (locked)
        {
            locked->addScheduler(this);
        }

        MO_LOG(debug, "Instantiating scheduler");
        m_default_work_queue = WorkQueue::create(PriorityLevels::DEFAULT, this);
        m_default_work_queue->setScheduler(this);
    }

    PriorityScheduler::PriorityScheduler(std::weak_ptr<ThreadPool> pool, std::condition_variable** wakeup_cv)
        : m_pool(std::move(pool))
        , m_work_threshold(std::numeric_limits<uint64_t>::max())
        , m_is_worker(true)
    {
        setCurrent(this);
        *wakeup_cv = &m_cv;
        auto locked = m_pool.lock();
        if (locked)
        {
            locked->addScheduler(this);
        }
        MO_LOG(debug, "Instantiating worker scheduler");
        m_default_work_queue = WorkQueue::create(PriorityLevels::DEFAULT, this);
        m_default_work_queue->setScheduler(this);
    }

    PriorityScheduler::~PriorityScheduler()
    {
        m_assistant.reset();
        auto locked = m_pool.lock();
        if (locked)
        {
            locked->removeScheduler(this);
        }
    }

    void PriorityScheduler::awakened(boost::fibers::context* ctx, FiberProperty& props) noexcept
    {
        const auto id = props.getId();
        const bool pinned = ctx->is_context(boost::fibers::type::pinned_context);
        if (m_is_worker)
        {
            if (!pinned)
            {
                ctx->detach();
            }
        }

        IAsyncStreamPtr_t stream = props.getStream();
        boost::fibers::detail::spinlock_lock lk(m_work_queue_spinlock);
        if (id != 0)
        {
            m_default_work_queue->disable(id);
        }
        m_default_work_queue->pushBack(*ctx);
    }

    boost::fibers::context* PriorityScheduler::pick_next() noexcept
    {
        boost::fibers::context* victim = nullptr;

        if (!m_default_work_queue->empty())
        {
            victim = m_default_work_queue->front();
            m_default_work_queue->popFront();
            if (victim->get_scheduler() != boost::fibers::context::active()->get_scheduler())
            {
                boost::fibers::context::active()->attach(victim);
            }
        }
        return victim;
    }

    bool PriorityScheduler::has_ready_fibers() const noexcept
    {
        boost::fibers::detail::spinlock_lock lk1{m_work_queue_spinlock};
        return !m_default_work_queue->empty();
    }

    void PriorityScheduler::property_change(boost::fibers::context* ctx, FiberProperty& props) noexcept
    {
        if (!ctx->ready_is_linked())
        {
            return;
        }
        PriorityScheduler* scheduler = props.getScheduler();
        if (scheduler != this && scheduler != nullptr)
        {
            // Need to migrate the fiber to a different scheduler
            boost::fibers::detail::spinlock_lock lk1{m_work_queue_spinlock};
            // This removes the context from any existing work queues
            ctx->ready_unlink();
            scheduler->awakened(ctx, props);
            return;
        }
        {
            boost::fibers::detail::spinlock_lock lk1{m_work_queue_spinlock};
            // This removes the context from any existing work queues
            ctx->ready_unlink();
        }
        awakened(ctx, props);
    }

    void PriorityScheduler::suspend_until(std::chrono::steady_clock::time_point const& time_point) noexcept
    {
        if (m_is_worker)
        {
            if (std::chrono::steady_clock::time_point::max() == time_point)
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
} // namespace mo
