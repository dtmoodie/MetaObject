#include "FiberScheduler.hpp"
#include <MetaObject/logging/logging.hpp>
#include <MetaObject/thread/ThreadPool.hpp>

#include <boost/context/detail/prefetch.hpp>
namespace mo
{

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
        for(auto itr = m_work_queue.begin(); itr != m_work_queue.end(); ++itr)
        {
            FiberProperty* prop = dynamic_cast<FiberProperty*>(itr->get_properties());
            if(prop)
            {
                if(prop->getId() == id)
                {
                    prop->disable();
                }
            }

        }
    }


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
        m_default_work_queue = std::make_shared<WorkQueue>();
        m_prioritized_work_queues.push_back(std::make_pair(PriorityLevels::DEFAULT, m_default_work_queue));
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
        m_default_work_queue = std::make_shared<WorkQueue>();
        m_prioritized_work_queues.push_back(std::make_pair(PriorityLevels::DEFAULT, m_default_work_queue));

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
        const auto ctx_priority = props.getPriority();
        const auto id = props.getId();
        if (m_is_worker)
        {
            if (!ctx->is_context(boost::fibers::type::pinned_context))
            {
                ctx->detach();
            }
        }

        std::shared_ptr<IAsyncStream> stream = props.getStream();
        WorkQueue* queue = nullptr;
        {
            boost::fibers::detail::spinlock_lock lk(m_work_queue_spinlock);
            if(stream)
            {
                auto queue_ = stream->getWorkQueue();
                auto pred = [queue_](const std::pair<PriorityLevels, std::weak_ptr<WorkQueue>>& it) -> bool
                {
                    return it.second.lock() == queue_;
                };
                if(std::find_if(m_prioritized_work_queues.begin(),
                                m_prioritized_work_queues.end(),
                                pred
                                ) == m_prioritized_work_queues.end())
                {
                    // Queue is not in our prioritized list
                    // Sorted order of work queues
                    std::pair<PriorityLevels, std::weak_ptr<WorkQueue>> pair = std::make_pair(ctx_priority, queue_);
                    auto itr = std::find_if(m_prioritized_work_queues.begin(),
                                 m_prioritized_work_queues.end(),
                                 [ctx_priority](const std::pair<PriorityLevels, std::weak_ptr<WorkQueue>>& pair)
                                 {
                                    return pair.first < ctx_priority;
                                 });
                    m_prioritized_work_queues.insert(itr, std::move(pair));
                }
                queue = queue_.get();
            }else
            {
                queue = m_default_work_queue.get();
            }
        }
        MO_ASSERT(queue != nullptr);
        if(id != 0)
        {
            queue->disable(id);
        }
        queue->pushBack(*ctx);
    }

    boost::fibers::context* PriorityScheduler::pick_next() noexcept
    {
        boost::fibers::context* victim = nullptr;
        {
            boost::fibers::detail::spinlock_lock lk{m_work_queue_spinlock};
            for(auto itr = m_prioritized_work_queues.begin(); itr != m_prioritized_work_queues.end(); ++itr)
            {
                auto queue = itr->second.lock();
                if(queue && !queue->empty())
                {
                    victim = queue->front();
                    queue->popFront();
                    // We no longer check for enabled / disable here because we cannot cleanup a context that we dont
                    // want to run instead events that are disabled just don't execute the functor
                    return victim;
                }
            }
        }

        return nullptr;
    }

    bool PriorityScheduler::has_ready_fibers() const noexcept
    {
        boost::fibers::detail::spinlock_lock lk1{m_work_queue_spinlock};
        for(auto itr = m_prioritized_work_queues.begin(); itr != m_prioritized_work_queues.end(); ++itr)
        {
            auto queue = itr->second.lock();
            if(queue && !queue->empty())
            {
                return true;
            }
        }
        return false;
    }

    void PriorityScheduler::property_change(boost::fibers::context* ctx, FiberProperty& props) noexcept
    {
        if (!ctx->ready_is_linked())
        {
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
