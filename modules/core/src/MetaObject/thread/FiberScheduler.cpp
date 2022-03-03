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
        //scheduler->attachQueue(*this, priority);
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

    void WorkQueue::remove(const boost::fibers::context& ctx)
    {
        boost::fibers::detail::spinlock_lock lk(m_work_spinlock);
        m_work_queue.remove(ctx);
    }

    PriorityScheduler* WorkQueue::getScheduler() const
    {
        return m_scheduler;
    }

    void WorkQueue::setScheduler(PriorityScheduler* scheduler)
    {
        m_scheduler = scheduler;
    }

    struct WorkQueueImpl: WorkQueue
    {
        WorkQueueImpl(PriorityLevels priority, PriorityScheduler* scheduler = PriorityScheduler::current()):
            WorkQueue(priority, scheduler)
        {

        }
    };

    std::shared_ptr<WorkQueue> WorkQueue::create(PriorityLevels priority)
    {
        auto output = std::make_shared<WorkQueueImpl>(priority);
        output->m_scheduler->attachQueue(*output, priority);
        return output;
    };

    std::shared_ptr<WorkQueue> WorkQueue::create(PriorityLevels priority, PriorityScheduler* scheduler)
    {
        auto output = std::make_shared<WorkQueueImpl>(priority, scheduler);
        scheduler->attachQueue(*output, priority);
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

    PriorityScheduler::PriorityScheduler(std::weak_ptr<ThreadPool> pool, const uint64_t wt, std::shared_ptr<WorkQueue>* work_queue)
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
        m_prioritized_work_queues.resize(PriorityLevels::HIGHEST+1);
        m_work_queue_index.resize(PriorityLevels::HIGHEST+1, 0);
        m_default_work_queue = WorkQueue::create(PriorityLevels::DEFAULT, this);
        m_default_work_queue->setScheduler(this);

        if(work_queue)
        {
            *work_queue = m_default_work_queue;
        }
    }

    PriorityScheduler::PriorityScheduler(std::weak_ptr<ThreadPool> pool, std::condition_variable** wakeup_cv, std::shared_ptr<WorkQueue>* work_queue)
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
        m_prioritized_work_queues.resize(PriorityLevels::HIGHEST+1);
        m_work_queue_index.resize(PriorityLevels::HIGHEST+1, 0);
        m_default_work_queue = WorkQueue::create(PriorityLevels::DEFAULT, this);
        m_default_work_queue->setScheduler(this);
        //m_prioritized_work_queues[PriorityLevels::DEFAULT].push_back(m_default_work_queue);

        if(work_queue)
        {
            *work_queue = m_default_work_queue;
        }

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

        std::shared_ptr<IAsyncStream> stream = props.getStream();
        WorkQueue* queue = nullptr;
        {
            boost::fibers::detail::spinlock_lock lk(m_work_queue_spinlock);
            if(stream)
            {
                auto queue_ = stream->getWorkQueue();
                if(queue_)
                {
                    if(queue_->getScheduler() == this)
                    {
                        // Need to check if we've already put this context in the default work queue
                        m_default_work_queue->remove(*ctx);
                        queue = queue_.get();
                    }else
                    {
                        m_default_work_queue->remove(*ctx);
                        queue_->pushBack(*ctx);
                        queue_->getScheduler()->notify();
                        return;
                    }
                }
            }else
            {
                queue = m_default_work_queue.get();
            }
        }
        if(queue)
        {
            if(id != 0)
            {
                queue->disable(id);
            }
            queue->pushBack(*ctx);
        }
    }

    boost::fibers::context* PriorityScheduler::checkQueue(int32_t priority, int32_t index)
    {
        boost::fibers::context* victim = nullptr;
        auto queue = m_prioritized_work_queues[priority][index].lock();
        if(queue)
        {
            const size_t size = queue->size();
            if(size > 0)
            {
                victim = queue->front();
                queue->popFront();
                return victim;
            }
        }
        return nullptr;
    }

    boost::fibers::context* PriorityScheduler::pick_next() noexcept
    {
        boost::fibers::context* victim = nullptr;
        {
            boost::fibers::detail::spinlock_lock lk{m_work_queue_spinlock};
            // look for work from the highest priority work queue first
            for(int32_t priority = PriorityLevels::HIGHEST; priority >= PriorityLevels::LOWEST; --priority)
            {
                const size_t num_queues = m_prioritized_work_queues[priority].size();
                if(num_queues > 0)
                {
                    // To implement round robin we want to start with the next queue in the list
                    //        q1, q2, q3, q4
                    // t0     |
                    // t1         |
                    // t2             |
                    // t3                 |
                    // However if we find a queue is empty, we want to skip it and move on to the next queue
                    // within this call, we can't just wait for pick_next to be called again because that causes an error in boost

                    int32_t& idx = m_work_queue_index[priority];
                    const int32_t last_queue_idx = idx;
                    // Start looking at the next queue
                    ++idx;
                    if(idx >= num_queues)
                    {
                        // loop back to beginning
                        idx = 0;
                    }
                    victim = checkQueue(priority, idx);
                    if(victim)
                    {
                        return victim;
                    }else
                    {
                        // we did not find a task, now we need to search for work
                        while(idx != last_queue_idx)
                        {
                            ++idx;
                            if(idx >= num_queues)
                            {
                                // loop back to beginning
                                idx = 0;
                            }
                            victim = checkQueue(priority, idx);
                            if(victim)
                            {
                                return victim;
                            }
                        }
                        // idx == last_queue_idx
                        victim = checkQueue(priority, idx);
                        if(victim)
                        {
                            return victim;
                        }
                    }
                }
            }
        }

        return nullptr;
    }

    bool PriorityScheduler::has_ready_fibers() const noexcept
    {
        boost::fibers::detail::spinlock_lock lk1{m_work_queue_spinlock};
        for(auto priority_iterator = m_prioritized_work_queues.begin(); priority_iterator != m_prioritized_work_queues.end(); ++priority_iterator)
        {
            for(auto queue_iterator = priority_iterator->begin(); queue_iterator != priority_iterator->end(); ++queue_iterator)
            {
                auto queue = queue_iterator->lock();
                if(queue && !queue->empty())
                {
                    return true;
                }
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

    void PriorityScheduler::attachQueue(WorkQueue& queue, PriorityLevels priority)
    {
        auto queue_ptr = queue.shared_from_this();
        PriorityScheduler*  scheduler = queue.getScheduler();
        if(scheduler != nullptr && queue.getScheduler() != this)
        {
            queue.getScheduler()->removeQueue(queue);
        }

        auto queue_predicate = [queue_ptr](const std::weak_ptr<WorkQueue>& it) -> bool
        {
            return it.lock() == queue_ptr;
        };

        boost::fibers::detail::spinlock_lock lk1{m_work_queue_spinlock};
        if(std::find_if(m_prioritized_work_queues[priority].begin(), m_prioritized_work_queues[priority].end(), queue_predicate) == m_prioritized_work_queues[priority].end())
        {
            // Queue is not in our prioritized list
            // Sorted order of work queues
            auto priority_predicate =
                [priority](const std::pair<PriorityLevels, std::weak_ptr<WorkQueue>>& pair)
            {
                return pair.first < priority;
            };
            m_prioritized_work_queues[priority].push_back(queue_ptr);
            queue.setScheduler(this);
        }
    }

    void PriorityScheduler::removeQueue(WorkQueue& queue)
    {
        if(queue.getScheduler() != this)
        {
            return;
        }
        auto queue_ptr = queue.shared_from_this();
        auto queue_predicate = [queue_ptr](const std::weak_ptr<WorkQueue>& it) -> bool
        {
            return it.lock() == queue_ptr;
        };
        boost::fibers::detail::spinlock_lock lk1{m_work_queue_spinlock};
        for(int32_t priority = PriorityLevels::LOWEST; priority <= PriorityLevels::HIGHEST; ++priority)
        {
            auto itr = std::find_if(m_prioritized_work_queues[priority].begin(), m_prioritized_work_queues[priority].end(), queue_predicate);
            if(itr != m_prioritized_work_queues[priority].end())
            {
                queue_ptr->setScheduler(nullptr);
                m_prioritized_work_queues[priority].erase(itr);
                return;
            }
        }
    }


} // namespace mo
