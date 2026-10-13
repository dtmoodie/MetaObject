#include "FiberScheduler.hpp"
#include <MetaObject/logging/logging.hpp>
#include <MetaObject/thread/ThreadPool.hpp>

#include <boost/context/detail/prefetch.hpp>
namespace mo
{
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

        boost::fibers::detail::spinlock_lock lk{m_work_spinlock};

        // Detach the context from its current worker-queue (if it is
        // attached to a scheduler) BEFORE the ready-queue insert, under
        // the same lock that guards the ready queue. boost.fiber's
        // worker_queue_ + the ctx's scheduler_ pointer carry no lock of
        // their own: fiber construction attaches a ctx on the creating
        // thread (boost fiber::start_ -> context::attach), and a stealer
        // otherwise unlinks stolen contexts on another thread. Detaching
        // here (same thread as the creating attach, before the insert)
        // leaves every ready-queue context with scheduler_ == nullptr, so
        // no thread except the queue owner ever touches its
        // worker_queue_ (invariant relied on by steal()).
        if (m_is_worker && !ctx->is_context(boost::fibers::type::pinned_context) && nullptr != ctx->get_scheduler())
        {
            ctx->detach();
        }

        if (id == 0)
        {
            auto itr =
                std::find_if(m_work_queue.begin(), m_work_queue.end(), [ctx_priority, this](boost::fibers::context& c) {
                    return properties(&c).getPriority() < ctx_priority;
                });

            m_work_queue.insert(itr, *ctx);
        }
        else
        {
            auto itr = std::find_if(
                m_work_queue.begin(), m_work_queue.end(), [ctx_priority, id, this](boost::fibers::context& c) {
                    return properties(&c).getPriority() < ctx_priority || properties(&c).getId() == id;
                });
            if (itr != m_work_queue.end())
            {
                if (properties(&*itr).getId() == id)
                {
                    itr = m_work_queue.insert(itr, *ctx);
                    ++itr;
                    BOOST_ASSERT(properties(&*itr).getId() == id);
                    BOOST_ASSERT(&*itr != ctx);
                    properties(&*itr).disable();
                }
            }
            else
            {
                m_work_queue.insert(itr, *ctx);
            }
        }

        const uint64_t size = m_work_queue.size();

        if (size > m_work_threshold && nullptr == m_assistant)
        {
            auto locked = m_pool.lock();
            if (locked)
            {
                m_assistant = locked->requestThread();
            }
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
        // Ready-queue contexts are detached on enqueue (awakened), so
        // scheduler_ is normally null here and there is nothing to
        // unlink. The check keeps steal safe (and null-deref-free) if a
        // context is ever enqueued while still attached.
        if (nullptr != victim->get_scheduler())
        {
            victim->detach();
        }
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
                    // Link into this scheduler's worker-queue under the
                    // work lock (the same lock guarding the ready queue
                    // the victim was just popped from and the lock
                    // steal()/awakened() use for worker-queue unlinks).
                    boost::fibers::detail::spinlock_lock lk{m_work_spinlock};
                    boost::fibers::context::active()->attach(victim);
                }
            }
        }
        else
        {
            std::uint32_t id = 0;
            auto locked = m_pool.lock();
            if (locked)
            {
                // Entries, not raw pointers: the list is copied out of the
                // pool lock and dereferenced after unlocking, so a worker
                // whose thread exits in between would otherwise leave a
                // stale PriorityScheduler* here (TOCTOU). Each entry checks
                // liveness under its own mutex and returns nullptr once its
                // scheduler has been retired (the worker retires its own
                // entry before its thread, and thus its scheduler, exits).
                const auto schedulers = locked->getSchedulers();
                if (schedulers.size() == 1)
                {
                    return victim;
                }
                const auto id_ = std::find_if(
                                     schedulers.begin(),
                                     schedulers.end(),
                                     [this](const std::shared_ptr<SchedulerEntry>& entry) {
                                         return entry->scheduler() == this;
                                     }) -
                                 schedulers.begin();
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
            }

            if (nullptr != victim)
            {
                boost::context::detail::prefetch_range(victim, sizeof(boost::fibers::context));
                BOOST_ASSERT(!victim->is_context(boost::fibers::type::pinned_context));
                boost::fibers::detail::spinlock_lock lk{m_work_spinlock};
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
        // The ready_is_linked() check and the unlink MUST be atomic with
        // steal()/pick_next()/awakened() on the SAME spinlock. They race
        // across threads: a context is inserted into m_work_queue by the
        // fiber constructor on thread A, and a worker on thread B may steal
        // it (pop + detach) at the same instant. If the check runs before
        // the steal and the unlink after it, ready_unlink() corrupts the
        // intrusive ready-list of a node that is no longer linked. (Seen as
        // a delayed SIGSEGV in context::~context during release_terminated_
        // on the other scheduler.)
        {
            boost::fibers::detail::spinlock_lock lk1{m_work_spinlock};
            if (!ctx->ready_is_linked())
            {
                return;
            }
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
                // Bounded park: an unbounded wait would deadlock the
                // dispatch loop's shutdown path (boost checks
                // worker_queue_.empty() every iteration to break out) if
                // a worker-linked context with a future sleep deadline
                // outlives it - only a loop iteration (sleep2ready_) can
                // drain the sleep queue, and a parked loop never iterates.
                // Periodic wakes keep that drain reachable at the cost of
                // one idle wakeup per interval.
                m_cv.wait_for(lk, std::chrono::milliseconds(500), [this]() { return m_flag; });
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

    void PriorityScheduler::releaseAssistant()
    {
        m_assistant.reset();
    }

    size_t PriorityScheduler::size() const
    {
        return m_work_queue.size();
    }
} // namespace mo
