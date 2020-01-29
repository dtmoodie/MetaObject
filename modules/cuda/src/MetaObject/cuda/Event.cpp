#include "Event.hpp"
#include "Stream.hpp"
#include <MetaObject/logging/logging.hpp>

#include <MetaObject/thread/fiber_include.hpp>

#include <cuda_runtime_api.h>

#include <mutex>

namespace mo
{

    namespace cuda
    {

        struct Event::Impl
        {
            mutable boost::fibers::mutex m_mtx;
            mutable std::shared_ptr<CUevent_st> m_event;
            std::function<void(void)> m_cb;
            ObjectPool<CUevent_st>* m_event_pool;
            volatile mutable bool m_complete;

            bool queryCompletion() const
            {
                std::lock_guard<boost::fibers::mutex> lock(m_mtx);
                if (m_complete)
                {
                    return m_complete;
                }

                MO_ASSERT(m_event != nullptr);
                if (cudaEventQuery(m_event.get()) == cudaSuccess)
                {
                    m_complete = true;
                    m_event.reset();
                }
                return m_complete;
            }

            bool synchronize(const Duration timeout = 0 * ms) const
            {
                if (m_complete)
                {
                    return true;
                }
                MO_ASSERT(m_event != nullptr);
                if (timeout == 0 * ms)
                {
                    while (!queryCompletion())
                    {
                        boost::this_fiber::yield();
                    }
                    return m_complete;
                }
                auto now = Time::now();
                const auto end = now + timeout;

                while (now <= end && !queryCompletion())
                {
                    boost::this_fiber::yield();
                    now = Time::now();
                }
                return m_complete;
            }
        };

        std::shared_ptr<CUevent_st> Event::create()
        {
            cudaEvent_t event = nullptr;
            cudaEventCreate(&event);
            return std::shared_ptr<CUevent_st>(event, [](cudaEvent_t ev) { cudaEventDestroy(ev); });
        }

        Event::Event(ObjectPool<CUevent_st>* event_pool)
            : m_impl(new Impl())
        {
            m_impl->m_complete = false;
            m_impl->m_event_pool = event_pool;
        }

        void Event::record(Stream& stream)
        {
            std::lock_guard<boost::fibers::mutex> lock(m_impl->m_mtx);
            MO_ASSERT(m_impl->m_event == nullptr);
            m_impl->m_event = m_impl->m_event_pool->get();
            cudaStream_t st = stream;
            cudaEventRecord(m_impl->m_event.get(), st);
        }

        bool Event::queryCompletion() const
        {
            return m_impl->queryCompletion();
        }

        bool Event::synchronize(const Duration timeout) const
        {
            return m_impl->synchronize(timeout);
        }

        Event::operator cudaEvent_t()
        {
            std::lock_guard<boost::fibers::mutex> lock(m_impl->m_mtx);
            if (m_impl->m_event)
            {
                return m_impl->m_event.get();
            }
            return nullptr;
        }

        Event::operator constCudaEvent_t() const
        {
            std::lock_guard<boost::fibers::mutex> lock(m_impl->m_mtx);
            if (m_impl->m_event)
            {
                return m_impl->m_event.get();
            }
            return nullptr;
        }

        void Event::setCallback(std::function<void(void)>&& cb)
        {
            std::lock_guard<boost::fibers::mutex> lock(m_impl->m_mtx);
            MO_ASSERT(m_impl->m_event != nullptr);
            MO_ASSERT(!m_impl->m_cb);
            m_impl->m_cb = std::move(cb);
            auto impl = m_impl;
            boost::fibers::fiber fib([impl]() {
                impl->synchronize();
                impl->m_cb();
            });
            fib.detach();
        }
    }
}
