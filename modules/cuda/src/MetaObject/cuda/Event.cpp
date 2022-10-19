#include "Event.hpp"
#include "Stream.hpp"

#include <MetaObject/logging/logging.hpp>
#include <MetaObject/logging/profiling.hpp>

#include <MetaObject/thread/FiberProperties.hpp>
#include <MetaObject/thread/Mutex.hpp>

#include <MetaObject/cuda/errors.hpp>

#include <boost/fiber/operations.hpp>

#include <cuda_runtime_api.h>

#include <mutex>

namespace mo
{

    namespace cuda
    {

        struct Event::Impl
        {
            mutable std::shared_ptr<CUevent_st> m_event;
            std::function<void(mo::IAsyncStream*)> m_cb;
            std::weak_ptr<ObjectPool<CUevent_st>> m_event_pool;
            mutable std::atomic<bool> m_complete;

            ~Impl()
            {
                if (m_cb)
                {
                    synchronize();
                }
            }

            bool queryCompletion() const
            {
                if (m_complete)
                {
                    return m_complete;
                }

                if (m_event == nullptr)
                {
                    // We haven't recorded anything, so of course it is complete
                    return true;
                }
                const cudaError_t status = cudaEventQuery(m_event.get());
                if (status == cudaSuccess)
                {
                    m_complete = true;
                    m_event.reset();
                    auto cb = std::move(m_cb);
                    if (cb)
                    {
                        auto stream = mo::IAsyncStream::current();
                        cb(stream.get());
                    }

                    return m_complete;
                }
                else
                {
                    if (status == cudaErrorCudartUnloading)
                    {
                        // event is complete because cuda is being shut down
                        return true;
                    }
                    if (status != cudaErrorNotReady)
                    {
                        THROW(error, "cuda error {}", status);
                    }
                }
                return m_complete;
            }

            bool synchronize(Duration sleep = 1 * ns, const Duration timeout = 0 * ms) const
            {
                mo::ScopedProfile profile("mo::cuda::Event::synchronize");
                if (m_complete)
                {
                    return true;
                }
                if (m_event == nullptr)
                {
                    return true;
                }
                if (timeout == 0 * ms)
                {
                    while (!queryCompletion())
                    {
                        boost::this_fiber::sleep_for(sleep);
                    }
                    return m_complete;
                }
                auto now = Time::now();
                const auto end = now + timeout;

                while (now <= end && !queryCompletion())
                {
                    boost::this_fiber::sleep_for(sleep);
                    now = Time::now();
                }
                return m_complete;
            }
        };

        std::shared_ptr<CUevent_st> Event::create()
        {
            cudaEvent_t event = nullptr;
            CHECK_CUDA_ERROR(&cudaEventCreate, &event);

            return std::shared_ptr<CUevent_st>(event, [](cudaEvent_t ev) { CHECK_CUDA_ERROR(&cudaEventDestroy, ev); });
        }

        Event::Event(std::shared_ptr<ObjectPool<CUevent_st>> event_pool)
            : m_impl(new Impl())
        {
            m_impl->m_complete = false;
            m_impl->m_event_pool = event_pool;
        }

        Event::~Event() = default;

        void Event::record(Stream& stream)
        {
            cudaStream_t st = stream;
            record(st);
        }

        void Event::record(cudaStream_t stream)
        {
            MO_ASSERT(m_impl->m_event == nullptr);
            auto pool = m_impl->m_event_pool.lock();
            MO_ASSERT(pool != nullptr);
            m_impl->m_event = pool->get();
            CHECK_CUDA_ERROR(&cudaEventRecord, m_impl->m_event.get(), stream);
        }

        bool Event::queryCompletion() const
        {
            return m_impl->queryCompletion();
        }

        bool Event::synchronize(Duration sleep, const Duration timeout) const
        {
            return m_impl->synchronize(sleep, timeout);
        }

        Event::operator cudaEvent_t()
        {
            if (m_impl->m_event)
            {
                return m_impl->m_event.get();
            }
            return nullptr;
        }

        Event::operator constCudaEvent_t() const
        {
            if (m_impl->m_event)
            {
                return m_impl->m_event.get();
            }
            return nullptr;
        }

        void Event::setCallback(std::function<void(mo::IAsyncStream*)>&& cb, uint64_t event_id)
        {
            MO_ASSERT(m_impl->m_event != nullptr);
            m_impl->m_cb = std::move(cb);
            auto impl = m_impl;
            boost::fibers::fiber fib([impl]() { impl->synchronize(); });
            auto& prop = fib.properties<FiberProperty>();
            prop.setId(event_id);
            fib.detach();
        }
    } // namespace cuda
} // namespace mo
