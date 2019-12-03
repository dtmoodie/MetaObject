#include "AsyncStream.hpp"
#include "common.hpp"

#include "MetaObject/core/AsyncStreamConstructor.hpp"
#include "MetaObject/core/AsyncStreamFactory.hpp"
#include <MetaObject/thread/FiberProperties.hpp>
#include <MetaObject/thread/FiberScheduler.hpp>

#include "MetaObject/logging/logging.hpp"
#include <MetaObject/logging/profiling.hpp>
#include <cuda_runtime_api.h>

namespace mo
{
    namespace cuda
    {
        AsyncStream::AsyncStream(Stream stream, DeviceAllocator::Ptr_t allocator, Allocator::Ptr_t host_alloc)
            : mo::AsyncStream(std::move(host_alloc))
            , m_stream(std::move(stream))
            , m_allocator(std::move(allocator))

        {
            init();
        }

        AsyncStream::~AsyncStream() = default;

        void AsyncStream::pushWork(std::function<void(void)>&& work, PriorityLevels priority)
        {
            boost::fibers::fiber fiber(std::move(work));
            auto& prop = fiber.properties<FiberProperty>();
            prop.setPriority(priority == NONE ? hostPriority() : priority);
            prop.setStream(this->shared_from_this());
            prop.setDeviceStream(std::dynamic_pointer_cast<IDeviceStream>(shared_from_this()));
            fiber.detach();
        }

        void AsyncStream::pushEvent(std::function<void(void)>&& event, uint64_t event_id, PriorityLevels priority)
        {
            boost::fibers::fiber fiber([event]() {
                if (boost::this_fiber::properties<FiberProperty>().enabled())
                {
                    event();
                }
            });

            auto& prop = fiber.properties<FiberProperty>();
            prop.setAll(priority == NONE ? hostPriority() : priority, event_id, false);
            prop.setStream(this->shared_from_this());
            prop.setDeviceStream(std::dynamic_pointer_cast<IDeviceStream>(shared_from_this()));
            fiber.detach();
        }

        void AsyncStream::init()
        {
            CUDA_ERROR_CHECK(cudaGetDevice(&m_device_id));
        }

        void AsyncStream::setName(const std::string& name)
        {
            setStreamName(name.c_str(), getStream());
            mo::AsyncStream::setName(name);
        }

        void AsyncStream::setDevicePriority(const PriorityLevels lvl)
        {
            m_stream = Stream::create(lvl);
        }

        void AsyncStream::setStream(const Stream& stream)
        {
            m_stream = stream;
        }

        Stream AsyncStream::getStream() const
        {
            return m_stream;
        }

        AsyncStream::operator cudaStream_t()
        {
            return m_stream;
        }

        Event AsyncStream::createEvent()
        {
            return mo::cuda::Event(&m_event_pool);
        }

        void AsyncStream::enqueueCallback(std::function<void(void)> cb)
        {
            auto event = createEvent();
            event.record(m_stream);
            pushWork([event, cb] {
                event.synchronize();
                cb();
            });
        }

        void AsyncStream::synchronize()
        {
            auto event = createEvent();
            event.record(m_stream);

            const auto sched = PriorityScheduler::current();
            auto size = sched->size();
            while (size != 1)
            {
                boost::this_fiber::yield();
                size = sched->size();
            }

            while (!event.queryCompletion())
            {
                boost::this_fiber::yield();
            }
        }

        void AsyncStream::synchronize(IDeviceStream* other)
        {
            auto typed = dynamic_cast<AsyncStream*>(other);
            if (typed)
            {
                auto event = createEvent();
                event.record(m_stream);
                typed->m_stream.waitEvent(event);
                pushWork([event]() { event.synchronize(); });
            }
        }

        void AsyncStream::hostToDevice(ct::TArrayView<void> dst, ct::TArrayView<const void> src)
        {
            MO_ASSERT(dst.data() != nullptr);
            MO_ASSERT(src.data() != nullptr);
            MO_ASSERT_EQ(dst.size(), src.size());
            CUDA_ERROR_CHECK(cudaMemcpyAsync(dst.data(), src.data(), src.size(), cudaMemcpyHostToDevice, m_stream));
        }

        void AsyncStream::deviceToHost(ct::TArrayView<void> dst, ct::TArrayView<const void> src)
        {
            MO_ASSERT(dst.data() != nullptr);
            MO_ASSERT(src.data() != nullptr);
            MO_ASSERT_EQ(dst.size(), src.size());
            CUDA_ERROR_CHECK(cudaMemcpyAsync(dst.data(), src.data(), src.size(), cudaMemcpyDeviceToHost, m_stream));
        }

        void AsyncStream::deviceToDevice(ct::TArrayView<void> dst, ct::TArrayView<const void> src)
        {
            MO_ASSERT(dst.data() != nullptr);
            MO_ASSERT(src.data() != nullptr);
            MO_ASSERT_EQ(dst.size(), src.size());
            CUDA_ERROR_CHECK(cudaMemcpyAsync(dst.data(), src.data(), src.size(), cudaMemcpyDeviceToDevice, m_stream));
        }

        void AsyncStream::hostToHost(ct::TArrayView<void> dst, ct::TArrayView<const void> src)
        {
            MO_ASSERT(dst.data() != nullptr);
            MO_ASSERT(src.data() != nullptr);
            MO_ASSERT_EQ(dst.size(), src.size());
            CUDA_ERROR_CHECK(cudaMemcpyAsync(dst.data(), src.data(), src.size(), cudaMemcpyHostToHost, m_stream));
        }

        std::shared_ptr<DeviceAllocator> AsyncStream::deviceAllocator() const
        {
            return m_allocator;
        }

        namespace
        {
            struct CUDAAsyncStreamConstructor : public AsyncStreamConstructor
            {
                CUDAAsyncStreamConstructor()
                {
                    SystemTable::dispatchToSystemTable(
                        [this](SystemTable* table) { AsyncStreamFactory::instance(table)->registerConstructor(this); });
                }

                uint32_t priority(int32_t) override
                {
                    return 2U;
                }

                Ptr_t create(const std::string& name, int32_t, PriorityLevels, PriorityLevels thread_priority) override
                {
                    auto stream = std::make_shared<mo::cuda::AsyncStream>();
                    stream->setName(name);
                    stream->setHostPriority(thread_priority);
                    return stream;
                }
            };

            CUDAAsyncStreamConstructor g_ctr;
        }
    }
}
