#include "AsyncStream.hpp"
#include "common.hpp"
#include "errors.hpp"

#include "MetaObject/core/AsyncStreamConstructor.hpp"
#include "MetaObject/core/AsyncStreamFactory.hpp"
#include <MetaObject/thread/FiberProperties.hpp>
#include <MetaObject/thread/FiberScheduler.hpp>

#include "MetaObject/logging/logging.hpp"
#include <MetaObject/logging/profiling.hpp>

#include <boost/fiber/fiber.hpp>
#include <boost/fiber/operations.hpp>

#include <cuda_runtime_api.h>

namespace mo
{
    namespace cuda
    {
        AsyncStream::AsyncStream(Stream stream, DeviceAllocator::Ptr_t allocator, Allocator::Ptr_t host_alloc)
            : mo::AsyncStream(std::move(host_alloc))
            , m_stream(std::move(stream))
            , m_event_pool(ObjectPool<CUevent_st>::create())
            , m_allocator(std::move(allocator))


        {
            init();
        }

        AsyncStream::~AsyncStream() = default;

        void AsyncStream::pushWork(std::function<void(mo::IAsyncStream*)>&& work)
        {
            Event event = this->createEvent();
            event.record(m_stream);
            // clang-format off
            auto callback = [work, event](mo::IAsyncStream* stream)
            {
                while(!event.queryCompletion())
                {
                    boost::this_fiber::sleep_for(std::chrono::nanoseconds(1000));
                }
                work(stream);
            };
            // clang-format on
            mo::AsyncStream::pushWork(std::move(callback));
        }

        void AsyncStream::pushEvent(std::function<void(mo::IAsyncStream*)>&& event, uint64_t event_id)
        {
            Event cuda_event = this->createEvent();
            cuda_event.record(m_stream);
            cuda_event.setCallback(
                [event, cuda_event](mo::IAsyncStream* stream) {
                    event(stream);
                },
                event_id);
        }

        void AsyncStream::init()
        {
            CHECK_CUDA_ERROR(&cudaGetDevice, &m_device_id);
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
            return mo::cuda::Event(m_event_pool);
        }

        void AsyncStream::enqueueCallback(std::function<void(mo::IAsyncStream*)> cb)
        {
            auto event = createEvent();
            event.record(m_stream);
            mo::AsyncStream::pushWork([event, cb](mo::IAsyncStream* stream) {
                event.synchronize();
                cb(stream);
            });
        }

        void AsyncStream::synchronize()
        {
            mo::ScopedProfile profile("mo::cuda::AsyncStream::synchronize");
            auto event = createEvent();
            event.record(m_stream);

            mo::AsyncStream::synchronize();
            event.synchronize();
        }

        void AsyncStream::synchronize(IDeviceStream* other)
        {
            auto typed = dynamic_cast<AsyncStream*>(other);
            if (typed)
            {
                auto event = createEvent();
                event.record(typed->m_stream);
                this->m_stream.waitEvent(event);
                // Not sure if this line is necessary
                mo::AsyncStream::pushWork([event](mo::IAsyncStream*) { event.synchronize(); });
            }
        }

        void AsyncStream::hostToDevice(ct::TArrayView<void> dst, ct::TArrayView<const void> src)
        {
            MO_ASSERT(dst.data() != nullptr);
            MO_ASSERT(src.data() != nullptr);
            MO_ASSERT_EQ(dst.size(), src.size());
            CHECK_CUDA_ERROR(&cudaMemcpyAsync, dst.data(), src.data(), src.size(), cudaMemcpyHostToDevice, m_stream);
        }

        void AsyncStream::deviceToHost(ct::TArrayView<void> dst, ct::TArrayView<const void> src)
        {
            MO_ASSERT(dst.data() != nullptr);
            MO_ASSERT(src.data() != nullptr);
            MO_ASSERT_EQ(dst.size(), src.size());
            CHECK_CUDA_ERROR(&cudaMemcpyAsync, dst.data(), src.data(), src.size(), cudaMemcpyDeviceToHost, m_stream);
        }

        void AsyncStream::deviceToDevice(ct::TArrayView<void> dst, ct::TArrayView<const void> src)
        {
            MO_ASSERT(dst.data() != nullptr);
            MO_ASSERT(src.data() != nullptr);
            MO_ASSERT_EQ(dst.size(), src.size());
            CHECK_CUDA_ERROR(&cudaMemcpyAsync, dst.data(), src.data(), src.size(), cudaMemcpyDeviceToDevice, m_stream);
        }

        void AsyncStream::hostToHost(ct::TArrayView<void> dst, ct::TArrayView<const void> src)
        {
            MO_ASSERT(dst.data() != nullptr);
            MO_ASSERT(src.data() != nullptr);
            MO_ASSERT_EQ(dst.size(), src.size());
            CHECK_CUDA_ERROR(&cudaMemcpyAsync, dst.data(), src.data(), src.size(), cudaMemcpyHostToHost, m_stream);
        }

        std::shared_ptr<DeviceAllocator> AsyncStream::deviceAllocator() const
        {
            return m_allocator;
        }

        bool AsyncStream::isDeviceStream() const
        {
            return true;
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
                    auto stream = std::make_shared<mo::cuda::AsyncStream>(Stream::create(),
                                                                          DeviceAllocator::getDefault(),
                                                                          Allocator::getDefault());
                    stream->setName(name);
                    stream->setHostPriority(thread_priority);
                    return stream;
                }
            };

            CUDAAsyncStreamConstructor g_ctr;
        } // namespace
    }     // namespace cuda
} // namespace mo
