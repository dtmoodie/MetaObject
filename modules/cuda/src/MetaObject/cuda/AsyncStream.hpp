#pragma once

#include "Event.hpp"
#include "Stream.hpp"

#include <MetaObject/core/AsyncStream.hpp>
#include <MetaObject/core/detail/ObjectPool.hpp>
#include <MetaObject/thread/PriorityLevels.hpp>

namespace mo
{
    namespace cuda
    {

        struct MO_EXPORTS IAsyncStream : virtual public mo::IDeviceStream
        {
            virtual Stream getStream() const = 0;
            virtual operator cudaStream_t() = 0;
            virtual void setStream(const Stream& stream) = 0;
            virtual void setDevicePriority(PriorityLevels lvl) = 0;
            virtual void enqueueCallback(std::function<void(mo::IAsyncStream*)> cb) = 0;

        }; // struct mo::cuda::Context

        struct MO_EXPORTS AsyncStream : virtual public cuda::IAsyncStream, virtual public mo::AsyncStream
        {
            AsyncStream(Stream stream = Stream::create(),
                        DeviceAllocator::Ptr_t allocator = DeviceAllocator::getDefault(),
                        Allocator::Ptr_t host_alloc = Allocator::getDefault());

            AsyncStream(const AsyncStream&) = delete;
            AsyncStream(AsyncStream&&) = delete;
            AsyncStream& operator=(const AsyncStream&) = delete;
            AsyncStream& operator=(AsyncStream&&) = delete;

            ~AsyncStream() override;

            void setStream(const Stream& stream) override;
            Stream getStream() const override;
            operator cudaStream_t() override;

            void pushWork(std::function<void(mo::IAsyncStream*)>&& work, const bool async = false) override;
            void pushEvent(std::function<void(mo::IAsyncStream*)>&& event, uint64_t event_id = 0) override;

            Event createEvent();
            void enqueueCallback(std::function<void(mo::IAsyncStream*)> cb) override;

            // This differs from Stream::synchronize in that it uses an event and boost::fibers to allow other fibers to
            // execute while the stream is waiting for completion.  Thus this should be more efficient than just
            // getStream().synchronize();
            void synchronize();
            void synchronize(Duration sleep) override;
            void synchronize(IDeviceStream* other) override;

            void setName(const std::string& name) override;

            void setDevicePriority(PriorityLevels lvl) override;

            void hostToDevice(ct::TArrayView<void> dst, ct::TArrayView<const void> src) override;
            void deviceToHost(ct::TArrayView<void> dst, ct::TArrayView<const void> src) override;
            void deviceToDevice(ct::TArrayView<void> dst, ct::TArrayView<const void> src) override;
            void hostToHost(ct::TArrayView<void> dst, ct::TArrayView<const void> src) override;

            std::shared_ptr<DeviceAllocator> deviceAllocator() const override;
            bool isDeviceStream() const override;

          private:
            void init();
            Stream m_stream;
            std::shared_ptr<ObjectPool<CUevent_st>> m_event_pool;
            int m_device_id = -1;
            std::shared_ptr<DeviceAllocator> m_allocator;
            // flag to indicate if a stream was accessed since the last synchronize
            mutable std::atomic<bool> m_stream_accessed;
        }; // struct mo::cuda::Context

    } // namespace cuda
} // namespace mo
