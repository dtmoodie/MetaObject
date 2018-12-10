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

        struct IAsyncStream : virtual public mo::IAsyncStream
        {
            virtual Stream getStream() const = 0;
            virtual void setStream(const Stream& stream) = 0;
            virtual void setDevicePriority(const PriorityLevels lvl) = 0;

        }; // struct mo::cuda::Context

        struct AsyncStream : virtual public cuda::IAsyncStream, public mo::AsyncStream
        {
            AsyncStream();
            AsyncStream(const Stream& stream);
            ~AsyncStream() override;

            Stream getStream() const override;
            Event createEvent();

            void setName(const std::string& name) override;

            void setDevicePriority(const PriorityLevels lvl) override;

          protected:
            AsyncStream(TypeInfo type);

          private:
            void init();
            Stream m_stream;
            ObjectPool<CUevent_st> m_event_pool;
            int m_device_id = -1;
        }; // struct mo::cuda::Context

    } // namespace mo::cuda
} // namespace mo
