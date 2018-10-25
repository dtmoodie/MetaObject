#pragma once
#include "IBuffer.hpp"
#include "Map.hpp"
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/params/ITInputParam.hpp"
#include <boost/thread/condition_variable.hpp>
namespace mo
{
    namespace buffer
    {

        class MO_EXPORTS StreamBuffer : public Map
        {
          public:
            static const BufferFlags Type = STREAM_BUFFER;

            StreamBuffer(const std::string& name = "");

            virtual void setFrameBufferCapacity(const uint64_t size) override;
            virtual void setTimePaddingCapacity(const Duration& time) override;
            virtual boost::optional<size_t> getFrameBufferCapacity() const override;
            virtual boost::optional<Duration> getTimePaddingCapacity() const override;

            virtual BufferFlags getBufferType() const override;

            virtual IContainerPtr_t getData(const Header& desired = Header()) override;
            virtual IContainerConstPtr_t getData(const Header& desired = Header()) const override;

          protected:
            virtual uint32_t prune(Map::Buffer_t& buffer);
            OptionalTime _current_timestamp;
            uint64_t _current_frame_number;
            boost::optional<Duration> _time_padding;
            boost::optional<uint64_t> _frame_padding;
        };

        class MO_EXPORTS BlockingStreamBuffer : public StreamBuffer
        {
          public:
            static const BufferFlags Type = BLOCKING_STREAM_BUFFER;

            BlockingStreamBuffer(const std::string& name = "");
            virtual void setFrameBufferCapacity(size_t size) override;
            virtual BufferFlags getBufferType() const override;

          protected:
            virtual uint32_t prune(Map::Buffer_t&) override;
            void onInputUpdate(const IDataContainerPtr_t&, IParam*, UpdateFlags) override;
            size_t _size;
            boost::condition_variable_any _cv;
        };

        class MO_EXPORTS DroppingStreamBuffer : public BlockingStreamBuffer
        {
          public:
            static const BufferFlags Type = DROPPING_STREAM_BUFFER;

            DroppingStreamBuffer(const std::string& name = "");
            virtual BufferFlags getBufferType() const override;

          protected:
            void onInputUpdate(const IDataContainerPtr_t&, IParam*, UpdateFlags) override;
        };
    }
}
