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
            static const ParamType Type = StreamBuffer_e;

            StreamBuffer(const std::string& name = "");

            virtual void setFrameBufferCapacity(const uint64_t size) override;
            virtual void setTimePaddingCapacity(const mo::Time_t& time) override;
            virtual boost::optional<size_t> getFrameBufferCapacity() const override;
            virtual OptionalTime_t getTimePaddingCapacity() const override;

            virtual ParamType getBufferType() const
            {
                return StreamBuffer_e;
            }

          protected:
            virtual void prune();
            OptionalTime_t _current_timestamp;
            size_t _current_frame_number;
            OptionalTime_t _time_padding;
            boost::optional<size_t> _frame_padding;
        };

        class MO_EXPORTS BlockingStreamBuffer : public StreamBuffer
        {
          public:
            static const ParamType Type = BlockingStreamBuffer_e;

            BlockingStreamBuffer(const std::string& name = "");
            virtual void setFrameBufferCapacity(size_t size);
            virtual ParamType getBufferType() const
            {
                return BlockingStreamBuffer_e;
            }

          protected:
            virtual void prune();
            size_t _size;
            boost::condition_variable_any _cv;
        };

        class MO_EXPORTS DroppingStreamBuffer : public BlockingStreamBuffer
        {
          public:
            static const ParamType Type = DroppingStreamBuffer_e;

            DroppingStreamBuffer(const std::string& name = "");
            virtual ParamType getBufferType() const
            {
                return Type;
            }

          protected:
        };
    }
}
