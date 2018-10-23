#pragma once
#include "MetaObject/core/detail/Enums.hpp"
#include "MetaObject/core/detail/Time.hpp"
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/params/InputParam.hpp"
namespace mo
{
    namespace buffer
    {
        class MO_EXPORTS IBuffer : public InputParam
        {
          public:
            virtual ~IBuffer();
            virtual void setFrameBufferCapacity(const uint64_t size) = 0;
            virtual void setTimePaddingCapacity(const Duration& time) = 0;
            virtual boost::optional<uint64_t> getFrameBufferCapacity() const = 0;
            virtual boost::optional<Duration> getTimePaddingCapacity() const = 0;

            // These are not const accessors because I may need to lock a mutex inside of them.
            virtual uint64_t getSize() const = 0;
            virtual bool getTimestampRange(mo::OptionalTime& start, mo::OptionalTime& end) = 0;
            virtual bool getFrameNumberRange(uint64_t& start, uint64_t& end) = 0;
            virtual BufferFlags getBufferType() const = 0;
        };
    }
}
