#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/Enums.hpp"
namespace mo
{
    namespace Buffer
    {
        class MO_EXPORTS IBuffer
        {
        public:
            virtual ~IBuffer() {}
            virtual void setFrameBufferCapacity(size_t size) = 0;
            virtual void setTimePaddingCapacity(mo::Time_t time) = 0;
            virtual boost::optional<size_t> getFrameBufferCapacity() = 0;
            virtual OptionalTime_t getTimePaddingCapacity() = 0;

            // These are not const accessors because I may need to lock a mutex inside of them.
            virtual size_t getSize() = 0;
            virtual bool getTimestampRange(mo::Time_t& start, mo::Time_t& end) = 0;
            virtual bool getFrameNumberRange(size_t& start, size_t& end) = 0;
            virtual ParamType getBufferType() const = 0;
        };
    }
}
