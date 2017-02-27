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
            virtual void SetSize(size_t size = -1) = 0;
            virtual void SetSize(mo::time_t time) = 0;

            // These are not const accessors because I may need to lock a mutex inside of them.
            virtual size_t GetSize() = 0;
            virtual bool GetTimestampRange(mo::time_t& start, mo::time_t& end) = 0;
            virtual bool GetFrameNumberRange(size_t& start, size_t& end) = 0;
            virtual ParameterTypeFlags GetBufferType() const = 0;
        };
    }
}
