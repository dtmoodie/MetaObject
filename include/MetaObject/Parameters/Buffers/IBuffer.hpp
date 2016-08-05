#pragma once
#include "MetaObject/Detail/Export.hpp"

namespace mo
{
    namespace Buffer
    {
        class MO_EXPORTS IBuffer
        {
        public:
            virtual ~IBuffer() {}
            virtual void SetSize(long long size = -1) = 0;
            // These are not const accessors because I may need to lock a mutex inside of them.
            virtual long long GetSize() = 0;
            virtual void GetTimestampRange(long long& start, long long& end) = 0;
        };
    }
}