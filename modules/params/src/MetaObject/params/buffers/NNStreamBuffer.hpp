#pragma once
#include "StreamBuffer.hpp"

namespace mo
{
    namespace buffer
    {
        class MO_EXPORTS NNStreamBuffer : public StreamBuffer
        {
          public:
            NNStreamBuffer(const std::string& name = "");



            virtual ParamType getBufferType() const
            {
                return NNStreamBuffer_e;
            }

          protected:
            typename std::map<SequenceKey, InputStorage_t>::iterator search(const OptionalTime_t& ts);
            typename std::map<SequenceKey, InputStorage_t>::iterator search(size_t fn);
        };
    }
}
#include "detail/NNStreamBufferImpl.hpp"
