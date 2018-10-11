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
            virtual ~NNStreamBuffer();

            virtual BufferFlags getBufferType() const
            {
                return NEAREST_NEIGHBOR_BUFFER;
            }

          protected:
            typename std::map<Header, IDataContainerPtr_t>::iterator search(const OptionalTime_t& ts);
            typename std::map<Header, IDataContainerPtr_t>::iterator search(size_t fn);
        };
    }
}
