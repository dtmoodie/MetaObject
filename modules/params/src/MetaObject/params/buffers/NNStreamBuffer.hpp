#pragma once
#include "StreamBuffer.hpp"

namespace mo
{
    namespace buffer
    {
        class MO_EXPORTS NNStreamBuffer : public StreamBuffer
        {
          public:
            static const BufferFlags Type = NEAREST_NEIGHBOR_BUFFER;
            NNStreamBuffer(const std::string& name = "");
            virtual ~NNStreamBuffer();

            virtual BufferFlags getBufferType() const;

          protected:
            virtual IDataContainerPtr_t search(const Header& hdr) const;
        };
    }
}
