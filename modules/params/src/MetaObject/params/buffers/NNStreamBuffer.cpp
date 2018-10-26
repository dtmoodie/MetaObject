#include "NNStreamBuffer.hpp"

namespace mo
{
    namespace buffer
    {
        NNStreamBuffer::NNStreamBuffer(const std::string& name)
        {
        }
        NNStreamBuffer::~NNStreamBuffer()
        {
        }

        BufferFlags NNStreamBuffer::getBufferType() const
        {
            return Type;
        }

        IDataContainerPtr_t NNStreamBuffer::search(const Header& hdr) const
        {
            IDataContainerPtr_t output;
            Map::modifyDataBuffer([this, &output, &hdr](const Map::Buffer_t& buffer) {
                if (!hdr.timestamp)
                {
                    if (!buffer.empty())
                    {
                        output = (--buffer.end())->second;
                        return;
                    }
                    else
                    {
                        return;
                    }
                }
                else
                {
                    auto upper = buffer.upper_bound(*hdr.timestamp);
                    auto lower = buffer.lower_bound(*hdr.timestamp);
                    if (upper != buffer.end() && lower != buffer.end())
                    {
                        if (*upper->first.timestamp - *hdr.timestamp < *lower->first.timestamp - *hdr.timestamp)
                        {
                            output = upper->second;
                            return;
                        }
                        else
                        {
                            output = lower->second;
                            return;
                        }
                    }
                    else if (lower != buffer.end())
                    {
                        output = lower->second;
                        return;
                    }
                    else if (upper != buffer.end())
                    {
                        output = upper->second;
                        return;
                    }
                }
                if (!buffer.empty())
                {
                    output = (--buffer.end())->second;
                }
            });
            return output;
        }
        static BufferConstructor<NNStreamBuffer> g_ctr_dropping_stream_buffer;
    }
}
