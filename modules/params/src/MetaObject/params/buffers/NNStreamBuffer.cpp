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
                    const auto ts = *hdr.timestamp;
                    auto upper = buffer.upper_bound(hdr);
                    auto lower = buffer.lower_bound(hdr);
                    if (upper != buffer.end() && lower != buffer.end())
                    {
                        const auto ts_upper = *upper->first.timestamp;
                        const auto ts_lower = *lower->first.timestamp;
                        const auto upperdelta = std::chrono::abs(ts_upper - ts);
                        const auto lowerdelta = std::chrono::abs(ts_lower - ts);
                        if (upperdelta < lowerdelta)
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
