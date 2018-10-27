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
                        // first check if lower is exactly right
                        const auto ts_upper = *upper->first.timestamp;
                        auto ts_lower = *lower->first.timestamp;
                        // Since std::Map::lower_bound can return the exact item if the timestamps match
                        // or the first item not less than the desired timestamp, we first check
                        // if the timestamps are exact, if they are not we look at the item before lower
                        // to be able to accurately check items on both sides of the desired timestamp
                        // https://stackoverflow.com/questions/529831/returning-the-greatest-key-strictly-less-than-the-given-key-in-a-c-map
                        if (ts_lower == hdr.timestamp)
                        {
                            output = lower->second;
                            return;
                        }
                        else
                        {
                            if (lower != buffer.begin())
                            {
                                --lower;
                                ts_lower = *lower->first.timestamp;
                            }
                        }
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
