#include "IBuffer.hpp"
#include <MetaObject/params/buffers/Map.hpp>

namespace mo
{
    namespace buffer
    {

        std::shared_ptr<IBuffer> IBuffer::create(BufferFlags type)
        {
            switch (type)
            {
            case BufferFlags::MAP_BUFFER:
                return std::make_shared<Map>("", Map::GROW, Map::EXACT);
            case BufferFlags::STREAM_BUFFER:
                return std::make_shared<Map>("", Map::PRUNE, Map::EXACT);
            case BufferFlags::BLOCKING_STREAM_BUFFER:
                return std::make_shared<Map>("", Map::BLOCK, Map::EXACT);
            case BufferFlags::DROPPING_STREAM_BUFFER:
                return std::make_shared<Map>("", Map::DROP, Map::EXACT);
            case BufferFlags::NEAREST_NEIGHBOR_BUFFER:
                return std::make_shared<Map>("", Map::BLOCK, Map::NEAREST);
            default:
                return {};
            }
        }

        IBuffer::~IBuffer()
        {
        }
    } // namespace buffer
} // namespace mo
