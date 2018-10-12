#include "StreamBuffer.hpp"

namespace mo
{
    namespace buffer
    {
        StreamBuffer::StreamBuffer(const std::string& name)
        {
        }

        void StreamBuffer::setFrameBufferCapacity(const uint64_t size)
        {
            _frame_padding = size;
        }

        void StreamBuffer::setTimePaddingCapacity(const mo::Time_t& time)
        {
            _time_padding = time;
        }

        boost::optional<size_t> StreamBuffer::getFrameBufferCapacity() const
        {
            return _frame_padding;
        }

        OptionalTime_t StreamBuffer::getTimePaddingCapacity() const
        {
            return _time_padding;
        }

        BufferFlags StreamBuffer::getBufferType() const
        {
            return Type;
        }

        void StreamBuffer::prune()
        {
        }

        StreamBuffer::IContainerPtr_t StreamBuffer::getData(const Header& desired)
        {
        }

        StreamBuffer::IContainerConstPtr_t StreamBuffer::getData(const Header& desired) const
        {
        }
    }
}
