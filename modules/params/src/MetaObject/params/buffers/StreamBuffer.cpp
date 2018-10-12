#include "StreamBuffer.hpp"
#include <boost/thread/recursive_mutex.hpp>

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

        void StreamBuffer::setTimePaddingCapacity(const Duration& time)
        {
            _time_padding = time;
        }

        boost::optional<size_t> StreamBuffer::getFrameBufferCapacity() const
        {
            return _frame_padding;
        }

        boost::optional<Duration> StreamBuffer::getTimePaddingCapacity() const
        {
            return _time_padding;
        }

        BufferFlags StreamBuffer::getBufferType() const
        {
            return Type;
        }

        void StreamBuffer::prune()
        {
            Lock lock(IParam::mtx());

            if (_current_timestamp && _time_padding)
            {
                auto itr = m_data_buffer.begin();
                while (itr != m_data_buffer.end())
                {
                    if (itr->first.timestamp && *itr->first.timestamp < mo::Time(*_current_timestamp - *_time_padding))
                    {
                        itr = m_data_buffer.erase(itr);
                    }
                    else
                    {
                        ++itr;
                    }
                }
            }
            if (_frame_padding && _current_frame_number > *_frame_padding)
            {
                auto itr = m_data_buffer.begin();
                while (itr != m_data_buffer.end())
                {
                    if (itr->first < (_current_frame_number - *_frame_padding))
                    {
                        itr = m_data_buffer.erase(itr);
                    }
                    else
                    {
                        ++itr;
                    }
                }
            }
        }

        IDataContainerPtr_t StreamBuffer::search(const Header& hdr)
        {
            Lock lock(mtx());
            auto ptr = Map::search(hdr);
            if (m_current_data)
            {
                prune();
            }
            return ptr;
        }
    }
}
