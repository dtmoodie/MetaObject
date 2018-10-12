#include "CircularBuffer.hpp"
#include "BufferConstructor.hpp"
#include <boost/thread/locks.hpp>
#include <boost/thread/recursive_mutex.hpp>

namespace mo
{
    namespace buffer
    {
        CircularBuffer::CircularBuffer()
        {
        }

        void CircularBuffer::setFrameBufferCapacity(uint64_t size)
        {
            m_buffer.set_capacity(size);
        }

        void CircularBuffer::setTimePaddingCapacity(const mo::Time_t&)
        {
        }

        boost::optional<uint64_t> CircularBuffer::getFrameBufferCapacity() const
        {
            return m_buffer.capacity();
        }

        OptionalTime_t CircularBuffer::getTimePaddingCapacity() const
        {
            return {};
        }

        uint64_t CircularBuffer::getSize() const
        {
            return m_buffer.size();
        }

        bool CircularBuffer::getTimestampRange(mo::OptionalTime_t& start, mo::OptionalTime_t& end)
        {
            if (m_buffer.size())
            {
                start = m_buffer.front()->getHeader().timestamp;
                end = m_buffer.back()->getHeader().timestamp;
                return true;
            }
            return false;
        }

        bool CircularBuffer::getFrameNumberRange(uint64_t& start, uint64_t& end)
        {
            if (m_buffer.size())
            {
                start = m_buffer.front()->getHeader().frame_number;
                end = m_buffer.back()->getHeader().frame_number;
                return true;
            }
            return false;
        }

        BufferFlags CircularBuffer::getBufferType() const
        {
            return Type;
        }

        void CircularBuffer::onInputUpdate(const std::shared_ptr<IDataContainer>& data, IParam*, UpdateFlags)
        {
            if (data)
            {
                Lock lock(mtx());
                m_buffer.push_back(data);
            }
        }

        CircularBuffer::IContainerPtr_t CircularBuffer::getData(const Header& desired)
        {
            for (auto itr = m_buffer.begin(); itr != m_buffer.end(); ++itr)
            {
                if ((*itr)->getHeader() == desired)
                {
                    return (*itr);
                }
            }
            return {};
        }

        CircularBuffer::IContainerConstPtr_t CircularBuffer::getData(const Header& desired) const
        {
            for (auto itr = m_buffer.begin(); itr != m_buffer.end(); ++itr)
            {
                if ((*itr)->getHeader() == desired)
                {
                    return (*itr);
                }
            }
            return {};
        }

        static BufferConstructor<CircularBuffer> g_ctr;
    }
}
