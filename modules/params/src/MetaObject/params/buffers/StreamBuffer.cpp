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

        StreamBuffer::IContainerPtr_t StreamBuffer::getData(const Header& desired)
        {
            auto result = Map::getData(desired);
            if (result)
            {
                _current_timestamp = result->getHeader().timestamp;
                _current_frame_number = result->getHeader().frame_number;
                Map::modifyDataBuffer([this](Map::Buffer_t& buffer) { prune(buffer); });
            }
            return result;
        }

        StreamBuffer::IContainerConstPtr_t StreamBuffer::getData(const Header& desired) const
        {
            return Map::getData(desired);
        }

        uint32_t StreamBuffer::prune(Map::Buffer_t& data_buffer)
        {
            uint32_t remove_count = 0;
            if (_current_timestamp && _time_padding)
            {
                auto itr = data_buffer.begin();
                while (itr != data_buffer.end())
                {
                    if (itr->first.timestamp && *itr->first.timestamp < mo::Time(*_current_timestamp - *_time_padding))
                    {
                        ++remove_count;
                        itr = data_buffer.erase(itr);
                    }
                    else
                    {
                        ++itr;
                    }
                }
            }
            if (_frame_padding && _current_frame_number > *_frame_padding)
            {
                const auto cutoff = _current_frame_number - *_frame_padding;
                auto itr = data_buffer.begin();
                while (itr != data_buffer.end())
                {
                    if (itr->first.frame_number <= cutoff)
                    {
                        ++remove_count;
                        itr = data_buffer.erase(itr);
                    }
                    else
                    {
                        ++itr;
                    }
                }
            }
            return remove_count;
        }

        BlockingStreamBuffer::BlockingStreamBuffer(const std::string& name)
            : StreamBuffer(name)
        {
            _frame_padding = 10;
        }

        BufferFlags BlockingStreamBuffer::getBufferType() const
        {
            return Type;
        }

        uint32_t BlockingStreamBuffer::prune(Map::Buffer_t& buffer)
        {
            const auto remove_count = StreamBuffer::prune(buffer);
            if (remove_count > 0)
            {
                _cv.notify_all();
            }
            return remove_count;
        }

        void BlockingStreamBuffer::onInputUpdate(const IDataContainerPtr_t& data, IParam* param, UpdateFlags flags)
        {
            Lock lock(IParam::mtx());
            if (_frame_padding)
            {
                while (Map::getSize() > (*_frame_padding + 1))
                {
                    _cv.wait_for(lock, boost::chrono::milliseconds(2));
                    if (lock)
                    {
                        lock.unlock();
                    }
                    IParam::emitUpdate(data->getHeader(), mo::BufferUpdated_e);
                    if (!lock)
                    {
                        lock.lock();
                    }
                }
                if (!lock)
                {
                    lock.lock();
                }
            }

            Map::onInputUpdate(data, param, flags);
        }

        DroppingStreamBuffer::DroppingStreamBuffer(const std::string& name)
            : BlockingStreamBuffer(name)
        {
        }

        BufferFlags DroppingStreamBuffer::getBufferType() const
        {
            return Type;
        }

        void DroppingStreamBuffer::onInputUpdate(const IDataContainerPtr_t& data, IParam* param, UpdateFlags fgs)
        {
            if (_frame_padding && (Map::getSize() > (*_frame_padding + 1)))
            {
                return;
            }

            BlockingStreamBuffer::onInputUpdate(data, param, fgs);
        }

        static BufferConstructor<StreamBuffer> g_ctr_stream_buffer;
        static BufferConstructor<BlockingStreamBuffer> g_ctr_blocking_stream_buffer;
        static BufferConstructor<DroppingStreamBuffer> g_ctr_dropping_stream_buffer;
    }
}
