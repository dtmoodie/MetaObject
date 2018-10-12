#include "Map.hpp"
#include <boost/thread/locks.hpp>
#include <boost/thread/recursive_mutex.hpp>

namespace mo
{
    namespace buffer
    {
        Map::Map(const std::string& name)
            : IParam(name)
        {
            m_update_slot = std::bind(
                &Map::onInputUpdate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
        }

        void Map::setFrameBufferCapacity(const uint64_t)
        {
        }

        void Map::setTimePaddingCapacity(const mo::Time_t&)
        {
        }

        boost::optional<uint64_t> Map::getFrameBufferCapacity() const
        {
            return {};
        }

        OptionalTime_t Map::getTimePaddingCapacity() const
        {
            return {};
        }

        size_t Map::getSize() const
        {
            return _data_buffer.size();
        }

        bool Map::getTimestampRange(mo::OptionalTime_t& start, mo::OptionalTime_t& end)
        {
            if (_data_buffer.size())
            {
                Lock lock(IParam::mtx());
                start = _data_buffer.begin()->first.timestamp;
                end = _data_buffer.rbegin()->first.timestamp;
                return true;
            }
            return false;
        }

        bool Map::getFrameNumberRange(uint64_t& start, uint64_t& end)
        {
            if (_data_buffer.size())
            {
                Lock lock(IParam::mtx());
                start = _data_buffer.begin()->first.frame_number;
                end = _data_buffer.rbegin()->first.frame_number;
                return true;
            }
            return false;
        }

        BufferFlags Map::getBufferType() const
        {
            return MAP_BUFFER;
        }

        void Map::onInputUpdate(const IDataContainerPtr_t& data, IParam*, UpdateFlags)
        {
            if (data)
            {
                Lock lock(mtx());
                _data_buffer[data->getHeader()] = data;
            }
        }

        Map::IContainerPtr_t Map::getData(const Header& desired)
        {
            Lock lock(IParam::mtx());
            m_current = search(desired);
            return m_current;
        }

        Map::IContainerConstPtr_t Map::getData(const Header& desired) const
        {
            Lock lock(IParam::mtx());
            m_current = search(desired);
            return m_current;
        }

        IDataContainerPtr_t Map::search(const Header& hdr) const
        {
            if (_data_buffer.size() == 0)
            {
                return {};
            }
            else
            {
                if (!hdr.timestamp)
                {
                    if (_data_buffer.size())
                    {
                        return (--this->_data_buffer.end())->second;
                    }
                    else
                    {
                        return {};
                    }
                }
            }

            return {};
        }

        static BufferConstructor<Map> g_ctr;
    }
}
