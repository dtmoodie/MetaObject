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

        void Map::setTimePaddingCapacity(const Duration&)
        {
        }

        boost::optional<uint64_t> Map::getFrameBufferCapacity() const
        {
            return {};
        }

        boost::optional<Duration> Map::getTimePaddingCapacity() const
        {
            return {};
        }

        size_t Map::getSize() const
        {
            Lock lock(IParam::mtx());
            return m_data_buffer.size();
        }

        uint64_t Map::clear()
        {
            Lock lock(IParam::mtx());
            const uint64_t sz = m_data_buffer.size();
            m_data_buffer.clear();
            return sz;
        }

        bool Map::getTimestampRange(mo::OptionalTime& start, mo::OptionalTime& end)
        {
            Lock lock(IParam::mtx());
            if (m_data_buffer.size())
            {
                Lock lock(IParam::mtx());
                start = m_data_buffer.begin()->first.timestamp;
                end = m_data_buffer.rbegin()->first.timestamp;
                return true;
            }
            return false;
        }

        bool Map::getFrameNumberRange(uint64_t& start, uint64_t& end)
        {
            Lock lock(IParam::mtx());
            if (m_data_buffer.size())
            {
                Lock lock(IParam::mtx());
                start = m_data_buffer.begin()->first.frame_number;
                end = m_data_buffer.rbegin()->first.frame_number;
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
                m_data_buffer[data->getHeader()] = data;
            }
        }

        Map::IContainerPtr_t Map::getData(const Header& desired)
        {
            Lock lock(IParam::mtx());
            m_current_data = search(desired);
            return m_current_data;
        }

        Map::IContainerConstPtr_t Map::getData(const Header& desired) const
        {
            Lock lock(IParam::mtx());
            auto result = search(desired);
            if (result)
            {
                return result;
            }
            return m_current_data;
        }

        IDataContainerPtr_t Map::search(const Header& hdr) const
        {
            if (m_data_buffer.size() == 0)
            {
                return {};
            }
            else
            {
                if (!hdr.timestamp)
                {
                    if (m_data_buffer.size())
                    {
                        return (--this->m_data_buffer.end())->second;
                    }
                    else
                    {
                        return {};
                    }
                }
                else
                {
                    auto itr = m_data_buffer.find(hdr);
                    if (itr != m_data_buffer.end())
                    {
                        return itr->second;
                    }
                }
            }

            return {};
        }

        static BufferConstructor<Map> g_ctr_map;
    }
}
