#include "Map.hpp"
#include <boost/thread/locks.hpp>
#include <boost/thread/recursive_mutex.hpp>

namespace mo
{
    namespace buffer
    {
        Map::Map(const std::string& name)
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

        TypeInfo Map::getTypeInfo() const
        {
            if (m_input_param)
            {
                return m_input_param->getTypeInfo();
            }
            else
            {
                return TypeInfo::Void();
            }
        }

        void Map::visit(IReadVisitor* visitor)
        {
            Lock lock(IParam::mtx());
            if (m_current)
            {
                m_current->visit(visitor);
            }
        }

        void Map::visit(IWriteVisitor* visitor) const
        {
            Lock lock(IParam::mtx());
            if (m_current)
            {
                m_current->visit(visitor);
            }
        }

        Map::IContainerPtr_t Map::getData(const Header& desired)
        {
            Lock lock(IParam::mtx());
            m_current.reset();
            auto itr = search(desired);
            if (itr != _data_buffer.end())
            {
                m_current = itr->second;
                return m_current;
            }
            return {};
        }

        Map::IContainerConstPtr_t Map::getData(const Header& desired) const
        {
            Lock lock(IParam::mtx());
            m_current.reset();
            auto itr = search(desired);
            if (itr != _data_buffer.end())
            {
                m_current = itr->second;
                return m_current;
            }
            return {};
        }

        bool Map::getInputData(const Header& desired, Header* retrieved)
        {
            auto data = getData(desired);
            if (data)
            {
                if (retrieved)
                {
                    *retrieved = data->getHeader();
                }
                return true;
            }
            return false;
        }

        IParam* Map::getInputParam() const
        {
            Lock lock(IParam::mtx());
            return m_input_param;
        }

        bool Map::setInput(const std::shared_ptr<IParam>& param)
        {
            Lock lock(mtx());
            if (setInput(param.get()))
            {
                m_shared_input = param;
                return true;
            }
            return false;
        }

        bool Map::setInput(IParam* param)
        {
            Lock lock(mtx());
            if (m_input_param)
            {
                m_input_param->unsubscribe();
                m_update_slot.clear();
                m_delete_slot.clear();
            }
            m_input_param = param;
            param->subscribe();
            param->registerUpdateNotifier(&m_update_slot);
            param->registerDeleteNotifier(&m_delete_slot);
            return true;
        }

        OptionalTime_t Map::getInputTimestamp()
        {
        }

        uint64_t Map::getInputFrameNumber()
        {
        }

        bool Map::isInputSet() const
        {
            return m_input_param != nullptr;
        }

        bool Map::acceptsInput(IParam*) const
        {
            return true;
        }

        bool Map::acceptsType(const TypeInfo&) const
        {
            return true;
        }

        static BufferConstructor<Map> g_ctr;
    }
}
