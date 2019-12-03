#include "Map.hpp"
#include <MetaObject/thread/fiber_include.hpp>

#include <boost/thread/locks.hpp>

namespace mo
{
    namespace buffer
    {
        Map::Map(const std::string& name, PushPolicy push_policy, SearchPolicy search_policy)
            : IParam(name)
            , m_push_policy(push_policy)
            , m_search_policy(search_policy)
        {
            m_update_slot = std::bind(
                &Map::onInputUpdate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
            IParam::setMtx(&m_mtx);
            appendFlags(mo::ParamFlags::kBUFFER);
        }

        Map::~Map()
        {
        }

        void Map::setFrameBufferCapacity(const uint64_t size)
        {
            m_frame_padding = size;
        }

        void Map::setTimePaddingCapacity(const Duration& time)
        {
            m_time_padding = time;
        }

        boost::optional<uint64_t> Map::getFrameBufferCapacity() const
        {
            return m_frame_padding;
        }

        boost::optional<Duration> Map::getTimePaddingCapacity() const
        {
            return m_time_padding;
        }

        uint64_t Map::getSize() const
        {
            Lock_t lock(m_mtx);
            return m_data_buffer.size();
        }

        uint64_t Map::clear()
        {
            Lock_t lock(m_mtx);
            const uint64_t sz = m_data_buffer.size();
            m_data_buffer.clear();
            m_current_frame_number = FrameNumber();
            m_current_timestamp.reset();
            return sz;
        }

        bool Map::getTimestampRange(mo::OptionalTime& start, mo::OptionalTime& end)
        {
            Lock_t lock(m_mtx);
            if (!m_data_buffer.empty())
            {
                Lock_t lock(IParam::mtx());
                start = m_data_buffer.begin()->first.timestamp;
                end = m_data_buffer.rbegin()->first.timestamp;
                return true;
            }
            return false;
        }

        bool Map::getFrameNumberRange(uint64_t& start, uint64_t& end)
        {
            Lock_t lock(m_mtx);
            if (!m_data_buffer.empty())
            {
                Lock_t lock(IParam::mtx());
                start = m_data_buffer.begin()->first.frame_number;
                end = m_data_buffer.rbegin()->first.frame_number;
                return true;
            }
            return false;
        }

        BufferFlags Map::getBufferType() const
        {
            return BufferFlags::MAP_BUFFER;
        }

        bool Map::setInput(const std::shared_ptr<IParam>& param)
        {
            return IBuffer::setInput(param);
        }

        bool Map::setInput(IParam* param)
        {
            if (IBuffer::setInput(param))
            {
                auto data = IBuffer::getData();
                if (data)
                {
                    pushData(data);
                }
                return true;
            }
            return false;
        }

        void Map::onInputUpdate(const IDataContainerPtr_t& data, IParam* param, UpdateFlags flags)
        {
            (void)param;
            (void)flags;
            if (data)
            {
                pushData(data);
            }
        }

        void Map::pushData(const IDataContainerPtr_t& data)
        {
            if ((m_push_policy == GROW) || (m_push_policy == PRUNE))
            {
                {
                    Lock_t lock(IParam::mtx());
                    m_data_buffer[data->getHeader()] = data;
                }
                emitUpdate(data, UpdateFlags::kBUFFER_UPDATED);
            }
            else
            {
                if (m_push_policy == BLOCK)
                {
                    pushAndWait(data);
                }
                else
                {
                    pushOrDrop(data);
                }
            }
        }

        void Map::pushOrDrop(const IDataContainerPtr_t& data)
        {
            {
                Lock_t lock(IParam::mtx());
                if (m_frame_padding)
                {
                    if (m_data_buffer.size() > (*m_frame_padding + 1))
                    {
                        return;
                    }
                }
                m_data_buffer[data->getHeader()] = data;
            }
            emitUpdate(data, mo::UpdateFlags::kBUFFER_UPDATED);
        }

        void Map::pushAndWait(const IDataContainerPtr_t& data)
        {
            {
                Lock_t lock(IParam::mtx());
                if (m_frame_padding)
                {
                    while (m_data_buffer.size() > (*m_frame_padding + 1))
                    {
                        m_cv.wait_for(lock, 100 * ns);
                    }
                    if (!lock)
                    {
                        lock.lock();
                    }
                }

                m_data_buffer[data->getHeader()] = data;
            }
            emitUpdate(data, mo::UpdateFlags::kBUFFER_UPDATED);
        }

        Map::IContainerPtr_t Map::getData(const Header& desired)
        {
            Lock_t lock(m_mtx);
            m_current_data = search(desired);
            if (m_current_data)
            {
                m_current_timestamp = m_current_data->getHeader().timestamp;
                m_current_frame_number = m_current_data->getHeader().frame_number;
                if (m_push_policy != GROW)
                {
                    prune();
                }
            }

            return m_current_data;
        }

        Map::IContainerConstPtr_t Map::getData(const Header& desired) const
        {
            Lock_t lock(m_mtx);
            auto result = search(desired);
            if (result)
            {
                return result;
            }
            return m_current_data;
        }

        void Map::setMtx(Mutex_t*)
        {
        }

        IDataContainerPtr_t Map::search(const Header& hdr) const
        {
            if (m_search_policy == EXACT)
            {
                return searchExact(hdr);
            }
            return searchNearest(hdr);
        }

        IDataContainerPtr_t Map::searchExact(const Header& hdr) const
        {
            if (m_data_buffer.empty())
            {
                return {};
            }

            if (!hdr.timestamp && !hdr.frame_number.valid())
            {
                return (--this->m_data_buffer.end())->second;
            }
            else
            {
                auto itr = m_data_buffer.find(hdr);
                if (itr != m_data_buffer.end())
                {
                    return itr->second;
                }
            }

            return {};
        }

        uint32_t Map::prune()
        {
            uint32_t remove_count = 0;
            if (m_current_timestamp && m_time_padding)
            {
                auto itr = m_data_buffer.begin();
                while (itr != m_data_buffer.end())
                {
                    if (itr->first.timestamp &&
                        *itr->first.timestamp < mo::Time(*m_current_timestamp - *m_time_padding))
                    {
                        ++remove_count;
                        itr = m_data_buffer.erase(itr);
                    }
                    else
                    {
                        ++itr;
                    }
                }
            }
            if (m_frame_padding && m_current_frame_number > *m_frame_padding)
            {
                const auto cutoff = m_current_frame_number - *m_frame_padding;
                auto itr = m_data_buffer.begin();
                while (itr != m_data_buffer.end())
                {
                    if (itr->first.frame_number <= cutoff)
                    {
                        ++remove_count;
                        itr = m_data_buffer.erase(itr);
                    }
                    else
                    {
                        ++itr;
                    }
                }
            }
            if (remove_count > 0)
            {
                m_cv.notify_all();
            }
            return remove_count;
        }

        IDataContainerPtr_t Map::searchNearest(const Header& hdr) const
        {
            if (!hdr.timestamp && !hdr.frame_number.valid())
            {
                if (!m_data_buffer.empty())
                {
                    return (--m_data_buffer.end())->second;
                }
                return {};
            }

            // const auto ts = *hdr.timestamp;
            auto upper = m_data_buffer.upper_bound(hdr);
            auto lower = m_data_buffer.lower_bound(hdr);

            if (upper != m_data_buffer.end() && lower != m_data_buffer.end())
            {
                // first check if lower is exactly right
                const auto ts_upper = upper->first.timestamp;
                auto ts_lower = lower->first.timestamp;
                const auto fn_upper = upper->first.frame_number;
                auto fn_lower = lower->first.frame_number;
                // Since std::Map::lower_bound can return the exact item if the timestamps match
                // or the first item not less than the desired timestamp, we first check
                // if the timestamps are exact, if they are not we look at the item before lower
                // to be able to accurately check items on both sides of the desired timestamp
                // https://stackoverflow.com/questions/529831/returning-the-greatest-key-strictly-less-than-the-given-key-in-a-c-map
                if (ts_upper && ts_lower && hdr.timestamp)
                {
                    if (ts_lower == hdr.timestamp)
                    {
                        return lower->second;
                    }

                    if (lower != m_data_buffer.begin())
                    {
                        --lower;
                        ts_lower = *lower->first.timestamp;
                    }
                    const auto upperdelta = std::chrono::abs(*ts_upper - *hdr.timestamp);
                    const auto lowerdelta = std::chrono::abs(*ts_lower - *hdr.timestamp);
                    if (upperdelta < lowerdelta)
                    {
                        return upper->second;
                    }
                    return lower->second;
                }
                // frame number based search
                if (fn_lower == hdr.frame_number)
                {
                    return lower->second;
                }

                if (lower != m_data_buffer.begin())
                {
                    --lower;
                    fn_lower = lower->first.frame_number;
                }
                const auto upperdelta = fn_upper - hdr.frame_number;
                const auto lowerdelta = fn_lower - hdr.frame_number;
                if (upperdelta < lowerdelta)
                {
                    return upper->second;
                }
                return lower->second;
            }
            if (lower != m_data_buffer.end())
            {
                return lower->second;
            }

            if (upper != m_data_buffer.end())
            {
                return upper->second;
            }
            if (!m_data_buffer.empty())
            {
                return (--m_data_buffer.end())->second;
            }
            return {};
        }

        OptionalTime Map::getTimestamp() const
        {
            return m_current_timestamp;
        }

        FrameNumber Map::getFrameNumber() const
        {
            return m_current_frame_number;
        }

        template <Map::PushPolicy PUSH_POLICY, Map::SearchPolicy SEARCH_POLICY, BufferFlags::EnumValueType TYPE>
        struct MapConstructor
        {
            MapConstructor()
            {
                buffer::BufferFactory::registerConstructor([]() { return new Map("", PUSH_POLICY, SEARCH_POLICY); },
                                                           TYPE);
            }
        };

        IBuffer* IBuffer::create(BufferFlags type)
        {
            switch (type)
            {
            case BufferFlags::MAP_BUFFER:
                return new Map("", Map::GROW, Map::EXACT);
            case BufferFlags::STREAM_BUFFER:
                return new Map("", Map::PRUNE, Map::EXACT);
            case BufferFlags::BLOCKING_STREAM_BUFFER:
                return new Map("", Map::BLOCK, Map::EXACT);
            case BufferFlags::DROPPING_STREAM_BUFFER:
                return new Map("", Map::DROP, Map::EXACT);
            case BufferFlags::NEAREST_NEIGHBOR_BUFFER:
                return new Map("", Map::BLOCK, Map::NEAREST);
            default:
                return nullptr;
            }
        }
        static MapConstructor<Map::GROW, Map::EXACT, BufferFlags::MAP_BUFFER> g_map_ctr;
        static MapConstructor<Map::PRUNE, Map::EXACT, BufferFlags::STREAM_BUFFER> g_stream_ctr;
        static MapConstructor<Map::BLOCK, Map::EXACT, BufferFlags::BLOCKING_STREAM_BUFFER> g_blocking_ctr;
        static MapConstructor<Map::DROP, Map::EXACT, BufferFlags::DROPPING_STREAM_BUFFER> g_dropping_ctr;
        static MapConstructor<Map::BLOCK, Map::NEAREST, BufferFlags::NEAREST_NEIGHBOR_BUFFER> g_nn_ctr;
    }
}
