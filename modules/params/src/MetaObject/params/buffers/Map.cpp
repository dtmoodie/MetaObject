#include "Map.hpp"
#include <MetaObject/params/IPublisher.hpp>

#include <boost/thread/locks.hpp>
#include <ct/bind.hpp>

namespace mo
{
    namespace buffer
    {
        Map::Map(const std::string& name, PushPolicy push_policy, SearchPolicy search_policy, const Duration& pad)
            : m_push_policy(push_policy)
            , m_search_policy(search_policy)
            , m_time_padding(pad)
        {

            this->appendFlags(mo::ParamFlags::kBUFFER);
            m_update_slot.bind(&Map::onInputUpdate, this);
        }

        Map::~Map()
        {
        }

        void Map::setFrameBufferCapacity(const uint64_t size)
        {
            Lock_t lock(m_mtx);
            m_frame_padding = size;
            m_time_padding = boost::none;
        }

        void Map::setTimePaddingCapacity(const Duration& time)
        {
            Lock_t lock(m_mtx);
            m_time_padding = time;
            m_frame_padding = boost::none;
        }

        boost::optional<uint64_t> Map::getFrameBufferCapacity() const
        {
            Lock_t lock(m_mtx);
            return m_frame_padding;
        }

        boost::optional<Duration> Map::getTimePaddingCapacity() const
        {
            Lock_t lock(m_mtx);
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

        bool Map::setInput(std::shared_ptr<IPublisher> param)
        {
            if (setInput(param.get()))
            {
                Lock_t lock(m_mtx);
                m_shared_publisher = std::move(param);
                return true;
            }
            return false;
        }

        bool Map::setInput(IPublisher* param)
        {
            Lock_t lock(m_mtx);
            if (param == nullptr)
            {
                m_update_slot.clear();
                m_publisher = nullptr;
                m_shared_publisher.reset();
                m_delete_slot.clear();
                return true;
            }
            m_update_connection = param->registerUpdateNotifier(m_update_slot);
            m_delete_connection = param->registerDeleteNotifier(m_delete_slot);

            if (m_update_connection && m_delete_connection)
            {
                auto stream = param->getStream();
                if (!stream)
                {
                    stream = IAsyncStream::current().get();
                }
                auto data = param->getData();
                if (data)
                {
                    pushData(data, stream);
                }
                m_publisher = param;
                return true;
            }
            else
            {
                m_update_connection.reset();
                m_delete_connection.reset();
            }
            return false;
        }

        bool Map::isInputSet() const
        {
            Mutex_t::Lock_t lock(m_mtx);
            return m_publisher != nullptr;
        }

        bool Map::acceptsPublisher(const IPublisher&) const
        {
            return true;
        }

        bool Map::acceptsType(const TypeInfo&) const
        {
            return true;
        }

        void Map::setAllocator(Allocator::Ptr_t)
        {
            // do we need to set an allocator?
        }

        std::vector<Header> Map::getAvailableHeaders() const
        {
            std::vector<Header> output;
            Mutex_t::Lock_t lock(m_mtx);
            for (const auto& itr : m_data_buffer)
            {
                output.push_back(itr.first);
            }

            return output;
        }

        boost::optional<Header> Map::getNewestHeader() const
        {
            Lock_t lock(m_mtx);
            if (!m_data_buffer.empty())
            {
                return m_data_buffer.end()->first;
            }
            return {};
        }

        bool Map::providesOutput(TypeInfo) const
        {
            return true;
        }

        uint32_t Map::getNumSubscribers() const
        {
            Mutex_t::Lock_t lock(m_mtx);
            if (m_publisher)
            {
                return m_publisher->getNumSubscribers();
            }
            return 0;
        }

        std::vector<TypeInfo> Map::getOutputTypes() const
        {
            Mutex_t::Lock_t lock(m_mtx);
            if (m_publisher)
            {
                return m_publisher->getOutputTypes();
            }
            return {};
        }

        std::vector<TypeInfo> Map::getInputTypes() const
        {
            return {};
        }

        IPublisher* Map::getPublisher() const
        {
            Mutex_t::Lock_t lock(m_mtx);
            return m_publisher;
        }

        void
        Map::onInputUpdate(IDataContainerConstPtr_t data, const IParam& param, UpdateFlags flags, IAsyncStream* stream)
        {
            (void)param;
            (void)flags;
            if (data)
            {
                pushData(data, stream);
            }
        }

        void Map::pushData(const IDataContainerConstPtr_t& data, IAsyncStream* stream)
        {
            if ((m_push_policy == GROW) || (m_push_policy == PRUNE))
            {
                {
                    Mutex_t::Lock_t lock(m_mtx);
                    m_data_buffer[data->getHeader()] = data;
                }
                m_update_signal(data, *this, ct::value(UpdateFlags::kBUFFER_UPDATED), stream);
            }
            else
            {
                if (m_push_policy == BLOCK)
                {
                    pushAndWait(data, stream);
                }
                else
                {
                    pushOrDrop(data, stream);
                }
            }
        }

        void Map::pushOrDrop(const IDataContainerConstPtr_t& data, IAsyncStream* stream)
        {
            {
                Mutex_t::Lock_t lock(m_mtx);
                if (m_frame_padding)
                {
                    if (m_data_buffer.size() > (*m_frame_padding + 1))
                    {
                        return;
                    }
                }
                m_data_buffer[data->getHeader()] = data;
            }
            m_update_signal(data, *this, ct::value(UpdateFlags::kBUFFER_UPDATED), stream);
        }

        void Map::pushAndWait(const IDataContainerConstPtr_t& data, IAsyncStream* stream)
        {
            {
                Mutex_t::Lock_t lock(m_mtx);
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
            m_update_signal(data, *this, ct::value(UpdateFlags::kBUFFER_UPDATED), stream);
        }

        IDataContainerConstPtr_t Map::getCurrentData(IAsyncStream* stream) const
        {
            Lock_t lock(m_mtx);
            return m_current_data;
        }

        IDataContainerConstPtr_t Map::getData(const Header* desired, IAsyncStream* stream)
        {
            IDataContainerConstPtr_t data;
            Lock_t lock(m_mtx);
            if (desired)
            {
                data = search(*desired);
            }
            else
            {
                // Just get the newest
                if (!m_data_buffer.empty())
                {
                    data = m_data_buffer.rbegin()->second;
                }
            }

            if (data)
            {
                m_current_data = data;
                m_current_timestamp = m_current_data->getHeader().timestamp;
                m_current_frame_number = m_current_data->getHeader().frame_number;
                if (m_push_policy != GROW)
                {
                    prune();
                }
            }
            return data;
        }

        void Map::setMtx(Mutex_t&)
        {
        }

        IDataContainerConstPtr_t Map::search(const Header& hdr) const
        {
            Lock_t lock(m_mtx);
            if (m_search_policy == EXACT)
            {
                return searchExact(hdr);
            }
            return searchNearest(hdr);
        }

        IDataContainerConstPtr_t Map::searchExact(const Header& hdr) const
        {
            Lock_t lock(m_mtx);
            if (m_data_buffer.empty())
            {
                return {};
            }

            if (!hdr.timestamp && !hdr.frame_number.valid())
            {
                return (--this->m_data_buffer.end())->second.getData();
            }
            else
            {
                auto itr = m_data_buffer.find(hdr);
                if (itr != m_data_buffer.end())
                {
                    return itr->second.getData();
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
                const mo::Time threshold(*m_current_timestamp - *m_time_padding);
                while (itr != m_data_buffer.end())
                {
                    if (itr->first.timestamp && *itr->first.timestamp < threshold)
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

        IDataContainerConstPtr_t Map::searchNearest(const Header& hdr) const
        {
            Lock_t lock(m_mtx);
            if (!hdr.timestamp && !hdr.frame_number.valid())
            {
                if (!m_data_buffer.empty())
                {
                    return (--m_data_buffer.end())->second.getData();
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
                        return lower->second.getData();
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
                        return upper->second.getData();
                    }
                    return lower->second.getData();
                }
                // frame number based search
                if (fn_lower == hdr.frame_number)
                {
                    return lower->second.getData();
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
                    return upper->second.getData();
                }
                return lower->second.getData();
            }
            if (lower != m_data_buffer.end())
            {
                return lower->second.getData();
            }

            if (upper != m_data_buffer.end())
            {
                return upper->second.getData();
            }
            if (!m_data_buffer.empty())
            {
                return (--m_data_buffer.end())->second.getData();
            }
            return {};
        }

        ConnectionPtr_t Map::registerUpdateNotifier(ISlot& f)
        {
            Lock_t lock(m_mtx);
            auto connection = TParam<IBuffer>::registerUpdateNotifier(f);
            if (!connection)
            {
                connection = m_update_signal.connect(f);
            }
            return connection;
        }

        ConnectionPtr_t Map::registerUpdateNotifier(const ISignalRelay::Ptr_t& relay)
        {
            Lock_t lock(m_mtx);
            MO_ASSERT_LOGGER(this->getLogger(), relay != nullptr);
            auto connection = TParam<IBuffer>::registerUpdateNotifier(relay);
            if (!connection)
            {
                auto tmp = relay;
                connection = m_update_signal.connect(tmp);
            }
            return connection;
        }

        bool Map::hasNewData() const
        {
            Lock_t lock(m_mtx);
            for (const auto& itr : m_data_buffer)
            {
                if (itr.second.hasBeenRetrieved() == false)
                {
                    return true;
                }
            }
            return false;
        }

        std::ostream& Map::print(std::ostream& os) const
        {
            // TODO

            return os;
        }

    } // namespace buffer
} // namespace mo
