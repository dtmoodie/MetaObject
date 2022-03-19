/*
Copyright (c) 2015 Daniel Moodie.
All rights reserved.

Redistribution and use in source and binary forms are permitted
provided that the above copyright notice and this paragraph are
duplicated in all such forms and that any documentation,
advertising materials, and other materials related to such
distribution and use acknowledge that the software was developed
by the Daniel Moodie. The name of
Daniel Moodie may not be used to endorse or promote products derived
from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

https://github.com/dtmoodie/MetaObject
*/
#ifndef MO_PARAMS_BUFFER_MAP_HPP
#define MO_PARAMS_BUFFER_MAP_HPP

#include "IBuffer.hpp"
#include <MetaObject/params/TParam.hpp>
#include <MetaObject/thread/ConditionVariable.hpp>
#include <MetaObject/thread/Mutex.hpp>

#include <map>

namespace mo
{
    namespace buffer
    {
        class MO_EXPORTS Map : public TParam<IBuffer>
        {
          public:
            enum PushPolicy
            {
                GROW,
                PRUNE,
                BLOCK,
                DROP
            };

            enum SearchPolicy
            {
                EXACT,
                NEAREST
            };

            constexpr static const BufferFlags Type = BufferFlags::MAP_BUFFER;

            Map(const std::string& name = "", PushPolicy push_policy = GROW, SearchPolicy search_policy = EXACT, const Duration& pad = std::chrono::milliseconds(100));
            ~Map() override;

            // IBuffer
            void setFrameBufferCapacity(const uint64_t size) override;
            void setTimePaddingCapacity(const Duration& time) override;

            boost::optional<uint64_t> getFrameBufferCapacity() const override;
            boost::optional<Duration> getTimePaddingCapacity() const override;

            uint64_t getSize() const override;
            uint64_t clear() override;
            bool getTimestampRange(mo::OptionalTime& start, mo::OptionalTime& end) override;
            bool getFrameNumberRange(uint64_t& start, uint64_t& end) override;
            BufferFlags getBufferType() const override;

            // ISubscriber
            bool acceptsPublisher(const IPublisher&) const override;
            bool acceptsType(const TypeInfo&) const override;
            std::vector<TypeInfo> getInputTypes() const override;
            bool isInputSet() const override;
            IPublisher* getPublisher() const override;

            bool setInput(std::shared_ptr<IPublisher> param) override;
            bool setInput(IPublisher* param = nullptr) override;

            IDataContainerConstPtr_t getCurrentData(IAsyncStream* = nullptr) const override;

            IDataContainerConstPtr_t getData(const Header* desired = nullptr, IAsyncStream* stream = nullptr) override;

            // IPublisher
            bool providesOutput(TypeInfo type) const override;
            std::vector<TypeInfo> getOutputTypes() const override;

            std::vector<Header> getAvailableHeaders() const override;
            boost::optional<Header> getNewestHeader() const override;

            uint32_t getNumSubscribers() const override;
            void setAllocator(Allocator::Ptr_t) override;

            // IParam
            void setMtx(Mutex_t& mtx) override;

            ConnectionPtr_t registerUpdateNotifier(ISlot& f) override;
            ConnectionPtr_t registerUpdateNotifier(const ISignalRelay::Ptr_t& relay) override;

            bool hasNewData() const override;

            std::ostream& print(std::ostream& os) const override;

          protected:
            void onInputUpdate(const IDataContainerConstPtr_t&, const IParam&, UpdateFlags, IAsyncStream*);

            IDataContainerConstPtr_t search(const Header& hdr) const;
            IDataContainerConstPtr_t searchExact(const Header& hdr) const;
            IDataContainerConstPtr_t searchNearest(const Header& hdr) const;

            struct StoragePair
            {
                StoragePair() = default;
                StoragePair(IDataContainerConstPtr_t&& d)
                    : data(std::move(d))
                {
                }

                StoragePair& operator=(const IDataContainerConstPtr_t& d)
                {
                    data = d;
                    return *this;
                }

                IDataContainerConstPtr_t getData() const
                {
                    retrieved = true;
                    return data;
                }

                bool hasBeenRetrieved() const
                {
                    return retrieved;
                }

                operator IDataContainerConstPtr_t() const
                {
                    retrieved = true;
                    return data;
                }
                operator bool()
                {
                    return retrieved;
                }

              private:
                IDataContainerConstPtr_t data;
                mutable bool retrieved = false;
            };

            using Buffer_t = std::map<Header, StoragePair>;

          private:
            void pushData(const IDataContainerConstPtr_t& data, IAsyncStream*);
            void pushOrDrop(const IDataContainerConstPtr_t& data, IAsyncStream*);
            void pushAndWait(const IDataContainerConstPtr_t& data, IAsyncStream*);
            uint32_t prune();

            ConditionVariable m_cv;

            Buffer_t m_data_buffer;
            IDataContainerConstPtr_t m_current_data;

            OptionalTime m_current_timestamp;
            FrameNumber m_current_frame_number;

            boost::optional<Duration> m_time_padding;
            boost::optional<uint64_t> m_frame_padding;

            PushPolicy m_push_policy = GROW;
            SearchPolicy m_search_policy = EXACT;

            mutable Mutex_t m_mtx;

            TSlot<DataUpdate_s> m_update_slot;
            TSlot<Delete_s> m_delete_slot;

            ConnectionPtr_t m_update_connection;
            ConnectionPtr_t m_delete_connection;

            TSignal<DataUpdate_s> m_update_signal;

            std::shared_ptr<IPublisher> m_shared_publisher;
            IPublisher* m_publisher = nullptr;
        };
    } // namespace buffer
} // namespace mo
#endif // MO_PARAMS_BUFFER_MAP_HPP
