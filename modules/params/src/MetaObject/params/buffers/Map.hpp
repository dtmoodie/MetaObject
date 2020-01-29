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
#pragma once

#include "BufferConstructor.hpp"
#include "IBuffer.hpp"

#include "MetaObject/params/InputParam.hpp"

#include <MetaObject/thread/fiber_include.hpp>

#include <map>

namespace mo
{
    namespace buffer
    {
        class Map : public IBuffer
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

            Map(const std::string& name = "", PushPolicy push_policy = GROW, SearchPolicy search_policy = EXACT);
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

            // InputParam
            bool setInput(const std::shared_ptr<IParam>& param) override;
            bool setInput(IParam* param = nullptr) override;

            // IParam

            OptionalTime getTimestamp() const override;
            FrameNumber getFrameNumber() const override;

            IContainerPtr_t getData(const Header& desired = Header()) override;
            IContainerConstPtr_t getData(const Header& desired = Header()) const override;
            void setMtx(Mutex_t* mtx) override;

          protected:
            void onInputUpdate(const IDataContainerPtr_t&, IParam*, UpdateFlags) override;
            IDataContainerPtr_t search(const Header& hdr) const;
            IDataContainerPtr_t searchExact(const Header& hdr) const;
            IDataContainerPtr_t searchNearest(const Header& hdr) const;

            using Buffer_t = std::map<Header, IDataContainerPtr_t>;

          private:
            void pushData(const IDataContainerPtr_t& data);
            void pushOrDrop(const IDataContainerPtr_t& data);
            void pushAndWait(const IDataContainerPtr_t& data);
            uint32_t prune();

            boost::fibers::condition_variable_any m_cv;

            Buffer_t m_data_buffer;

            OptionalTime m_current_timestamp;
            FrameNumber m_current_frame_number;

            boost::optional<Duration> m_time_padding;
            boost::optional<uint64_t> m_frame_padding;
            PushPolicy m_push_policy = GROW;
            SearchPolicy m_search_policy = EXACT;
            mutable mo::Mutex_t m_mtx;
        };
    }
}
