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

#include "IBuffer.hpp"
#include "MetaObject/params/ITInputParam.hpp"
#include "MetaObject/params/ITParam.hpp"
#include "MetaObject/params/MetaParam.hpp"
#include "MetaObject/params/ParamConstructor.hpp"
#include <boost/circular_buffer.hpp>

namespace mo
{
    namespace buffer
    {
        struct CircularBuffer : public IBuffer
        {
            static constexpr const auto Type = CIRCULAR_BUFFER;
            CircularBuffer();

            // IBuffer
            virtual void setFrameBufferCapacity(const uint64_t size) override;
            virtual void setTimePaddingCapacity(const Duration& time) override;

            virtual boost::optional<uint64_t> getFrameBufferCapacity() const override;
            virtual boost::optional<Duration> getTimePaddingCapacity() const override;

            virtual uint64_t getSize() const override;
            bool getTimestampRange(mo::OptionalTime& start, mo::OptionalTime& end) override;
            bool getFrameNumberRange(uint64_t& start, uint64_t& end) override;
            virtual BufferFlags getBufferType() const override;

            // IParam

            virtual IContainerPtr_t getData(const Header& desired = Header()) override;
            virtual IContainerConstPtr_t getData(const Header& desired = Header()) const override;

            // InputParam

          protected:
            void onInputUpdate(const IDataContainerPtr_t&, IParam*, UpdateFlags) override;

          private:
            boost::circular_buffer<IDataContainerPtr_t> m_buffer;
        };
    }
}
