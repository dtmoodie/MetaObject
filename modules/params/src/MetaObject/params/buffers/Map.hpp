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

#include <map>

namespace mo
{
    namespace buffer
    {
        class Map : public IBuffer, public InputParam
        {
          public:
            static const BufferFlags Type = MAP_BUFFER;

            Map(const std::string& name = "");

            virtual void setFrameBufferCapacity(const uint64_t size) override;
            virtual void setTimePaddingCapacity(const mo::Time_t& time) override;

            virtual boost::optional<uint64_t> getFrameBufferCapacity() const override;
            virtual OptionalTime_t getTimePaddingCapacity() const override;

            virtual size_t getSize() const override;
            virtual bool getTimestampRange(mo::OptionalTime_t& start, mo::OptionalTime_t& end) override;
            virtual bool getFrameNumberRange(uint64_t& start, uint64_t& end) override;
            virtual BufferFlags getBufferType() const override;

          protected:
            void onInputUpdate(const IDataContainerPtr_t&, IParam*, UpdateFlags);

          private:
            std::map<Header, IDataContainerPtr_t> _data_buffer;
            TSlot<DataUpdate_s> m_update_slot;
            IParam* m_input_param;
        };
    }
}
