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

            // IBuffer
            virtual void setFrameBufferCapacity(const uint64_t size) override;
            virtual void setTimePaddingCapacity(const mo::Time_t& time) override;

            virtual boost::optional<uint64_t> getFrameBufferCapacity() const override;
            virtual OptionalTime_t getTimePaddingCapacity() const override;

            virtual size_t getSize() const override;
            virtual bool getTimestampRange(mo::OptionalTime_t& start, mo::OptionalTime_t& end) override;
            virtual bool getFrameNumberRange(uint64_t& start, uint64_t& end) override;
            virtual BufferFlags getBufferType() const override;

            // IParam
            virtual TypeInfo getTypeInfo() const override;

            virtual void visit(IReadVisitor*) override;
            virtual void visit(IWriteVisitor*) const override;

            virtual IContainerPtr_t getData(const Header& desired = Header()) override;
            virtual IContainerConstPtr_t getData(const Header& desired = Header()) const override;

            // InputParam
            virtual bool getInputData(const Header& desired, Header* retrieved) override;
            virtual IParam* getInputParam() const override;
            virtual bool setInput(const std::shared_ptr<IParam>& param) override;
            virtual bool setInput(IParam* param = nullptr) override;

            virtual OptionalTime_t getInputTimestamp() override;
            virtual uint64_t getInputFrameNumber() override;
            virtual bool isInputSet() const override;

            virtual bool acceptsInput(IParam* param) const override;
            virtual bool acceptsType(const TypeInfo& type) const override;

          protected:
            void onInputUpdate(const IDataContainerPtr_t&, IParam*, UpdateFlags);
            virtual std::map<Header, IDataContainerPtr_t>::iterator search(const Header& hdr);
            virtual std::map<Header, IDataContainerPtr_t>::const_iterator search(const Header& hdr) const;

          private:
            std::map<Header, IDataContainerPtr_t> _data_buffer;
            mutable IDataContainerPtr_t m_current;
            TSlot<DataUpdate_s> m_update_slot;
            TSlot<void(const IParam*)> m_delete_slot;
            IParam* m_input_param;
            std::shared_ptr<IParam> m_shared_input;
        };
    }
}
