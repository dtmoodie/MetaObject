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

#include "MetaObject/detail/Export.hpp"
#include "MetaObject/detail/TypeInfo.hpp"
#include "MetaObject/params/IParam.hpp"
#include <functional>
#include <memory>
#include <string>
namespace mo
{
    class MO_EXPORTS InputParam : virtual public IParam
    {
      public:
        using Qualifier_f = std::function<bool(IParam*)>;
        typedef std::shared_ptr<InputParam> Ptr;

        InputParam();
        virtual ~InputParam() override;

        // This gets a pointer to the variable that feeds into this input
        virtual IParam* getInputParam() const;

        virtual bool setInput(const std::shared_ptr<IParam>& param);
        virtual bool setInput(IParam* param = nullptr);

        // These values can differ from the current timestamp and frame number
        // since these values represent the next value that can be read, whereas getTimestamp and
        // getFramenumber represent that data currently loaded
        virtual OptionalTime getInputTimestamp();
        virtual uint64_t getInputFrameNumber();
        virtual bool isInputSet() const;

        virtual bool acceptsInput(IParam* param) const;
        virtual bool acceptsType(const TypeInfo& type) const;

        void setQualifier(const Qualifier_f& f);

        std::ostream& print(std::ostream& os) const override;

        virtual TypeInfo getTypeInfo() const override;

        virtual void visit(IReadVisitor&) override;
        virtual void visit(IWriteVisitor&) const override;
        virtual void visit(BinaryInputVisitor&) override;
        virtual void visit(BinaryOutputVisitor& ar) const override;

        virtual IContainerPtr_t getData(const Header& desired = Header()) override;
        virtual IContainerConstPtr_t getData(const Header& desired = Header()) const override;

      protected:
        virtual void onInputUpdate(const IDataContainerPtr_t&, IParam*, UpdateFlags) = 0;
        virtual void onInputDelete(const IParam* param);

        InputParam(const InputParam&) = delete;
        InputParam& operator=(const InputParam&) = delete;
        InputParam& operator=(InputParam&&) = delete;

        TSlot<DataUpdate_s> m_update_slot;
        TSlot<void(const IParam*)> m_delete_slot;
        Qualifier_f qualifier;
        IParam* m_input_param = nullptr;
        std::shared_ptr<IParam> m_shared_input;
        mutable IDataContainerPtr_t m_current_data;
    };
}
