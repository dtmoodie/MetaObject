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
        typedef std::function<bool(std::weak_ptr<IParam>)> qualifier_f;
        typedef std::shared_ptr<InputParam> Ptr;

        InputParam();
        virtual ~InputParam();

        // This loads the value at the requested timestamp into the input
        // Param such that it can be read
        virtual bool getInput(const OptionalTime_t& ts, size_t* fn = nullptr);
        virtual bool getInput(size_t fn, OptionalTime_t* ts = nullptr);
        // This gets a pointer to the variable that feeds into this input
        virtual IParam* getInputParam() const = 0;

        virtual bool setInput(std::shared_ptr<IParam> param) = 0;
        virtual bool setInput(IParam* param = nullptr) = 0;

        virtual OptionalTime_t getInputTimestamp() = 0;
        virtual size_t getInputFrameNumber() = 0;
        virtual bool isInputSet() const = 0;

        virtual bool acceptsInput(IParam* param) const = 0;
        virtual bool acceptsType(const TypeInfo& type) const = 0;

        void setQualifier(std::function<bool(std::weak_ptr<IParam>)> f) { qualifier = f; }

        std::ostream& print(std::ostream& os) const override;
      protected:
        InputParam(const InputParam&) = delete;
        InputParam& operator=(const InputParam&) = delete;
        InputParam& operator=(InputParam&&) = delete;
        qualifier_f qualifier;
    };
}
