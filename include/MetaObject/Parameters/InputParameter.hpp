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

https://github.com/dtmoodie/parameters
*/
#pragma once
#include "MetaObject/Detail/TypeInfo.h"
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Parameters/IParameter.hpp"
#include <functional>
#include <string>
#include <memory>
namespace mo
{
    
    class MO_EXPORTS InputParameter: public virtual IParameter
    {
    public:
        typedef std::function<bool(std::weak_ptr<IParameter>)> qualifier_f;
        typedef std::shared_ptr<InputParameter> Ptr;

        virtual bool SetInput(std::shared_ptr<IParameter> param) = 0;
        virtual bool SetInput(IParameter* param) = 0;
        virtual IParameter* GetInput() = 0;
        virtual bool AcceptsInput(std::weak_ptr<IParameter> param) const = 0;
        virtual bool AcceptsInput(IParameter* param) const = 0;
        virtual bool AcceptsType(TypeInfo type) const = 0;
        virtual void SetQualifier(std::function<bool(std::weak_ptr<IParameter>)> f)
        {
            qualifier = f;
        }
    protected:
        qualifier_f qualifier;
    };
}