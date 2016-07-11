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
#include "Parameter_def.hpp"
#include "Parameter.hpp"
#include "LokiTypeInfo.h"

#include <functional>
#include <string>
namespace Parameters
{
    class PARAMETER_EXPORTS InputParameter
    {
    public:
        typedef std::function<bool(Parameter*)> qualifier_f;
        typedef std::shared_ptr<InputParameter> Ptr;
        virtual bool SetInput(const std::string& name_) = 0;
        virtual bool SetInput(Parameter*param) = 0;
        virtual Parameter* GetInput() = 0;
        virtual bool AcceptsInput(Parameter* param) = 0;
        virtual bool AcceptsType(const Loki::TypeInfo& type) = 0;
        virtual void SetQualifier(const std::function<bool(Parameter*)>& f)
        {
            qualifier = f;
        }
        virtual Loki::TypeInfo GetTypeInfo() = 0;
    protected:
        qualifier_f qualifier;
    };
}