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
#include "ITParam.hpp"
#include "MetaObject/params/MetaParam.hpp"

namespace mo
{
    template <typename T>
    class MO_EXPORTS TParam : virtual public ITAccessibleParam<T>
    {
      public:
        TParam(const std::string& name, const T& value)
            : IParam(name)
        {
            ParamTraits<T>::reset(_data, value);
        }
        TParam(const std::string& name)
            : IParam(name)
        {
        }
        TParam();
    };
}
#include "detail/TParamImpl.hpp"
