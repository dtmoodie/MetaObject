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

https://github.com/dtmoodie/Params
*/
#pragma once
#include "ITParam.hpp"
#include "ParamConstructor.hpp"
#include "MetaObject/Params/MetaParam.hpp"
namespace mo {
template<typename T>
class MO_EXPORTS TParam : virtual public ITParam<T> {
public:
    static const ParamType Type = TParam_e;
    TParam();

    virtual bool getData(Storage_t& data, const OptionalTime_t& ts = OptionalTime_t(),
        Context* ctx = nullptr, size_t* fn_ = nullptr);

    virtual bool getData(Storage_t& data, size_t fn, Context* ctx = nullptr, OptionalTime_t* ts_ = nullptr);

protected:
    virtual bool updateDataImpl(ConstStorageRef_t data, OptionalTime_t ts, Context* ctx, size_t fn, ICoordinateSystem* cs);
    Storage_t _data;
private:
    static ParamConstructor<TParam<T>> _typed_param_constructor;
    static MetaParam<T, 100> _meta_param;
};
}
#include "detail/TParamImpl.hpp"
