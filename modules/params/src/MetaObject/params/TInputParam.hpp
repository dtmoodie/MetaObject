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
#include "ITInputParam.hpp"

namespace mo {

// Meant to reference a pointer variable in user space, and to update that variable whenever
// IE int* myVar;
// auto TParam = TInputParamPtr(&myVar); // TInputParam now updates myvar to point to whatever the
// input variable is for TParam.
template <typename T>
class MO_EXPORTS TInputParamPtr : virtual public ITInputParam<T> {
public:
    typedef typename ParamTraits<T>::Storage_t         Storage_t;
    typedef typename ParamTraits<T>::ConstStorageRef_t ConstStorageRef_t;
    typedef typename ParamTraits<T>::InputStorage_t    InputStorage_t;
    typedef typename ParamTraits<T>::Input_t           Input_t;
    typedef void(TUpdateSig_t)(ConstStorageRef_t, IParam*, Context*, OptionalTime_t, size_t, const std::shared_ptr<ICoordinateSystem>&, UpdateFlags);
    typedef TSignal<TUpdateSig_t> TUpdateSignal_t;
    typedef TSlot<TUpdateSig_t>   TUpdateSlot_t;

    TInputParamPtr(const std::string& name = "", Input_t* userVar_ = nullptr, Context* ctx = nullptr);
    bool setInput(std::shared_ptr<IParam> input);
    bool setInput(IParam* input);
    void setUserDataPtr(Input_t* user_var_);
    bool getInput(const OptionalTime_t& ts, size_t* fn = nullptr);
    bool getInput(size_t fn, OptionalTime_t* ts = nullptr);

protected:
    virtual bool updateDataImpl(const Storage_t&, const OptionalTime_t&, Context*, size_t, const std::shared_ptr<ICoordinateSystem>&) {
        return true;
    }
    Input_t*       _user_var; // Pointer to the user space pointer variable of type T
    InputStorage_t _current_data;
    virtual void   onInputUpdate(ConstStorageRef_t, IParam*, Context*, OptionalTime_t, size_t, const std::shared_ptr<ICoordinateSystem>&, UpdateFlags);
};
}
#include "MetaObject/params/detail/TInputParamImpl.hpp"
#include "detail/TInputParamPtrImpl.hpp"
