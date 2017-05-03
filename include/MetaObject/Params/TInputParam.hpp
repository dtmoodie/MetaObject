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
/*template<typename T> class TInputParamCopy : virtual public ITInputParam<T> {
public:
    TInputParamCopy(const std::string& name, T* userVar_,
                            ParamType type = Control_e) {
        this->input = nullptr;
    }

    
    
protected:
    T* userVar; // Pointer to the user space variable of type T

    void onInputUpdate(Context* ctx, IParam* param) {
        if(this->input && userVar)
            this->input->GetData(*userVar, -1, this->getContext());
        IParam::OnUpdate(ctx);
    }
    void onInputDelete(IParam* param) {
        this->input = nullptr;
        IParam::OnUpdate(nullptr);
    }
};*/

// Meant to reference a pointer variable in user space, and to update that variable whenever
// IE int* myVar;
// auto TParam = TInputParamPtr(&myVar); // TInputParam now updates myvar to point to whatever the
// input variable is for TParam.
template<typename T> class MO_EXPORTS TInputParamPtr : virtual public ITInputParam<T> {
public:
    TInputParamPtr(const std::string& name = "", Input_t* userVar_ = nullptr, Context* ctx = nullptr);
    bool setInput(std::shared_ptr<IParam> input);
    bool setInput(IParam* input);
    void setUserDataPtr(Input_t* user_var_);
    bool getInput(OptionalTime_t ts, size_t* fn = nullptr);
    bool getInput(size_t fn, OptionalTime_t* ts = nullptr);

protected:
    virtual bool updateDataImpl(ConstStorageRef_t, OptionalTime_t, Context*, size_t, ICoordinateSystem*) {
        return true;
    }
    Input_t* _user_var; // Pointer to the user space pointer variable of type T
    InputStorage_t _current_data;
    virtual void onInputUpdate(ConstStorageRef_t, IParam*, Context*, OptionalTime_t, size_t, ICoordinateSystem*, UpdateFlags);
};
}
#include "MetaObject/Params/detail/TInputParamImpl.hpp"
