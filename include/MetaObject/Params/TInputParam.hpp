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
template<typename T> class TInputParamCopy : virtual public ITInputParam<T> {
public:
    TInputParamCopy(const std::string& name, T* userVar_,
                            ParamType type = Control_e) {
        this->input = nullptr;
    }

    T* GetDataPtr(mo::Time_t ts = -1 * mo::second, Context* ctx = nullptr) {
        return userVar;
    }
    bool GetData(T& value, mo::Time_t ts = -1 * mo::second, Context* ctx = nullptr) {
        if (userVar) {
            value = *userVar;
            return true;
        }
        return false;
    }
    T GetData(mo::Time_t ts = -1 * mo::second, Context* ctx = nullptr) {
        if(this->input)
            return this->input->GetData(ts, ctx);
        if(this->shared_input)
            return this->shared_input->GetData(ts, ctx);
        return false;
    }
    void UpdateData(T& data_, mo::Time_t ts = -1 * mo::second, cv::cuda::Stream* stream = nullptr) {
        if(userVar)
            *userVar = data_;
    }
    void UpdateData(const T& data_, mo::Time_t ts = -1 * mo::second, cv::cuda::Stream* stream = nullptr) {
        if(userVar)
            *userVar = data_;
    }
    void UpdateData(T* data_, mo::Time_t ts = -1 * mo::second, cv::cuda::Stream* stream = nullptr) {
        if(userVar )
            *userVar = *data_;
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
};

// Meant to reference a pointer variable in user space, and to update that variable whenever
// IE int* myVar;
// auto TParam = TInputParamPtr(&myVar); // TInputParam now updates myvar to point to whatever the
// input variable is for TParam.
template<typename T> class MO_EXPORTS TInputParamPtr : virtual public ITInputParam<T> {
public:
    TInputParamPtr(const std::string& name = "", const T** userVar_ = nullptr, Context* ctx = nullptr);
    bool SetInput(std::shared_ptr<IParam> input);
    bool SetInput(IParam* input);
    void SetUserDataPtr(const T** user_var_);
    bool getInput(OptionalTime_t ts, size_t* fn = nullptr);
    bool getInput(size_t fn, OptionalTime_t* ts = nullptr);

protected:
    virtual bool UpdateDataImpl(const T& data, OptionalTime_t ts, Context* ctx, boost::optional<size_t> fn, ICoordinateSystem* cs) {
        return true;
    }
    const T** userVar; // Pointer to the user space pointer variable of type T
    void updateUserVar();
    virtual void onInputUpdate(Context* ctx, IParam* param);
    virtual void onInputDelete(IParam const* param);
};
}
#include "MetaObject/Params/detail/TInputParamImpl.hpp"
//#include "MetaObject/Params/detail/TInputParamPtrImpl.hpp"
