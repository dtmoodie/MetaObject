#pragma once
#include "MetaObject/Logging/Log.hpp"
#include "MetaObject/Detail/TypeInfo.hpp"
#include "MetaObject/Params/ITParam.hpp"
#include "MetaObject/Params/TParam.hpp"
namespace mo {
class IMetaObject;

template<class T>
ITParam<T>* IMetaObject::getParam(const std::string& name) const {
    IParam* param = getParam(name);
    ITParam<T>* typed = dynamic_cast<ITParam<T>*>(param);
    if(typed) {
        return typed;
    }
    THROW(debug) << "Param \"" << name << "\" not convertable to type " << TypeInfo(typeid(T)).name();
    return nullptr;
}

template<class T>
T IMetaObject::getParamValue(const std::string& name, const OptionalTime_t& ts, Context* ctx) const {
    T data;
    MO_ASSERT(getParam<T>(name)->getData(data, ts, ctx));
    return data;
}
template<class T>
ITParam<T>* IMetaObject::getParamOptional(const std::string& name) const {
    auto param = getParamOptional(name);
    ITParam<T>* typed = dynamic_cast<ITParam<T>*>(param);
    return typed;
}

template<class T>
ITParam<T>* IMetaObject::updateParam(const std::string& name, T& value, const OptionalTime_t& ts, Context* ctx) {
    if(ctx == nullptr)
        ctx = _ctx;
    auto param = getParamOptional<T>(name);
    if(param) {
        param->updateData(value, ts, ctx);
        return param;
    } else {
        std::shared_ptr<TParam<T>> new_param(new TParam<T>(name, value));
        addParam(new_param);
        return new_param.get();
    }
}
template<class T>
ITParam<T>* IMetaObject::updateParam(const std::string& name, const T& value, const OptionalTime_t& ts, Context* ctx) {
    if (ctx == nullptr)
        ctx = _ctx;
    auto param = getParamOptional<T>(name);
    if (param) {
        param->UpdateData(value, ts, ctx);
        return param;
    } else {
        std::shared_ptr<ITParam<T>> new_param(new TParam<T>(name, value));
        addParam(new_param);
        return new_param.get();
    }
}
template<class T>
ITParam<T>* IMetaObject::updateParamPtr(const std::string& name, T& ptr) {
    return nullptr;
}

template<class T>
TSlot<T>* IMetaObject::getSlot(const std::string& name) const {
    return dynamic_cast<TSlot<T>*>(this->getSlot(name, TypeInfo(typeid(T))));
}

template<class T>
std::vector<InputParam*> IMetaObject::getInputs(const std::string& name_filter) const {
    return getInputs(TypeInfo(typeid(T)), name_filter);
}

template<class T>
ITInputParam<T>* IMetaObject::getInput(const std::string& name) {
    auto ptr = getInput(name);
    if(ptr) {
        return dynamic_cast<ITInputParam<T>*>(ptr);
    }
    return nullptr;
}
template<class T>
ITParam<T>* IMetaObject::getOutput(const std::string& name) const {
    auto ptr = getOutput(name);
    if(ptr) {
        if(ptr->getTypeInfo() == TypeInfo(typeid(T))) {
            return dynamic_cast<ITParam<T>*>(ptr);
        }
    }
    return nullptr;
}
template<class T>
bool IMetaObject::connect(IMetaObject* sender, const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name) {
    return connect(sender, signal_name, receiver, slot_name, TypeInfo(typeid(T)));
}
}
