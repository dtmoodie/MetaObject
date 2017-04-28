#pragma once
#include "MetaObject/Logging/Log.hpp"
#include "MetaObject/Detail/TypeInfo.hpp"
#include "MetaObject/Params/ITParam.hpp"
namespace mo {
class IMetaObject;

template<class T>
ITParam<T>* IMetaObject::getParam(const std::string& name) const {
    IParam* param = getParam(name);
    ITParam<T>* T = dynamic_cast<ITParam<T>*>(param);
    if(T) {
        return T;
    }
    THROW(debug) << "Param \"" << name << "\" not convertable to type " << TypeInfo(typeid(T)).name();
    return nullptr;
}

template<class T>
T IMetaObject::getParamValue(const std::string& name, mo::Time_t ts, Context* ctx) const {
    return getParam<T>(name)->GetData(ts, ctx);
}
template<class T>
ITParam<T>* IMetaObject::getParamOptional(const std::string& name) const {
    auto param = getParamOptional(name);
    ITParam<T>* T = dynamic_cast<ITParam<T>*>(param);
    return T;
}

template<class T>
ITParam<T>* IMetaObject::UpdateParam(const std::string& name, T& value, mo::Time_t  ts, Context* ctx) {
    if(ctx == nullptr)
        ctx = _ctx;
    auto param = getParamOptional<T>(name);
    if(param) {
        param->UpdateData(value, ts, ctx);
        return param;
    } else {
        std::shared_ptr<ITParam<T>> new_param(new TParam<T>(name, value));
        addParam(new_param);
        return new_param.get();
    }
}
template<class T>
ITParam<T>* IMetaObject::UpdateParam(const std::string& name, const T& value, mo::Time_t ts, Context* ctx) {
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
ITParam<T>* IMetaObject::UpdateParamPtr(const std::string& name, T& ptr) {
    return nullptr;
}

/*template<class Sig>
bool IMetaObject::connectCallback(const std::string& callback_name, const std::string& slot_name, IMetaObject* slot_owner, bool force_queue)
{
    connectCallback(TInfo(typeid(Sig)), callback_name, slot_name, slot_owner, force_queue);
}*/
template<class T>
TSlot<T>* IMetaObject::getSlot(const std::string& name) const {
    return dynamic_cast<TSlot<T>*>(this->getSlot(name, TypeInfo(typeid(T))));
}
/*template<class T>
std::vector<IParam*> IMetaObject::getOutputs(const std::string& name_filter) const {
    return getOutputs(TypeInfo(typeid(T)), name_filter);
}*/

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
