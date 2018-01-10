#pragma once
#include "MetaObject/detail/TypeInfo.hpp"
#include "MetaObject/logging/logging.hpp"
#include "MetaObject/params/ITParam.hpp"
#include "MetaObject/params/TParam.hpp"
#include "MetaObject/object/MetaObject.hpp"
namespace mo
{
    class IMetaObject;

    template <class T>
    ITParam<T>* MetaObject::getParam(const std::string& name) const
    {
        IParam* param = getParam(name);
        ITParam<T>* typed = dynamic_cast<ITParam<T>*>(param);
        if (typed)
        {
            return typed;
        }
        THROW(debug) << "Param \"" << name << "\" not convertable to type " << TypeInfo(typeid(T)).name();
        return nullptr;
    }

    template <class T>
    T MetaObject::getParamValue(const std::string& name, const OptionalTime_t& ts, Context* ctx) const
    {
        T data;
        MO_ASSERT(getParam<T>(name)->getData(data, ts, ctx));
        return data;
    }

    template <class T>
    ITParam<T>* MetaObject::getParamOptional(const std::string& name) const
    {
        auto param = getParamOptional(name);
        ITParam<T>* typed = dynamic_cast<ITParam<T>*>(param);
        return typed;
    }

    template <class T>
    ITParam<T>* MetaObject::updateParam(const std::string& name, T& value, const OptionalTime_t& ts, Context* ctx)
    {
        if (ctx == nullptr)
            ctx = getContext().get();
        auto param = getParamOptional<T>(name);
        if (param)
        {
            param->updateData(value, ts, ctx);
            return param;
        }
        else
        {
            std::shared_ptr<TParam<T>> new_param(new TParam<T>(name, value));
            addParam(new_param);
            return new_param.get();
        }
    }

    template <class T>
    ITParam<T>* MetaObject::updateParam(const std::string& name, const T& value, const OptionalTime_t& ts, Context* ctx)
    {
        if (ctx == nullptr)
            ctx = getContext().get();
        auto param = getParamOptional<T>(name);
        if (param)
        {
            param->updateData(value, ts, ctx);
            return param;
        }
        else
        {
            std::shared_ptr<ITParam<T>> new_param(new TParam<T>(name, value));
            addParam(new_param);
            return new_param.get();
        }
    }

    template <class T>
    ITParam<T>* MetaObject::updateParamPtr(const std::string& name, T& ptr)
    {
        return nullptr;
    }

    template <class T>
    TSlot<T>* MetaObject::getSlot(const std::string& name) const
    {
        return dynamic_cast<TSlot<T>*>(this->getSlot(name, TypeInfo(typeid(T))));
    }

    template <class T>
    std::vector<InputParam*> MetaObject::getInputs(const std::string& name_filter) const
    {
        return getInputs(TypeInfo(typeid(T)), name_filter);
    }

    template <class T>
    ITInputParam<T>* MetaObject::getInput(const std::string& name)
    {
        auto ptr = getInput(name);
        if (ptr)
        {
            return dynamic_cast<ITInputParam<T>*>(ptr);
        }
        return nullptr;
    }

    template <class T>
    ITParam<T>* MetaObject::getOutput(const std::string& name) const
    {
        auto ptr = getOutput(name);
        if (ptr)
        {
            if (ptr->getTypeInfo() == TypeInfo(typeid(T)))
            {
                return dynamic_cast<ITParam<T>*>(ptr);
            }
        }
        return nullptr;
    }

    template <class T>
    bool IMetaObject::connect(IMetaObject* sender,
                              const std::string& signal_name,
                              IMetaObject* receiver,
                              const std::string& slot_name)
    {
        return connect(sender, signal_name, receiver, slot_name, TypeInfo(typeid(T)));
    }
}
