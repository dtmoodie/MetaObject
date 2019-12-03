#pragma once
#include <MetaObject/core/detail/forward.hpp>
#include <MetaObject/detail/TypeInfo.hpp>
#include <MetaObject/logging/logging.hpp>
#include <MetaObject/object/MetaObject.hpp>

namespace mo
{
    class IMetaObject;

    template <class T>
    TParam<T>* MetaObject::getParam(const std::string& name) const
    {
        IParam* param = getParam(name);
        TParam<T>* typed = dynamic_cast<TParam<T>*>(param);
        if (typed)
        {
            return typed;
        }
        THROW(debug, "Param {} not convertable to {}", name, TypeInfo(typeid(T)).name());
        return nullptr;
    }

    template <class T>
    T MetaObject::getParamValue(const std::string& name, const OptionalTime& ts, IAsyncStream* ctx) const
    {
        T data;
        MO_ASSERT(getParam<T>(name)->getData(data, ts, ctx));
        return data;
    }

    template <class T>
    TParam<T>* MetaObject::getParamOptional(const std::string& name) const
    {
        auto param = getParamOptional(name);
        TParam<T>* typed = dynamic_cast<TParam<T>*>(param);
        return typed;
    }

    template <class T>
    TParam<T>* MetaObject::updateParam(const std::string& name, T& value, const OptionalTime& ts)
    {
        auto param = getParamOptional<T>(name);
        if (param)
        {
            param->updateData(value, ts);
            return param;
        }

        std::shared_ptr<TParam<T>> new_param(new TParam<T>(name));
        new_param->updateData(value);
        addParam(new_param);
        return new_param.get();
    }

    template <class T>
    TParam<T>* MetaObject::updateParam(const std::string& name, const T& value, const OptionalTime& ts)
    {
        auto param = getParamOptional<T>(name);
        if (param)
        {
            param->updateData(value, ts);
            return param;
        }

        std::shared_ptr<TParam<T>> new_param(new TParam<T>(name, value));
        addParam(new_param);
        return new_param.get();
    }

    template <class T>
    TParam<T>* MetaObject::updateParamPtr(const std::string& name, T& ptr)
    {
        return nullptr;
    }

    template <class SIGNATURE>
    TSignal<SIGNATURE> IMetaObject::getSignal(const std::string& name)
    {
        return dynamic_cast<TSignal<SIGNATURE>*>(this->getSignal(name, TypeInfo(typeid(SIGNATURE))));
    }

    template <class T>
    TSlot<T>* IMetaObject::getSlot(const std::string& name)
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
    TParam<T>* MetaObject::getOutput(const std::string& name) const
    {
        auto ptr = getOutput(name);
        if (ptr)
        {
            if (ptr->getTypeInfo() == TypeInfo(typeid(T)))
            {
                return dynamic_cast<TParam<T>*>(ptr);
            }
        }
        return nullptr;
    }

    template <class T>
    bool IMetaObject::connect(IMetaObject& sender,
                              const std::string& signal_name,
                              IMetaObject& receiver,
                              const std::string& slot_name)
    {
        return connect(sender, signal_name, receiver, slot_name, TypeInfo(typeid(T)));
    }
} // namespace mo
