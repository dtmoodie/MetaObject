#pragma once
#ifndef __CUDACC__
#include "../ITParam.hpp"
#include <MetaObject/thread/fiber_include.hpp>

namespace mo
{
    // *************************************************
    // TParamImpl

    template <typename T>
    TypeInfo TParamImpl<T>::getTypeInfo() const
    {
        return _type_info;
    }

    template <class T>
    std::shared_ptr<Connection> TParamImpl<T>::registerUpdateNotifier(UpdateSlot_t* f)
    {
        return IParam::registerUpdateNotifier(f);
    }

    template <class T>
    std::shared_ptr<Connection> TParamImpl<T>::registerUpdateNotifier(std::shared_ptr<TSignalRelay<UpdateSig_t>>& relay)
    {
        return IParam::registerUpdateNotifier(relay);
    }

    template <class T>
    std::shared_ptr<Connection> TParamImpl<T>::registerUpdateNotifier(TUpdateSlot_t* f)
    {
        return f->connect(&_typed_update_signal);
    }

    template <class T>
    std::shared_ptr<Connection> TParamImpl<T>::registerUpdateNotifier(ISlot* f)
    {
        if (f->getSignature() == TypeInfo(typeid(TUpdateSig_t)))
        {
            auto typed = dynamic_cast<TUpdateSlot_t*>(f);
            if (typed)
            {
                return registerUpdateNotifier(typed);
            }
            else
            {
            }
        }
        else
        {
            return IParam::registerUpdateNotifier(f);
        }
        return {};
    }

    template <class T>
    std::shared_ptr<Connection>
    TParamImpl<T>::registerUpdateNotifier(std::shared_ptr<TSignalRelay<TUpdateSig_t>>& relay)
    {
        return _typed_update_signal.connect(relay);
    }

    template <class T>
    std::shared_ptr<Connection> TParamImpl<T>::registerUpdateNotifier(std::shared_ptr<ISignalRelay> relay)
    {
        auto typed = std::dynamic_pointer_cast<TSignalRelay<TUpdateSig_t>>(relay);
        if (typed)
        {
            return registerUpdateNotifier(typed);
        }
        else
        {
            return IParam::registerUpdateNotifier(relay);
        }
    }

    template <typename T>
    const TypeInfo TParamImpl<T>::_type_info(typeid(T));
} // namespace mo
#endif
