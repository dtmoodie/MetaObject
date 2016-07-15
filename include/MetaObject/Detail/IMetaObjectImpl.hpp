#pragma once

namespace mo
{
    class IMetaObject;

    template<class T> std::weak_ptr<ITypedParameter<T>> IMetaObject::GetParameter(const std::string& name) const
    {
        
    }

    template<class Sig> bool IMetaObject::ConnectCallback(const std::string& callback_name, const std::string& slot_name, IMetaObject* slot_owner, bool force_queue)
    {
        ConnectCallback(TypedInfo(typeid(Sig)), callback_name, slot_name, slot_owner, force_queue);
    }
}