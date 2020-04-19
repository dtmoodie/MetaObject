#pragma once
#include <MetaObject/core/detail/forward.hpp>
#include <MetaObject/detail/TypeInfo.hpp>
#include <MetaObject/logging/logging.hpp>
#include <MetaObject/object/MetaObject.hpp>
#include <MetaObject/params/ITControlParam.hpp>

namespace mo
{
    class IMetaObject;

    template <class T>
    std::vector<ISubscriber*> MetaObject::getInputs(const std::string& name_filter) const
    {
        return getInputs(TypeInfo(typeid(T)), name_filter);
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
