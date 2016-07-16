#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/TypeInfo.h"
#include <string>
#include <memory>
#include <map>
#include <set>
namespace mo
{
    class ICallback;
    class Connection;
    class ISignal;
    class ISlot;
    class IParameter;
    struct MO_EXPORTS IMetaObject::impl
    {
        std::map<ISignal*, std::shared_ptr<Connection>>     connections;
        std::map<std::string, std::set<ICallback*>>        _callback_name_map;
        std::map<TypeInfo, std::set<ICallback*>>           _callback_signature_map;
        std::set<ICallback*>                                _explicit_callbacks;

        std::map<std::string, std::map<TypeInfo, std::weak_ptr<ISignal>>> _signals;
        std::map<std::string, std::map<TypeInfo, ISlot*>> _slots;

        std::map<std::string, IParameter*> _parameters; // statically defined in object
        std::map<std::string, std::shared_ptr<IParameter>> _implicit_parameters; // Can be changed at runtime
    };
}