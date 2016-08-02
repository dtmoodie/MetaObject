#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/TypeInfo.h"
#include "shared_ptr.hpp"
#include <string>
#include <memory>
#include <map>
#include <set>
#include <list>
namespace mo
{
    class ICallback;
    class Connection;
    class ISignal;
    class ISlot;
    class IParameter;
	class IMetaObject;
	struct MO_EXPORTS ConnectionInfo
	{
		std::string signal_name;
		std::string slot_name;
		TypeInfo signature;
		rcc::weak_ptr<IMetaObject> obj;
		std::shared_ptr<Connection> connection;
	};
    struct MO_EXPORTS IMetaObject::impl
    {
        std::map<std::string, std::map<TypeInfo, ISignal*>> _signals;
        std::map<std::string, std::map<TypeInfo, ISlot*>>   _slots;

        std::map<std::string, IParameter*>				    _parameters; // statically defined in object
        std::map<std::string, std::shared_ptr<IParameter>>  _implicit_parameters; // Can be changed at runtime
		std::list<ConnectionInfo> _connections;
    };
}