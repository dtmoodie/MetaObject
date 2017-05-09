#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/Enums.hpp"
#include "MetaObject/Detail/TypeInfo.hpp"
#include "MetaObject/Signals/TSignal.hpp"
#include "MetaObject/Params/IParam.hpp"
#include "RuntimeObjectSystem/shared_ptr.hpp"
#include <string>
#include <memory>
#include <map>
#include <set>
#include <list>
namespace mo {
class ICallback;
class Connection;
class ISignal;
class ISlot;
class IParam;
class IMetaObject;
struct MO_EXPORTS ConnectionInfo {
    ConnectionInfo() {
    }
    ConnectionInfo(const ConnectionInfo& info) {
        signal_name = info.signal_name;
        slot_name = info.slot_name;
        signature = info.signature;
        obj = info.obj;
        connection = info.connection;
    }
    std::string signal_name;
    std::string slot_name;
    TypeInfo signature;
    rcc::weak_ptr<IMetaObject> obj;
    std::shared_ptr<Connection> connection;
};

struct MO_EXPORTS ParamConnectionInfo {
    ParamConnectionInfo(const rcc::weak_ptr<IMetaObject>& obj, const std::string& output, const std::string& input, ParamType type) :
        output_object(obj), output_param(output), input_param(input), connection_type(type) {
    }
    rcc::weak_ptr<IMetaObject> output_object;
    std::string output_param;
    std::string input_param;
    ParamType connection_type;
};

struct MO_EXPORTS IMetaObject::impl {
    impl() {
        _variable_manager = nullptr;
        _signals["Param_updated"][_sig_param_updated.getSignature()] = &_sig_param_updated;
        _signals["Param_added"][_sig_param_updated.getSignature()] = &_sig_param_added;
    }
    std::map<std::string, std::map<TypeInfo, ISignal*>> _signals;
    std::map<std::string, std::map<TypeInfo, ISlot*>>   _slots;

    std::map<std::string, IParam*>				    _params; // statically defined in object

    std::map<std::string, std::shared_ptr<IParam>>  _implicit_params; // Can be changed at runtime
    std::list<ConnectionInfo> _connections;
    std::list<ParamConnectionInfo> _param_connections;

    TSignal<void(IMetaObject*, IParam*)>  _sig_param_updated;
    TSignal<void(IMetaObject*, IParam*)>  _sig_param_added;
    std::map<std::string, InputParam*>    _input_Params;
    TSlot<mo::UpdateSig_t>               _slot_param_updated;
    IVariableManager*                     _variable_manager;
};
}
