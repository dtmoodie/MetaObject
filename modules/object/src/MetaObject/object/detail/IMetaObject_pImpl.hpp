#pragma once
#include "MetaObject/core/detail/Enums.hpp"
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/detail/TypeInfo.hpp"
#include "MetaObject/params/IParam.hpp"
#include "MetaObject/signals/TSignal.hpp"
#include "RuntimeObjectSystem/shared_ptr.hpp"
#include <boost/fiber/recursive_timed_mutex.hpp>

#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>

namespace mo
{
    class ICallback;
    class Connection;
    class ISignal;
    class ISlot;
    class IParam;
    class IMetaObject;
    struct MO_EXPORTS ConnectionInfo
    {
        ConnectionInfo()
        {
        }
        ConnectionInfo(const ConnectionInfo& info)
        {
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

    struct MO_EXPORTS ParamConnectionInfo
    {
        ParamConnectionInfo(const rcc::weak_ptr<IMetaObject>& obj,
                            const std::string& output,
                            const std::string& input,
                            BufferFlags type)
            : output_object(obj)
            , output_param(output)
            , input_param(input)
            , connection_type(type)
        {
        }
        rcc::weak_ptr<IMetaObject> output_object;
        std::string output_param;
        std::string input_param;
        BufferFlags connection_type;
    };

    struct MO_EXPORTS MetaObject::impl
    {
        impl()
        {
            _variable_manager = nullptr;
            _signals["param_updated"][_sig_param_updated.getSignature()] = &_sig_param_updated;
            _signals["param_added"][_sig_param_updated.getSignature()] = &_sig_param_added;
        }

        std::map<std::string, std::map<TypeInfo, ISignal*>> _signals;
        std::map<std::string, std::map<TypeInfo, ISlot*>> _slots;

        std::map<std::string, IParam*> _params; // statically defined in object

        std::map<std::string, std::shared_ptr<IParam>> _implicit_params; // Can be changed at runtime
        std::list<ConnectionInfo> _connections;
        std::list<ParamConnectionInfo> _param_connections;

        TSignal<void(IMetaObject*, IParam*)> _sig_param_updated;
        TSignal<void(IMetaObject*, IParam*)> _sig_param_added;
        std::map<std::string, InputParam*> _input_params;
        TSlot<Update_s> _slot_param_updated;
        IParamServer* _variable_manager;
        mutable mo::Mutex_t m_mutex;
    };
}
