#include "MetaObject/object/IMetaObject.hpp"

#include "MetaObject/core/Demangle.hpp"
#include "MetaObject/logging/logging.hpp"
#include "MetaObject/object/RelayManager.hpp"
#include "MetaObject/object/detail/IMetaObjectImpl.hpp"
#include "MetaObject/object/detail/IMetaObject_pImpl.hpp"
#include "MetaObject/params/IParam.hpp"
#include "MetaObject/params/InputParam.hpp"
#include "MetaObject/params/InputParam.hpp"
#include "MetaObject/params/VariableManager.hpp"
#include "MetaObject/params/buffers/BufferFactory.hpp"
#include "MetaObject/signals/ISignal.hpp"
#include "MetaObject/signals/ISlot.hpp"
#include "MetaObject/signals/SignalInfo.hpp"
#include "MetaObject/signals/SlotInfo.hpp"

#include "RuntimeObjectSystem/IObjectState.hpp"
#include "RuntimeObjectSystem/ISimpleSerializer.h"

#include <boost/thread/recursive_mutex.hpp>

namespace mo
{
    IMetaObject::~IMetaObject() {}

    int IMetaObject::connect(IMetaObject* sender,
                             const std::string& signal_name,
                             IMetaObject* receiver,
                             const std::string& slot_name)
    {
        int count = 0;
        auto my_signals = sender->getSignals(signal_name);
        auto my_slots = receiver->getSlots(slot_name);

        for (auto signal : my_signals)
        {
            for (auto slot : my_slots)
            {
                if (signal->getSignature() == slot->getSignature())
                {
                    auto connection = slot->connect(signal);
                    if (connection)
                    {
                        sender->addConnection(
                            std::move(connection), signal_name, slot_name, slot->getSignature(), receiver);
                        ++count;
                    }
                    break;
                }
                else
                {
                    MO_LOG(debug) << "Signature mismatch, Slot (" << slot_name << " -  " << slot->getSignature().name()
                                  << ") != Signal (" << signal_name << " - " << signal->getSignature().name() << ")";
                }
            }
        }

        return count;
    }

    bool IMetaObject::connect(IMetaObject* sender,
                              const std::string& signal_name,
                              IMetaObject* receiver,
                              const std::string& slot_name,
                              const TypeInfo& signature)
    {
        auto signal = sender->getSignal(signal_name, signature);
        if (signal)
        {
            auto slot = receiver->getSlot(slot_name, signature);
            if (slot)
            {
                auto connection = slot->connect(signal);
                if (connection)
                {
                    sender->addConnection(std::move(connection), signal_name, slot_name, signature, receiver);
                    return true;
                }
            }
        }
        return false;
    }

    MetaObject::MetaObject()
    {
        _pimpl = new impl();
        _sig_manager = nullptr;
        _pimpl->_slot_param_updated = std::bind(&MetaObject::onParamUpdate,
                                                this,
                                                std::placeholders::_1,
                                                std::placeholders::_2,
                                                std::placeholders::_3,
                                                std::placeholders::_4,
                                                std::placeholders::_5,
                                                std::placeholders::_6);
    }

    MetaObject::~MetaObject()
    {
        if (_pimpl->_variable_manager)
        {
            _pimpl->_variable_manager->removeParam(this);
        }
        delete _pimpl;
    }

    Mutex_t& MetaObject::getMutex() const { return _pimpl->m_mutex; }

    void IMetaObject::Init(bool firstInit) {}

    void MetaObject::Init(bool firstInit)
    {
        initParams(firstInit);
        initSignals(firstInit);
        bindSlots(firstInit);
        initCustom(firstInit);
        auto params = getParams();
        for (auto param : params)
        {
            auto update_slot = getSlot<mo::UpdateSig_t>("on_" + param->getName() + "_modified");
            if (update_slot)
            {
                auto connection = param->registerUpdateNotifier(update_slot);
                if (connection)
                {
                    this->addConnection(std::move(connection),
                                        param->getName() + "_modified",
                                        "on_" + param->getName() + "_modified",
                                        update_slot->getSignature(),
                                        this);
                }
            }
            auto delete_slot = getSlot<void(mo::IParam const*)>("on_" + param->getName() + "_deleted");
            if (delete_slot)
            {
                auto connection = param->registerDeleteNotifier(delete_slot);
                if (connection)
                {
                    this->addConnection(std::move(connection),
                                        param->getName() + "_deleted",
                                        "on_" + param->getName() + "_modified",
                                        update_slot->getSignature(),
                                        this);
                }
            }
        }

        if (firstInit == false)
        {
            auto connections_copy = _pimpl->_param_connections;
            _pimpl->_param_connections.clear();
            for (auto& param_connection : connections_copy)
            {
                rcc::shared_ptr<IMetaObject> obj(param_connection.output_object);
                if (obj)
                {
                    auto output = obj->getOutput(param_connection.output_param);
                    if (output == nullptr)
                    {
                        MO_LOG(debug) << "Unable to find " << param_connection.output_param << " in "
                                      << obj->GetTypeName() << " reinitializing";
                        obj->initParams(firstInit);
                        output = obj->getOutput(param_connection.output_param);
                        if (output == nullptr)
                        {
                            MO_LOG(info) << "Unable to find " << param_connection.output_param << " in "
                                         << obj->GetTypeName() << " unable to reConnect "
                                         << param_connection.input_param << " from object " << this->GetTypeName();
                            continue;
                        }
                    }
                    auto input = this->getInput(param_connection.input_param);
                    if (input)
                    {
                        if (this->connectInput(input, obj.get(), output, param_connection.connection_type))
                        {
                            input->getInput(mo::OptionalTime());
                            MO_LOG(debug) << "Reconnected " << GetTypeName() << ":" << param_connection.input_param
                                          << " to " << obj->GetTypeName() << ":" << param_connection.output_param;
                        }
                        else
                        {
                            MO_LOG(info) << "Reconnect FAILED " << GetTypeName() << ":" << param_connection.input_param
                                         << " to " << obj->GetTypeName() << ":" << param_connection.output_param;
                        }
                    }
                    else
                    {
                        MO_LOG(debug) << "Unable to find input Param " << param_connection.input_param << " in object "
                                      << this->GetTypeName();
                    }
                }
                else
                {
                    MO_LOG(debug) << "Output object no longer exists for input [" << param_connection.input_param
                                  << "] expected output name [" << param_connection.output_param << "]";
                }
            }
            // Rebuild Connections
            for (auto& connection : _pimpl->_connections)
            {
                if (!connection.obj.empty())
                {
                    auto signal = this->getSignal(connection.signal_name, connection.signature);
                    auto slot = connection.obj->getSlot(connection.slot_name, connection.signature);
                    if (signal == nullptr)
                    {
                        MO_LOG(debug) << "Unable to find signal with name \"" << connection.signal_name
                                      << "\" and signature: " << connection.signature.name()
                                      << " in new object of type " << this->GetTypeName();
                    }
                    if (slot == nullptr)
                    {
                        connection.obj->bindSlots(firstInit);
                        slot = connection.obj->getSlot(connection.slot_name, connection.signature);
                        if (slot == nullptr)
                        {
                            MO_LOG(debug) << "Unable to find slot with name \"" << connection.slot_name
                                          << "\" and signature: " << connection.signature.name()
                                          << " in new object of type " << connection.obj->GetTypeName();
                        }
                    }
                    if (signal && slot)
                    {
                        auto connection_ = slot->connect(signal);
                        if (connection_)
                        {
                            connection.connection = connection_;
                        }
                        else
                        {
                            MO_LOG(info) << "Unable to reconnect signal \"" << connection.signal_name << "\" <"
                                         << signal->getSignature().name() << "> to slot \"" << connection.slot_name
                                         << "\" <" << slot->getSignature().name() << ">";
                        }
                    }
                }
            }
        }
    }

    void MetaObject::initCustom(bool /*firstInit*/) {}

    int MetaObject::setupSignals(RelayManager* manager)
    {
        _sig_manager = manager;
        int count = 0;
        for (auto& my_slots : _pimpl->_slots)
        {
            for (auto& slot : my_slots.second)
            {
                ConnectionInfo info;
                info.connection = manager->connect(slot.second, my_slots.first, this);
                info.slot_name = my_slots.first;
                info.signature = slot.first;
                _pimpl->_connections.push_back(info);
                ++count;
            }
        }

        for (auto& my_signals : _pimpl->_signals)
        {
            for (auto& signal : my_signals.second)
            {
                auto Connection = manager->connect(signal.second, my_signals.first, this);
                ConnectionInfo info;
                info.signal_name = my_signals.first;
                info.signature = signal.first;
                info.connection = Connection;
                _pimpl->_connections.push_back(info);
                ++count;
            }
        }

        return count;
    }

    RelayManager* MetaObject::getRelayManager()
    {
        return _sig_manager;
    }

    int IMetaObject::setupVariableManager(IVariableManager* /*manager*/) { return 0; }

    int MetaObject::setupVariableManager(IVariableManager* manager)
    {
        if (_pimpl->_variable_manager != nullptr)
        {
            removeVariableManager(_pimpl->_variable_manager);
        }
        _pimpl->_variable_manager = manager;
        int count = 0;
        for (auto& param : _pimpl->_implicit_params)
        {
            manager->addParam(this, param.second.get());
            ++count;
        }
        for (auto& param : _pimpl->_params)
        {
            manager->addParam(this, param.second);
            ++count;
        }
        return count;
    }

    int MetaObject::removeVariableManager(IVariableManager* mgr)
    {
        int count = 0;
        mgr->removeParam(this);
        return count;
    }

    void IMetaObject::Serialize(ISimpleSerializer* pSerializer) {}

    void MetaObject::Serialize(ISimpleSerializer* pSerializer)
    {
        IObject::Serialize(pSerializer);
        serializeConnections(pSerializer);
        serializeParams(pSerializer);
    }

    void MetaObject::serializeConnections(ISimpleSerializer* pSerializer)
    {
        SERIALIZE(_pimpl->_connections);
        SERIALIZE(_pimpl->_param_connections);
        SERIALIZE(_ctx);
        SERIALIZE(_sig_manager);
    }

    void IMetaObject::setContext(const std::shared_ptr<Context>& ctx, bool overwrite) {}

    void MetaObject::setContext(const std::shared_ptr<Context>& ctx, bool overwrite)
    {
        if (_ctx.get() && overwrite == false)
            return;
        if (ctx == nullptr)
            MO_LOG(info) << "Setting context to nullptr";
        _ctx = ctx;
        for (auto& param : _pimpl->_implicit_params)
        {
            param.second->setContext(ctx.get());
        }
        for (auto& param : _pimpl->_params)
        {
            param.second->setContext(ctx.get());
        }
    }

    int MetaObject::disconnectByName(const std::string& name)
    {
        auto my_signals = this->getSignals(name);
        int count = 0;
        for (auto& sig : my_signals)
        {
            count += sig->disconnect() ? 1 : 0;
        }
        return count;
    }

    bool MetaObject::disconnect(ISignal* sig) { return false; }

    int MetaObject::disconnect(IMetaObject* obj)
    {
        auto obj_signals = obj->getSignals();
        int count = 0;
        for (auto signal : obj_signals)
        {
            count += disconnect(signal.first) ? 1 : 0;
        }
        return count;
    }

    std::vector<IParam*> MetaObject::getDisplayParams() const
    {
        std::vector<IParam*> output;
        for (auto& param : _pimpl->_params)
        {
            output.push_back(param.second);
        }
        for (auto& param : _pimpl->_implicit_params)
        {
            output.push_back(param.second.get());
        }
        return output;
    }

    IParam* MetaObject::getParam(const std::string& name) const
    {
        auto param = this->getParamOptional(name);
        if (!param)
            THROW(debug) << "Param with name \"" << name << "\" not found";
        return nullptr;
    }

    std::vector<IParam*> IMetaObject::getParams(const std::string& /*filter*/) const { return {}; }

    std::vector<IParam*> MetaObject::getParams(const std::string& filter) const
    {
        std::vector<IParam*> output;
        for (auto& itr : _pimpl->_params)
        {
            if (filter.size())
            {
                if (itr.first.find(filter) != std::string::npos)
                    output.push_back(itr.second);
            }
            else
            {
                output.push_back(itr.second);
            }
        }
        for (auto& itr : _pimpl->_implicit_params)
        {
            if (filter.size())
            {
                if (itr.first.find(filter) != std::string::npos)
                    output.push_back(itr.second.get());
            }
            else
            {
                output.push_back(itr.second.get());
            }
        }
        return output;
    }

    std::vector<IParam*> MetaObject::getParams(const TypeInfo& filter) const
    {
        std::vector<IParam*> output;
        for (auto& itr : _pimpl->_params)
        {
            if (itr.second->getTypeInfo() == filter)
                output.push_back(itr.second);
        }
        for (auto& itr : _pimpl->_implicit_params)
        {
            if (itr.second->getTypeInfo() == filter)
                output.push_back(itr.second.get());
        }
        return output;
    }

    std::vector<std::shared_ptr<IParam>> MetaObject::getImplicitParams() const
    {
        std::vector<std::shared_ptr<IParam>> output;
        for (const auto& param : _pimpl->_implicit_params)
            output.emplace_back(param.second);
        return output;
    }

    IParam* MetaObject::getParamOptional(const std::string& name) const
    {
        auto itr = _pimpl->_params.find(name);
        if (itr != _pimpl->_params.end())
        {
            return itr->second;
        }
        auto itr2 = _pimpl->_implicit_params.find(name);
        if (itr2 != _pimpl->_implicit_params.end())
        {
            return itr2->second.get();
        }
        MO_LOG(trace) << "Param with name \"" << name << "\" not found";
        return nullptr;
    }

    InputParam* MetaObject::getInput(const std::string& name) const
    {
        auto itr = _pimpl->_input_Params.find(name);
        if (itr != _pimpl->_input_Params.end())
        {
            return itr->second;
        }
        return nullptr;
    }

    std::shared_ptr<Context> MetaObject::getContext() { return _ctx; }

    std::vector<InputParam*> MetaObject::getInputs(const std::string& name_filter) const
    {
        std::vector<InputParam*> output;
        for (auto param : _pimpl->_input_Params)
        {
            if (name_filter.size())
            {
                if (param.second->getName().find(name_filter) != std::string::npos)
                    output.push_back(param.second);
            }
            else
            {
                output.push_back(param.second);
            }
        }
        return output;
    }

    std::vector<InputParam*> MetaObject::getInputs(const TypeInfo& type_filter, const std::string& name_filter) const
    {
        std::vector<InputParam*> output;
        for (auto param : _pimpl->_params)
        {
            if (param.second->checkFlags(ParamFlags::Input_e))
            {
                if (param.second->getTypeInfo() == type_filter)
                {
                    if (name_filter.size())
                    {
                        if (name_filter.find(param.first) != std::string::npos)
                            if (auto out = dynamic_cast<InputParam*>(param.second))
                                output.push_back(out);
                    }
                    else
                    {
                        if (auto out = dynamic_cast<InputParam*>(param.second))
                            output.push_back(out);
                    }
                }
            }
        }
        for (auto param : _pimpl->_implicit_params)
        {
            if (param.second->checkFlags(ParamFlags::Input_e))
            {
                if (param.second->getTypeInfo() == type_filter)
                {
                    if (name_filter.size())
                    {
                        if (name_filter.find(param.first) != std::string::npos)
                            if (auto out = dynamic_cast<InputParam*>(param.second.get()))
                                output.push_back(out);
                    }
                    else
                    {
                        if (auto out = dynamic_cast<InputParam*>(param.second.get()))
                            output.push_back(out);
                    }
                }
            }
        }
        return output;
    }

    IParam* IMetaObject::getOutput(const std::string& /*name*/) const { return nullptr; }

    IParam* MetaObject::getOutput(const std::string& name) const
    {
        auto itr = _pimpl->_params.find(name);
        if (itr != _pimpl->_params.end())
        {
            if (itr->second->checkFlags(mo::ParamFlags::Output_e))
            {
                return itr->second;
            }
        }
        auto itr2 = _pimpl->_implicit_params.find(name);
        if (itr2 != _pimpl->_implicit_params.end())
        {
            if (itr->second->checkFlags(mo::ParamFlags::Output_e))
            {
                return itr2->second.get();
            }
        }
        // to a pass through all params to see if the param was renamed
        for (const auto& itr : _pimpl->_params)
        {
            if (itr.second->getName() == name)
            {
                if (itr.second->checkFlags(mo::ParamFlags::Output_e))
                {
                    return itr.second;
                }
            }
        }
        for (const auto& itr : _pimpl->_implicit_params)
        {
            if (itr.second->getName() == name)
            {
                if (itr.second->checkFlags(mo::ParamFlags::Output_e))
                {
                    return itr.second.get();
                }
            }
        }
        return nullptr;
    }

    std::vector<IParam*> IMetaObject::getOutputs(const std::string& /*name_filter*/) const { return {}; }

    std::vector<IParam*> MetaObject::getOutputs(const std::string& name_filter) const
    {
        std::vector<IParam*> output;
        for (auto param : _pimpl->_params)
        {
            if (param.second->checkFlags(ParamFlags::Output_e))
            {
                if (name_filter.size())
                {
                    if (param.first.find(name_filter) != std::string::npos)
                    {
                        output.push_back(param.second);
                    }
                }
                else
                {
                    output.push_back(param.second);
                }
            }
        }
        for (auto param : _pimpl->_implicit_params)
        {
            if (param.second->checkFlags(ParamFlags::Output_e))
            {
                if (name_filter.size())
                {
                    if (param.first.find(name_filter) != std::string::npos)
                    {
                        output.push_back(param.second.get());
                    }
                }
                else
                {
                    output.push_back(param.second.get());
                }
            }
        }
        return output;
    }

    std::vector<IParam*> IMetaObject::getOutputs(const TypeInfo& type_filter, const std::string& name_filter) const
    {
        return {};
    }

    std::vector<IParam*> MetaObject::getOutputs(const TypeInfo& type_filter, const std::string& name_filter) const
    {
        std::vector<IParam*> output;
        for (auto param : _pimpl->_params)
        {
            if (param.second->checkFlags(ParamFlags::Output_e))
            {
                if (name_filter.size())
                {
                    if (name_filter.find(param.first) != std::string::npos)
                    {
                        if (param.second->getTypeInfo() == type_filter)
                            output.push_back(param.second);
                    }
                }
                else
                {
                    if (param.second->getTypeInfo() == type_filter)
                        output.push_back(param.second);
                }
            }
        }
        for (auto param : _pimpl->_implicit_params)
        {
            if (param.second->checkFlags(ParamFlags::Output_e))
            {
                if (name_filter.size())
                {
                    if (name_filter.find(param.first) != std::string::npos)
                    {
                        if (param.second->getTypeInfo() == type_filter)
                            output.push_back(param.second.get());
                    }
                }
                else
                {
                    if (param.second->getTypeInfo() == type_filter)
                        output.push_back(param.second.get());
                }
            }
        }
        return output;
    }

    bool IMetaObject::connectInput(const std::string& /*input_name*/,
                                   IMetaObject* /*output_object*/,
                                   IParam* /*output*/,
                                   BufferFlags /*type_*/)
    {
        return false;
    }

    bool
    MetaObject::connectInput(const std::string& input_name, IMetaObject* output_object, IParam* output, BufferFlags type_)
    {
        auto input = getInput(input_name);
        if (input && output)
            return connectInput(input, output_object, output, type_);

        auto inputs = getInputs();
        auto print_inputs = [inputs]() -> std::string {
            std::stringstream ss;
            for (auto _input : inputs)
            {
                ss << dynamic_cast<IParam*>(_input)->getName() << ", ";
            }
            return ss.str();
        };
        MO_LOG(debug) << "Unable to find input by name " << input_name << " in object " << this->GetTypeName()
                      << " with inputs [" << print_inputs() << "]";
        return false;
    }

    // Overriden below for MetaObject
    bool IMetaObject::connectInput(InputParam* /*input*/,
                                   IMetaObject* /*output_object*/,
                                   IParam* /*output*/,
                                   BufferFlags /*type_*/)
    {
        return false;
    }

    bool MetaObject::connectInput(InputParam* input, IMetaObject* output_object, IParam* output, BufferFlags type_)
    {
        if (input == nullptr || output == nullptr)
        {
            MO_LOG(debug) << "NULL input or output passed in";
            return false;
        }

        if (input && input->acceptsInput(output))
        {
            // Check contexts to see if a buffer needs to be setup
            auto output_ctx = output->getContext();
            auto input_context = input->getContext();
            if (type_ == Default_e && output_ctx != input_context)
            {
                type_ = getDefaultBufferType(output_ctx, input_context);
            }
            if (type_ & ForceBufferedConnection_e || input->checkFlags(mo::ParamFlags::RequestBuffered_e) ||
                output->checkFlags(mo::ParamFlags::RequestBuffered_e))
            {
                type_ = BufferFlags(type_ & ~ForceBufferedConnection_e);
                auto buffer = Buffer::BufferFactory::createProxy(output, type_);
                if (!buffer)
                {
                    MO_LOG(warning) << "Unable to create " << BufferFlagsToString(type_) << " for datatype "
                                    << Demangle::typeToName(output->getTypeInfo());
                    return false;
                }
                std::string buffer_type = BufferFlagsToString(type_);
                buffer->setName(output->getTreeName() + " " + buffer_type + " buffer for " + input->getTreeName());
                if (input->setInput(buffer))
                {
                    _pimpl->_param_connections.emplace_back(output_object, output->getName(), input->getName(), type_);
                    return true;
                }
                else
                {
                    MO_LOG(debug) << "Failed to connect output " << output->getName() << "["
                                  << Demangle::typeToName(output->getTypeInfo()) << "] to input "
                                  << dynamic_cast<IParam*>(input)->getName() << "["
                                  << Demangle::typeToName(dynamic_cast<IParam*>(input)->getTypeInfo()) << "]";
                    return false;
                }
            }
            if (output_ctx && _ctx.get())
            {
                if (output_ctx->thread_id != _ctx.get()->thread_id)
                {
                    auto buffer = Buffer::BufferFactory::createProxy(output, type_);
                    if (buffer)
                    {
                        buffer->setName(output->getTreeName() + " buffer for " + input->getTreeName());
                        if (input->setInput(buffer))
                        {
                            _pimpl->_param_connections.emplace_back(
                                output_object, output->getName(), input->getName(), type_);
                            return true;
                        }
                        else
                        {
                            MO_LOG(debug) << "Failed to connect output " << output->getName() << "["
                                          << Demangle::typeToName(output->getTypeInfo()) << "] to input "
                                          << dynamic_cast<IParam*>(input)->getName() << "["
                                          << Demangle::typeToName(dynamic_cast<IParam*>(input)->getTypeInfo()) << "]";
                            return false;
                        }
                    }
                    else
                    {
                        MO_LOG(debug) << "No buffer of desired type found for type "
                                      << Demangle::typeToName(output->getTypeInfo());
                    }
                }
                else
                {
                    if (input->setInput(output))
                    {
                        _pimpl->_param_connections.emplace_back(
                            output_object, output->getName(), input->getName(), type_);
                        return true;
                    }
                    else
                    {
                        MO_LOG(debug) << "Failed to connect output " << output->getName() << "["
                                      << Demangle::typeToName(output->getTypeInfo()) << "] to input "
                                      << dynamic_cast<IParam*>(input)->getName() << "["
                                      << Demangle::typeToName(dynamic_cast<IParam*>(input)->getTypeInfo()) << "]";
                        return false;
                    }
                }
            }
            else
            {
                if (input->setInput(output))
                {
                    _pimpl->_param_connections.emplace_back(output_object, output->getName(), input->getName(), type_);
                    return true;
                }
                else
                {
                    MO_LOG(debug) << "Failed to connect output " << output->getName() << "["
                                  << Demangle::typeToName(output->getTypeInfo()) << "] to input "
                                  << dynamic_cast<IParam*>(input)->getName() << "["
                                  << Demangle::typeToName(dynamic_cast<IParam*>(input)->getTypeInfo()) << "]";
                    return false;
                }
            }
        }
        MO_LOG(debug) << "Input \"" << input->getTreeName()
                      << "\"  does not accept input of type: " << Demangle::typeToName(output->getTypeInfo());
        return false;
    }

    bool IMetaObject::connectInput(
        IMetaObject* out_obj, IParam* out_param, IMetaObject* in_obj, InputParam* in_param, BufferFlags type)
    {
        return in_obj->connectInput(in_param, out_obj, out_param, type);
    }

    IParam* IMetaObject::addParam(std::shared_ptr<IParam> param) { return nullptr; }

    IParam* MetaObject::addParam(std::shared_ptr<IParam> param)
    {
        param->setMtx(&getMutex());
        param->setContext(_ctx.get());
#ifdef _DEBUG
        for (auto& param_ : _pimpl->_params)
        {
            if (param_.second == param.get())
            {
                MO_LOG(debug) << "Trying to add a param a second time";
                return param.get();
            }
        }
#endif
        _pimpl->_implicit_params[param->getName()] = param;
        if (param->checkFlags(ParamFlags::Input_e))
        {
            _pimpl->_input_Params[param->getName()] = dynamic_cast<InputParam*>(param.get());
        }
        auto connection = param->registerUpdateNotifier(&(this->_pimpl->_slot_param_updated));
        _pimpl->_sig_param_added(this, param.get());
        return param.get();
    }

    IParam* IMetaObject::addParam(IParam* param) { return nullptr; }

    IParam* MetaObject::addParam(IParam* param)
    {
        param->setMtx(&getMutex());
        param->setContext(_ctx.get());
#ifdef _DEBUG
        for (auto& param_ : _pimpl->_params)
        {
            if (param_.second == param)
            {
                MO_LOG(debug) << "Trying to add a Param a second time";
                return param;
            }
        }
#endif
        _pimpl->_params[param->getName()] = param;
        if (param->checkFlags(ParamFlags::Input_e))
        {
            _pimpl->_input_Params[param->getName()] = dynamic_cast<InputParam*>(param);
        }
        auto connection = param->registerUpdateNotifier(&(this->_pimpl->_slot_param_updated));
        _pimpl->_sig_param_added(this, param);
        this->addConnection(std::move(connection),
                            "param_updated",
                            "param_updated",
                            this->_pimpl->_slot_param_updated.getSignature(),
                            this);
        return param;
    }

    void MetaObject::setParamRoot(const std::string& root)
    {
        for (auto& param : _pimpl->_params)
        {
            param.second->setTreeRoot(root);
        }
        for (auto& param : _pimpl->_implicit_params)
        {
            param.second->setTreeRoot(root);
        }
    }

    std::vector<ParamInfo*> MetaObject::getParamInfo(const std::string& /*name*/) const
    {
        std::vector<ParamInfo*> output;
        getParamInfo(output);

        return output;
    }

    std::vector<ParamInfo*> MetaObject::getParamInfo() const
    {
        std::vector<ParamInfo*> output;
        getParamInfo(output);

        return output;
    }

    std::vector<SignalInfo*> MetaObject::getSignalInfo(const std::string& name) const
    {
        std::vector<SignalInfo*> info;
        getSignalInfo(info);

        return info;
    }

    std::vector<SignalInfo*> MetaObject::getSignalInfo() const
    {
        std::vector<SignalInfo*> info;
        getSignalInfo(info);
        return info;
    }

    std::vector<SlotInfo*> MetaObject::getSlotInfo() const
    {
        std::vector<SlotInfo*> output;
        getSlotInfo(output);
        return output;
    }

    std::vector<SlotInfo*> MetaObject::getSlotInfo(const std::string& name) const
    {
        std::vector<SlotInfo*> tmp;
        getSlotInfo(tmp);
        std::vector<SlotInfo*> output;
        for (auto& itr : tmp)
        {
            if (itr->name.find(name) != std::string::npos)
                output.push_back(itr);
        }

        return output;
    }

    std::vector<std::pair<ISlot*, std::string>> MetaObject::getSlots() const
    {
        std::vector<std::pair<ISlot*, std::string>> my_slots;
        for (auto itr1 : _pimpl->_slots)
        {
            for (auto itr2 : itr1.second)
            {
                my_slots.push_back(std::make_pair(itr2.second, itr1.first));
            }
        }
        return my_slots;
    }

    std::vector<ISlot*> MetaObject::getSlots(const std::string& name) const
    {
        std::vector<ISlot*> output;
        auto itr = _pimpl->_slots.find(name);
        if (itr != _pimpl->_slots.end())
        {
            for (auto slot : itr->second)
            {
                output.push_back(slot.second);
            }
        }
        return output;
    }

    std::vector<std::pair<ISlot*, std::string>> MetaObject::getSlots(const TypeInfo& signature) const
    {
        std::vector<std::pair<ISlot*, std::string>> output;
        for (auto& type : _pimpl->_slots)
        {
            auto itr = type.second.find(signature);
            if (itr != type.second.end())
            {
                output.push_back(std::make_pair(itr->second, type.first));
            }
        }
        return output;
    }

    ISlot* MetaObject::getSlot(const std::string& name, const TypeInfo& signature) const
    {
        auto itr1 = _pimpl->_slots.find(name);
        if (itr1 != _pimpl->_slots.end())
        {
            auto itr2 = itr1->second.find(signature);
            if (itr2 != itr1->second.end())
            {
                return itr2->second;
            }
        }
        if (name == "param_updated")
        {
            return &(_pimpl->_slot_param_updated);
        }
        return nullptr;
    }

    TSlot<void(IParam*, Context*, OptionalTime, size_t, const std::shared_ptr<ICoordinateSystem>&, UpdateFlags)>*
    MetaObject::getSlot_param_updated() const
    {
        return &_pimpl->_slot_param_updated;
    }

    bool MetaObject::connectByName(const std::string& name, ISlot* slot)
    {
        auto signal = getSignal(name, slot->getSignature());
        if (signal)
        {
            auto connection = signal->connect(slot);
            if (connection)
            {
                addConnection(std::move(connection), name, "", slot->getSignature());
                return true;
            }
        }
        return false;
    }

    bool MetaObject::connectByName(const std::string& name, ISignal* signal)
    {
        auto slot = getSlot(name, signal->getSignature());
        if (slot)
        {
            auto connection = slot->connect(signal);
            if (connection)
            {
                addConnection(std::move(connection), "", name, signal->getSignature());
                return true;
            }
        }
        return false;
    }

    int MetaObject::connectByName(const std::string& /*name*/, RelayManager* /*mgr*/)
    {
        // TODO implement
        return 0;
    }

    int MetaObject::connectByName(const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name)
    {
        int count = 0;
        auto my_signals = getSignals(signal_name);
        auto my_slots = receiver->getSlots(slot_name);
        for (auto signal : my_signals)
        {
            for (auto slot : my_slots)
            {
                if (signal->getSignature() == slot->getSignature())
                {
                    auto connection = slot->connect(signal);
                    if (connection)
                    {
                        addConnection(std::move(connection), signal_name, slot_name, slot->getSignature(), receiver);
                        ++count;
                        break;
                    }
                }
            }
        }
        return count;
    }

    bool MetaObject::connectByName(const std::string& signal_name,
                                   IMetaObject* receiver,
                                   const std::string& slot_name,
                                   const TypeInfo& signature)
    {
        auto signal = getSignal(signal_name, signature);
        auto slot = receiver->getSlot(slot_name, signature);
        if (signal && slot)
        {
            auto connection = slot->connect(signal);
            if (connection)
            {
                addConnection(std::move(connection), signal_name, slot_name, signature, receiver);
                return true;
            }
        }
        return false;
    }

    int MetaObject::connectAll(RelayManager* mgr)
    {
        auto my_signals = getSignalInfo();
        int count = 0;
        for (auto& signal : my_signals)
        {
            count += connectByName(signal->name, mgr);
        }
        return count;
    }

    void MetaObject::addSlot(ISlot* slot, const std::string& name)
    {
        _pimpl->_slots[name][slot->getSignature()] = slot;
        slot->setParent(this);
    }

    void MetaObject::addSignal(ISignal* sig, const std::string& name)
    {
        _pimpl->_signals[name][sig->getSignature()] = sig;
        sig->setParent(this);
        if (_sig_manager)
        {
            auto Connection = _sig_manager->connect(sig, name, this);
            ConnectionInfo info;
            info.signal_name = name;
            info.signature = sig->getSignature();
            info.connection = Connection;
            _pimpl->_connections.push_back(info);
        }
    }

    std::vector<std::pair<ISignal*, std::string>> MetaObject::getSignals() const
    {
        std::vector<std::pair<ISignal*, std::string>> my_signals;
        for (auto& name_itr : _pimpl->_signals)
        {
            for (auto& sig_itr : name_itr.second)
            {
                my_signals.push_back(std::make_pair(sig_itr.second, name_itr.first));
            }
        }
        return my_signals;
    }

    std::vector<ISignal*> MetaObject::getSignals(const std::string& name) const
    {
        std::vector<ISignal*> my_signals;
        auto itr = _pimpl->_signals.find(name);
        if (itr != _pimpl->_signals.end())
        {
            for (auto& sig_itr : itr->second)
            {
                my_signals.push_back(sig_itr.second);
            }
        }
        return my_signals;
    }

    std::vector<std::pair<ISignal*, std::string>> MetaObject::getSignals(const TypeInfo& type) const
    {
        std::vector<std::pair<ISignal*, std::string>> my_signals;
        for (auto& name_itr : _pimpl->_signals)
        {
            auto type_itr = name_itr.second.find(type);
            if (type_itr != name_itr.second.end())
            {
                my_signals.push_back(std::make_pair(type_itr->second, name_itr.first));
            }
        }
        return my_signals;
    }

    ISignal* MetaObject::getSignal(const std::string& name, const TypeInfo& type) const
    {
        auto name_itr = _pimpl->_signals.find(name);
        if (name_itr != _pimpl->_signals.end())
        {
            auto type_itr = name_itr->second.find(type);
            if (type_itr != name_itr->second.end())
            {
                return type_itr->second;
            }
        }
        if (name == "param_updated")
        {
            return &(_pimpl->_sig_param_updated);
        }
        if (name == "param_added")
        {
            return &(_pimpl->_sig_param_added);
        }
        return nullptr;
    }

    void MetaObject::addConnection(std::shared_ptr<Connection>&& connection,
                                   const std::string& signal_name,
                                   const std::string& slot_name,
                                   const TypeInfo& signature,
                                   IMetaObject* obj)
    {
        ConnectionInfo info;
        info.connection = std::move(connection);
        info.obj = rcc::weak_ptr<IMetaObject>(obj);
        info.signal_name = signal_name;
        info.slot_name = slot_name;
        info.signature = signature;
        _pimpl->_connections.push_back(info);
    }

    void IMetaObject::onParamUpdate(
        IParam* param, Context*, OptionalTime, size_t, const std::shared_ptr<ICoordinateSystem>&, UpdateFlags)
    {
    }

    void MetaObject::onParamUpdate(
        IParam* param, Context*, OptionalTime, size_t, const std::shared_ptr<ICoordinateSystem>&, UpdateFlags)
    {
        this->_pimpl->_sig_param_updated(this, param);
    }
} // namespace mo
