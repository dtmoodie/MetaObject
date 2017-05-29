#include "MetaObject/object/IMetaObject.hpp"

#include "MetaObject/logging/Log.hpp"
#include "MetaObject/core/Demangle.hpp"
#include "MetaObject/signals/ISignal.hpp"
#include "MetaObject/signals/ISlot.hpp"
#include "MetaObject/signals/SignalInfo.hpp"
#include "MetaObject/signals/SlotInfo.hpp"
#include "MetaObject/object/RelayManager.hpp"
#include "MetaObject/object/detail/IMetaObject_pImpl.hpp"
#include "MetaObject/object/detail/IMetaObjectImpl.hpp"
#include "MetaObject/params/IParam.hpp"
#include "MetaObject/params/InputParam.hpp"
#include "MetaObject/params/buffers/BufferFactory.hpp"
#include "MetaObject/params/InputParam.hpp"
#include "MetaObject/params/VariableManager.hpp"

#include "RuntimeObjectSystem/ISimpleSerializer.h"
#include "RuntimeObjectSystem/IObjectState.hpp"

#include <boost/thread/recursive_mutex.hpp>


namespace mo{
int IMetaObject::connect(IMetaObject* sender, const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name) {
    int count = 0;
    auto my_signals = sender->getSignals(signal_name);
    auto my_slots = receiver->getSlots(slot_name);

    for (auto signal : my_signals) {
        for (auto slot : my_slots) {
            if (signal->getSignature() == slot->getSignature()) {
                auto Connection = slot->connect(signal);
                if (Connection) {
                    sender->addConnection(Connection, signal_name, slot_name, slot->getSignature(), receiver);
                    ++count;
                }
                break;
            } else {
                LOG(debug) << "Signature mismatch, Slot (" << slot_name << " -  " <<  slot->getSignature().name()
                           << ") != Signal (" << signal_name << " - " << signal->getSignature().name() << ")";
            }
        }
    }

    return count;
}

bool IMetaObject::connect(IMetaObject* sender, const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name, const TypeInfo& signature) {
    auto signal = sender->getSignal(signal_name, signature);
    if (signal) {
        auto slot = receiver->getSlot(slot_name, signature);
        if (slot) {
            auto Connection = slot->connect(signal);
            sender->addConnection(Connection, signal_name, slot_name, signature, receiver);
            return true;
        }
    }
    return false;
}

IMetaObject::IMetaObject() {
    _mtx = new mo::Mutex_t();
    _pimpl = new impl();
    _sig_manager = nullptr;
    _pimpl->_slot_param_updated = std::bind(&IMetaObject::onParamUpdate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);
}


IMetaObject::~IMetaObject() {
    delete _mtx;
    delete _pimpl;
}

void IMetaObject::Init(bool firstInit) {
    initParams(firstInit);
    initSignals(firstInit);
    bindSlots(firstInit);
    initCustom(firstInit);
    auto params = getParams();
    for(auto param : params) {
        auto update_slot = this->getSlot<void(mo::Context*, mo::IParam*)>("on_" + param->getName() + "_modified");
        if(update_slot) {
            auto Connection = param->registerUpdateNotifier(update_slot);
            this->addConnection(Connection, param->getName() + "_modified", "on_" + param->getName() + "_modified", update_slot->getSignature(), this);
        }
        auto delete_slot = this->getSlot<void(mo::IParam const*)>("on_" + param->getName() + "_deleted");
        if(delete_slot) {
            auto Connection = param->registerDeleteNotifier(delete_slot);
            this->addConnection(Connection, param->getName() + "_deleted", "on_" + param->getName() + "_modified", update_slot->getSignature(), this);
        }
    }

    if(firstInit == false) {
        auto connections_copy = _pimpl->_param_connections;
        _pimpl->_param_connections.clear();
        for (auto& param_connection : connections_copy) {
            rcc::shared_ptr<IMetaObject> obj(param_connection.output_object);
            if (obj) {
                auto output = obj->getOutput(param_connection.output_param);
                if (output == nullptr) {
                    LOG(debug) << "Unable to find " << param_connection.output_param
                               << " in " << obj->GetTypeName() << " reinitializing";
                    obj->initParams(firstInit);
                    output = obj->getOutput(param_connection.output_param);
                    if (output == nullptr) {
                        LOG(info) << "Unable to find " << param_connection.output_param << " in "
                                  << obj->GetTypeName() << " unable to reConnect " << param_connection.input_param
                                  << " from object " << this->GetTypeName();
                        continue;
                    }
                }
                auto input = this->getInput(param_connection.input_param);
                if (input) {
                    if(this->connectInput(input, obj.get(), output, param_connection.connection_type)) {
                        LOG(debug) << "ReConnected " << GetTypeName() << ":" << param_connection.input_param
                                   << " to " << obj->GetTypeName() << ":" << param_connection.output_param;
                    } else {
                        LOG(info) << "ReConnect FAILED " << GetTypeName() << ":" << param_connection.input_param
                                  << " to " << obj->GetTypeName() << ":" << param_connection.output_param;
                    }
                } else {
                    LOG(debug) << "Unable to find input Param "
                               << param_connection.input_param
                               << " in object " << this->GetTypeName();
                }
            } else {
                LOG(debug) << "Output object no longer exists for input ["
                           << param_connection.input_param
                           << "] expected output name ["
                           << param_connection.output_param << "]";
            }
        }
        // Rebuild Connections
        for(auto& connection : _pimpl->_connections) {
            if(!connection.obj.empty()) {
                auto signal = this->getSignal(connection.signal_name, connection.signature);
                auto slot = connection.obj->getSlot(connection.slot_name, connection.signature);
                if(signal == nullptr) {
                    LOG(debug) << "Unable to find signal with name \"" << connection.signal_name
                               << "\" and signature: " << connection.signature.name()
                               << " in new object of type " << this->GetTypeName();
                }
                if(slot == nullptr) {
                    connection.obj->bindSlots(firstInit);
                    slot = connection.obj->getSlot(connection.slot_name, connection.signature);
                    if(slot == nullptr) {
                        LOG(debug) << "Unable to find slot with name \"" << connection.slot_name
                                   << "\" and signature: " << connection.signature.name()
                                   << " in new object of type " << connection.obj->GetTypeName();
                    }
                }
                if(signal && slot) {
                    auto connection_ = slot->connect(signal);
                    if (connection_) {
                        connection.connection = connection_;
                    }
                }
            }
        }
    }
}
void  IMetaObject::initCustom(bool firstInit) {

}

int IMetaObject::setupSignals(RelayManager* manager) {
    _sig_manager = manager;
    int count = 0;
    for(auto& my_slots : _pimpl->_slots) {
        for(auto& slot : my_slots.second) {
            ConnectionInfo info;
            info.connection = manager->connect(slot.second, my_slots.first, this);
            info.slot_name = my_slots.first;
            info.signature = slot.first;
            _pimpl->_connections.push_back(info);
            ++count;
        }
    }

    for(auto& my_signals : _pimpl->_signals) {
        for(auto& signal : my_signals.second) {
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

int IMetaObject::setupVariableManager(IVariableManager* manager) {
    if(_pimpl->_variable_manager != nullptr) {
        removeVariableManager(_pimpl->_variable_manager);
    }
    _pimpl->_variable_manager = manager;
    int count = 0;
    for(auto& param : _pimpl->_implicit_params) {
        manager->addParam(param.second.get());
        ++count;
    }
    for(auto& param : _pimpl->_params) {
        manager->addParam(param.second);
        ++count;
    }
    return count;
}

int IMetaObject::removeVariableManager(IVariableManager* mgr) {
    int count = 0;
    for (auto& param : _pimpl->_implicit_params) {
        mgr->removeParam(param.second.get());
        ++count;
    }
    for (auto& param : _pimpl->_params) {
        mgr->removeParam(param.second);
        ++count;
    }
    return count;
}

void IMetaObject::Serialize(ISimpleSerializer *pSerializer) {
    IObject::Serialize(pSerializer);
    serializeConnections(pSerializer);
    serializeParams(pSerializer);
}

void IMetaObject::serializeConnections(ISimpleSerializer* pSerializer) {
    SERIALIZE(_pimpl->_connections);
    SERIALIZE(_pimpl->_param_connections);
    SERIALIZE(_ctx);
    SERIALIZE(_sig_manager);
}
void IMetaObject::serializeParams(ISimpleSerializer* pSerializer) {

}
void IMetaObject::setContext(const std::shared_ptr<Context>& ctx, bool overwrite) {
    if(_ctx.get() && overwrite == false)
        return;
    if(ctx == nullptr)
        LOG(info) << "Setting context to nullptr";
    _ctx = ctx;
    for(auto& param : _pimpl->_implicit_params) {
        param.second->setContext(ctx.get());
    }
    for(auto& param : _pimpl->_params) {
        param.second->setContext(ctx.get());
    }
}

int IMetaObject::disConnectByName(const std::string& name) {
    auto my_signals = this->getSignals(name);
    int count = 0;
    for(auto& sig : my_signals) {
        count += sig->disConnect() ? 1 : 0;
    }
    return count;
}

bool IMetaObject::disConnect(ISignal* sig) {
    return false;
}

int IMetaObject::disConnect(IMetaObject* obj) {
    auto obj_signals = obj->getSignals();
    int count = 0;
    for(auto signal : obj_signals) {
        count += disConnect(signal.first) ? 1 : 0;
    }
    return count;
}


std::vector<IParam*> IMetaObject::getDisplayParams() const {
    std::vector<IParam*> output;
    for(auto& param : _pimpl->_params) {
        output.push_back(param.second);
    }
    for(auto& param : _pimpl->_implicit_params) {
        output.push_back(param.second.get());
    }
    return output;
}

IParam* IMetaObject::getParam(const std::string& name) const {
    auto itr = _pimpl->_params.find(name);
    if(itr != _pimpl->_params.end()) {
        return itr->second;
    }
    auto itr2 = _pimpl->_implicit_params.find(name);
    if(itr2 != _pimpl->_implicit_params.end()) {
        return itr2->second.get();
    }
    THROW(debug) << "Param with name \"" << name << "\" not found";
    return nullptr;
}
std::vector<IParam*> IMetaObject::getParams(const std::string& filter) const {
    std::vector<IParam*> output;
    for(auto& itr : _pimpl->_params) {
        if(filter.size()) {
            if(itr.first.find(filter) != std::string::npos)
                output.push_back(itr.second);
        } else {
            output.push_back(itr.second);
        }
    }
    for(auto& itr : _pimpl->_implicit_params) {
        if (filter.size()) {
            if (itr.first.find(filter) != std::string::npos)
                output.push_back(itr.second.get());
        } else {
            output.push_back(itr.second.get());
        }
    }
    return output;
}

std::vector<IParam*> IMetaObject::getParams(const TypeInfo& filter) const {
    std::vector<IParam*> output;
    for (auto& itr : _pimpl->_params) {
        if(itr.second->getTypeInfo() == filter)
            output.push_back(itr.second);

    }
    for (auto& itr : _pimpl->_implicit_params) {
        if(itr.second->getTypeInfo() == filter)
            output.push_back(itr.second.get());
    }
    return output;
}

IParam* IMetaObject::getParamOptional(const std::string& name) const {
    auto itr = _pimpl->_params.find(name);
    if(itr != _pimpl->_params.end()) {
        return itr->second;
    }
    auto itr2 = _pimpl->_implicit_params.find(name);
    if(itr2 != _pimpl->_implicit_params.end()) {
        return itr2->second.get();
    }
    LOG(trace) << "Param with name \"" << name << "\" not found";
    return nullptr;
}

InputParam* IMetaObject::getInput(const std::string& name) const {
    auto itr = _pimpl->_input_Params.find(name);
    if(itr != _pimpl->_input_Params.end()) {
        return itr->second;
    }
    return nullptr;
}

std::shared_ptr<Context> IMetaObject::getContext() const {
    return _ctx;
}

std::vector<InputParam*> IMetaObject::getInputs(const std::string& name_filter) const {
    std::vector<InputParam*> output;
    for(auto param : _pimpl->_input_Params) {
        if(name_filter.size()) {
            if(param.second->getName().find(name_filter) != std::string::npos)
                output.push_back(param.second);
        } else {
            output.push_back(param.second);
        }

    }
    return output;
}

std::vector<InputParam*> IMetaObject::getInputs(const TypeInfo& type_filter, const std::string& name_filter) const {
    std::vector<InputParam*> output;
    for(auto param : _pimpl->_params) {
        if(param.second->checkFlags(Input_e)) {
            if(param.second->getTypeInfo() == type_filter) {
                if(name_filter.size()) {
                    if(name_filter.find(param.first) != std::string::npos)
                        if(auto out = dynamic_cast<InputParam*>(param.second))
                            output.push_back(out);
                } else {
                    if(auto out = dynamic_cast<InputParam*>(param.second))
                        output.push_back(out);
                }
            }
        }
    }
    for(auto param : _pimpl->_implicit_params) {
        if(param.second->checkFlags(Input_e)) {
            if(param.second->getTypeInfo() == type_filter) {
                if(name_filter.size()) {
                    if(name_filter.find(param.first) != std::string::npos)
                        if(auto out = dynamic_cast<InputParam*>(param.second.get()))
                            output.push_back(out);
                } else {
                    if(auto out = dynamic_cast<InputParam*>(param.second.get()))
                        output.push_back(out);
                }
            }
        }
    }
    return output;
}

IParam* IMetaObject::getOutput(const std::string& name) const {
    auto itr = _pimpl->_params.find(name);
    if(itr != _pimpl->_params.end()) {
        return itr->second;
    }
    auto itr2 = _pimpl->_implicit_params.find(name);
    if(itr2 != _pimpl->_implicit_params.end()) {
        return itr2->second.get();
    }
    return nullptr;
}
std::vector<IParam*> IMetaObject::getOutputs(const std::string& name_filter) const {
    std::vector<IParam*> output;
    for(auto param : _pimpl->_params) {
        if(param.second->checkFlags(Output_e)) {
            if(name_filter.size()) {
                if(param.first.find(name_filter) != std::string::npos) {
                    output.push_back(param.second);
                }
            } else {
                output.push_back(param.second);
            }
        }
    }
    for(auto param : _pimpl->_implicit_params) {
        if(param.second->checkFlags(Output_e)) {
            if(name_filter.size()) {
                if(param.first.find(name_filter) != std::string::npos) {
                    output.push_back(param.second.get());
                }
            } else {
                output.push_back(param.second.get());
            }
        }
    }
    return output;
}

std::vector<IParam*> IMetaObject::getOutputs(const TypeInfo& type_filter, const std::string& name_filter) const {
    std::vector<IParam*> output;
    for(auto param : _pimpl->_params) {
        if(param.second->checkFlags(Output_e)) {
            if(name_filter.size()) {
                if(name_filter.find(param.first) != std::string::npos) {
                    if(param.second->getTypeInfo() == type_filter)
                        output.push_back(param.second);
                }
            } else {
                if(param.second->getTypeInfo() == type_filter)
                    output.push_back(param.second);
            }
        }
    }
    for(auto param : _pimpl->_implicit_params) {
        if(param.second->checkFlags(Output_e)) {
            if(name_filter.size()) {
                if(name_filter.find(param.first) != std::string::npos) {
                    if(param.second->getTypeInfo() == type_filter)
                        output.push_back(param.second.get());
                }
            } else {
                if(param.second->getTypeInfo() == type_filter)
                    output.push_back(param.second.get());
            }
        }
    }
    return output;
}
bool IMetaObject::connectInput(const std::string& input_name,
                               IMetaObject* output_object,
                               IParam* output,
                               ParamType type_) {
    auto input = getInput(input_name);
    if(input && output)
        return connectInput(input, output_object, output, type_);

    auto inputs = getInputs();
    auto print_inputs = [inputs]()->std::string {
        std::stringstream ss;
        for(auto _input : inputs) {
            ss << dynamic_cast<IParam*>(_input)->getName() << ", ";
        }
        return ss.str();
    };
    LOG(debug) << "Unable to find input by name "
               << input_name << " in object "
               << this->GetTypeName()
               << " with inputs [" << print_inputs() << "]";
    return false;
}

bool IMetaObject::connectInput(InputParam* input,
                               IMetaObject* output_object,
                               IParam* output,
                               ParamType type_) {
    if(input == nullptr || output == nullptr) {
        LOG(debug) << "NULL input or output passed in";
        return false;
    }

    if(input && input->acceptsInput(output)) {
        // Check contexts to see if a buffer needs to be setup
        auto output_ctx = output->getContext();
        if(type_ & ForceBufferedConnection_e || input->checkFlags(mo::RequestBuffered_e) || output->checkFlags(mo::RequestBuffered_e)) {
            type_ = ParamType(type_ & ~ForceBufferedConnection_e);
            auto buffer = Buffer::BufferFactory::CreateProxy(output, type_);
            if(!buffer) {
                LOG(warning) << "Unable to create " << paramTypeToString(type_)
                             << " for datatype " << Demangle::TypeToName(output->getTypeInfo());
                return false;
            }
            std::string buffer_type = paramTypeToString(type_);
            buffer->setName(output->getTreeName() + " " + buffer_type + " buffer for " + input->getTreeName());
            if(input->setInput(buffer)) {
                _pimpl->_param_connections.emplace_back(output_object, output->getName(), input->getName(), type_);
                return true;
            } else {
                LOG(debug) << "Failed to connect output " << output->getName()
                           << "[" << Demangle::TypeToName(output->getTypeInfo())<< "] to input "
                           << dynamic_cast<IParam*>(input)->getName()
                           << "[" << Demangle::TypeToName(dynamic_cast<IParam*>(input)->getTypeInfo()) << "]";
                return false;
            }
        }
        if(output_ctx && _ctx.get()) {
            if(output_ctx->thread_id != _ctx.get()->thread_id) {
                auto buffer = Buffer::BufferFactory::CreateProxy(output, type_);
                if(buffer) {
                    buffer->setName(output->getTreeName() + " buffer for " + input->getTreeName());
                    if(input->setInput(buffer)) {
                        _pimpl->_param_connections.emplace_back(output_object, output->getName(), input->getName(), type_);
                        return true;
                    } else {
                        LOG(debug) << "Failed to connect output " << output->getName()
                                   << "[" << Demangle::TypeToName(output->getTypeInfo()) << "] to input "
                                   << dynamic_cast<IParam*>(input)->getName() << "["
                                   << Demangle::TypeToName(dynamic_cast<IParam*>(input)->getTypeInfo()) << "]";
                        return false;
                    }
                } else {
                    LOG(debug) << "No buffer of desired type found for type " << Demangle::TypeToName(output->getTypeInfo());
                }
            } else {
                if(input->setInput(output)) {
                    _pimpl->_param_connections.emplace_back(output_object, output->getName(), input->getName(), type_);
                    return true;
                } else {
                    LOG(debug) << "Failed to connect output " << output->getName()
                               << "[" << Demangle::TypeToName(output->getTypeInfo()) << "] to input "
                               << dynamic_cast<IParam*>(input)->getName()
                               << "[" << Demangle::TypeToName(dynamic_cast<IParam*>(input)->getTypeInfo()) << "]";
                    return false;
                }
            }
        } else {
            if(input->setInput(output)) {
                _pimpl->_param_connections.emplace_back(output_object, output->getName(), input->getName(), type_);
                return true;
            } else {
                LOG(debug) << "Failed to connect output " << output->getName()
                           << "[" << Demangle::TypeToName(output->getTypeInfo()) << "] to input "
                           << dynamic_cast<IParam*>(input)->getName()
                           << "[" << Demangle::TypeToName(dynamic_cast<IParam*>(input)->getTypeInfo()) << "]";
                return false;
            }
        }
    }
    LOG(debug) << "Input \"" << input->getTreeName()
               << "\"  does not accept input of type: "
               << Demangle::TypeToName(output->getTypeInfo());
    return false;
}

bool IMetaObject::connectInput(IMetaObject* out_obj, IParam* out_param,
                               IMetaObject* in_obj, InputParam* in_param,
                               ParamType type) {
    return in_obj->connectInput(in_param, out_obj, out_param, type);
}
IParam* IMetaObject::addParam(std::shared_ptr<IParam> param) {
    param->setMtx(_mtx);
    param->setContext(_ctx.get());
#ifdef _DEBUG
    for(auto& param_ : _pimpl->_params) {
        if(param_.second == param.get()) {
            LOG(debug) << "Trying to add a Param a second time";
            return param.get();
        }
    }
#endif
    _pimpl->_implicit_params[param->getName()] = param;
    if(param->checkFlags(Input_e)) {
        _pimpl->_input_Params[param->getName()] = dynamic_cast<InputParam*>(param.get());
    }
    param->registerUpdateNotifier(&(this->_pimpl->_slot_param_updated));
    _pimpl->_sig_param_added(this, param.get());
    return param.get();
}

IParam* IMetaObject::addParam(IParam* param) {
    param->setMtx(_mtx);
    param->setContext(_ctx.get());
#ifdef _DEBUG
    for(auto& param_ : _pimpl->_params) {
        if(param_.second == param) {
            LOG(debug) << "Trying to add a Param a second time";
            return param;
        }
    }
#endif
    _pimpl->_params[param->getName()] = param;
    if(param->checkFlags(Input_e)) {
        _pimpl->_input_Params[param->getName()] = dynamic_cast<InputParam*>(param);
    }
    auto Connection = param->registerUpdateNotifier(&(this->_pimpl->_slot_param_updated));
    _pimpl->_sig_param_added(this, param);
    this->addConnection(Connection, "Param_update", "Param_updated",
                        this->_pimpl->_slot_param_updated.getSignature(), this);
    return param;
}

void IMetaObject::setParamRoot(const std::string& root) {
    for(auto& param : _pimpl->_params) {
        param.second->setTreeRoot(root);
    }
    for(auto& param : _pimpl->_implicit_params) {
        param.second->setTreeRoot(root);
    }
}

std::vector<ParamInfo*> IMetaObject::getParamInfo(const std::string& name) const {
    std::vector<ParamInfo*> output;
    getParamInfo(output);

    return output;
}

std::vector<ParamInfo*> IMetaObject::getParamInfo() const {
    std::vector<ParamInfo*> output;
    getParamInfo(output);

    return output;
}

std::vector<SignalInfo*>    IMetaObject::getSignalInfo(const std::string& name) const {
    std::vector<SignalInfo*> info;
    getSignalInfo(info);

    return info;
}
std::vector<SignalInfo*>    IMetaObject::getSignalInfo() const {
    std::vector<SignalInfo*> info;
    getSignalInfo(info);
    return info;
}
std::vector<SlotInfo*> IMetaObject::getSlotInfo() const {
    std::vector<SlotInfo*> output;
    getSlotInfo(output);
    return output;
}
std::vector<SlotInfo*> IMetaObject::getSlotInfo(const std::string& name) const {
    std::vector<SlotInfo*> tmp;
    getSlotInfo(tmp);
    std::vector<SlotInfo*> output;
    for (auto& itr : tmp) {
        if (itr->name.find(name) != std::string::npos)
            output.push_back(itr);
    }

    return output;
}

std::vector<std::pair<ISlot*, std::string>>  IMetaObject::getSlots() const {
    std::vector<std::pair<ISlot*, std::string>>  my_slots;
    for(auto itr1 : _pimpl->_slots) {
        for(auto itr2: itr1.second) {
            my_slots.push_back(std::make_pair(itr2.second, itr1.first));
        }
    }
    return my_slots;
}

std::vector<ISlot*> IMetaObject::getSlots(const std::string& name) const {
    std::vector<ISlot*> output;
    auto itr = _pimpl->_slots.find(name);
    if(itr != _pimpl->_slots.end()) {
        for(auto slot : itr->second) {
            output.push_back(slot.second);
        }
    }
    return output;
}

std::vector<std::pair<ISlot*, std::string>> IMetaObject::getSlots(const TypeInfo& signature) const {
    std::vector<std::pair<ISlot*, std::string>> output;
    for(auto& type : _pimpl->_slots) {
        auto itr = type.second.find(signature);
        if(itr != type.second.end()) {
            output.push_back(std::make_pair(itr->second, type.first));
        }
    }
    return output;
}

ISlot* IMetaObject::getSlot(const std::string& name, const TypeInfo& signature) const {
    auto itr1 = _pimpl->_slots.find(name);
    if(itr1 != _pimpl->_slots.end()) {
        auto itr2 = itr1->second.find(signature);
        if(itr2 != itr1->second.end()) {
            return itr2->second;
        }
    }
    if(name == "Param_updated") {
        return &(_pimpl->_slot_param_updated);
    }
    return nullptr;
}

bool IMetaObject::ConnectByName(const std::string& name, ISlot* slot) {
    auto signal = getSignal(name, slot->getSignature());
    if (signal) {
        auto Connection = signal->connect(slot);
        if (Connection) {
            addConnection(Connection, name, "", slot->getSignature());
            return true;
        }
    }
    return false;
}
bool IMetaObject::ConnectByName(const std::string& name, ISignal* signal) {
    auto slot = getSlot(name, signal->getSignature());
    if (slot) {
        auto Connection = slot->connect(signal);
        if (Connection) {
            addConnection(Connection, "", name, signal->getSignature());
            return true;
        }
    }
    return false;
}

int IMetaObject::ConnectByName(const std::string& name, RelayManager* mgr) {

    return 0;
}

int  IMetaObject::ConnectByName(const std::string& signal_name,
                                IMetaObject* receiver,
                                const std::string& slot_name) {
    int count = 0;
    auto my_signals = getSignals(signal_name);
    auto my_slots = receiver->getSlots(slot_name);
    for (auto signal : my_signals) {
        for (auto slot : my_slots) {
            if (signal->getSignature() == slot->getSignature()) {
                auto Connection = slot->connect(signal);
                if (Connection) {
                    addConnection(Connection, signal_name,
                                  slot_name, slot->getSignature(), receiver);
                    ++count;
                    break;
                }
            }
        }
    }
    return count;
}

bool IMetaObject::ConnectByName(const std::string& signal_name,
                                IMetaObject* receiver,
                                const std::string& slot_name,
                                const TypeInfo& signature) {
    auto signal = getSignal(signal_name, signature);
    auto slot = receiver->getSlot(slot_name, signature);
    if (signal && slot) {
        auto Connection = slot->connect(signal);
        if (Connection) {
            addConnection(Connection, signal_name, slot_name, signature, receiver);
            return true;
        }
    }
    return false;
}

int IMetaObject::ConnectAll(RelayManager* mgr) {
    auto my_signals = getSignalInfo();
    int count = 0;
    for(auto& signal : my_signals) {
        count += ConnectByName(signal->name, mgr);
    }
    return count;
}

void IMetaObject::addSlot(ISlot* slot, const std::string& name) {
    _pimpl->_slots[name][slot->getSignature()] = slot;
    slot->setParent(this);
}

void IMetaObject::addSignal(ISignal* sig, const std::string& name) {
    _pimpl->_signals[name][sig->getSignature()] = sig;
    sig->setParent(this);
}

std::vector<std::pair<ISignal*, std::string>> IMetaObject::getSignals() const {
    std::vector<std::pair<ISignal*, std::string>> my_signals;
    for(auto& name_itr : _pimpl->_signals) {
        for(auto& sig_itr : name_itr.second) {
            my_signals.push_back(std::make_pair(sig_itr.second, name_itr.first));
        }
    }
    return my_signals;
}

std::vector<ISignal*> IMetaObject::getSignals(const std::string& name) const {
    std::vector<ISignal*> my_signals;
    auto itr = _pimpl->_signals.find(name);
    if(itr != _pimpl->_signals.end()) {
        for(auto& sig_itr : itr->second) {
            my_signals.push_back(sig_itr.second);
        }
    }
    return my_signals;
}

std::vector<std::pair<ISignal*, std::string>> IMetaObject::getSignals(const TypeInfo& type) const {
    std::vector<std::pair<ISignal*, std::string>> my_signals;
    for(auto& name_itr : _pimpl->_signals) {
        auto type_itr = name_itr.second.find(type);
        if(type_itr != name_itr.second.end()) {
            my_signals.push_back(std::make_pair(type_itr->second, name_itr.first));
        }
    }
    return my_signals;
}

ISignal* IMetaObject::getSignal(const std::string& name, const TypeInfo& type) const {
    auto name_itr = _pimpl->_signals.find(name);
    if (name_itr != _pimpl->_signals.end()) {
        auto type_itr = name_itr->second.find(type);
        if (type_itr != name_itr->second.end()) {
            return type_itr->second;
        }
    }
    if(name == "Param_updated") {
        return &(_pimpl->_sig_param_updated);
    }
    if(name == "Param_added") {
        return &(_pimpl->_sig_param_added);
    }
    return nullptr;
}

void IMetaObject::addConnection(std::shared_ptr<Connection>& Connection,
                                const std::string& signal_name,
                                const std::string& slot_name,
                                const TypeInfo& signature,
                                IMetaObject* obj) {
    ConnectionInfo info;
    info.connection = Connection;
    info.obj = rcc::weak_ptr<IMetaObject>(obj);
    info.signal_name = signal_name;
    info.slot_name = slot_name;
    info.signature = signature;
    _pimpl->_connections.push_back(info);
}
void IMetaObject::onParamUpdate(IParam* param, Context*, OptionalTime_t, size_t, ICoordinateSystem*, UpdateFlags) {
    this->_pimpl->_sig_param_updated(this, param);
}
} // namespace mo
