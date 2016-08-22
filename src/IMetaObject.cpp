#include "MetaObject/IMetaObject.hpp"

#include "MetaObject/Logging/Log.hpp"
#include "MetaObject/Parameters/Demangle.hpp"
#include "MetaObject/Signals/ISignal.hpp"
#include "MetaObject/Signals/ISlot.hpp"
#include "MetaObject/Signals/SignalInfo.hpp"
#include "MetaObject/Signals/SlotInfo.hpp"

#include "MetaObject/Detail/IMetaObject_pImpl.hpp"

#include "MetaObject/Parameters/IParameter.hpp"
#include "MetaObject/Parameters/InputParameter.hpp"
#include "MetaObject/Parameters/Buffers/BufferFactory.hpp"

#include "ISimpleSerializer.h"
#include "IObjectState.hpp"

using namespace mo;
int IMetaObject::Connect(IMetaObject* sender, const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name)
{
	int count = 0;
	auto signals = sender->GetSignals(signal_name);
	auto slots = receiver->GetSlots(slot_name);
	
	for (auto signal : signals)
	{
		for (auto slot : slots)
		{
			if (signal->GetSignature() == slot->GetSignature())
			{
				auto connection = slot->Connect(signal);
				if (connection)
				{
					sender->AddConnection(connection, signal_name, slot_name, slot->GetSignature(), receiver);
					++count;
				}
				break;
			}
		}
	}
	
	return count;
}

bool IMetaObject::Connect(IMetaObject* sender, const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name, const TypeInfo& signature)
{
	auto signal = sender->GetSignal(signal_name, signature);
	if (signal)
	{
		auto slot = receiver->GetSlot(slot_name, signature);
		if (slot)
		{
			auto connection = slot->Connect(signal);
			sender->AddConnection(connection, signal_name, slot_name, signature, receiver);
			return true;
		}
	}
	return false;
}

IMetaObject::IMetaObject()
{
    _pimpl = new impl();
    _ctx = nullptr;
    _sig_manager = nullptr;
}

IMetaObject::~IMetaObject()
{
    delete _pimpl;
}

void IMetaObject::Init(bool firstInit)
{
    InitParameters(firstInit);
	InitSignals(firstInit);
    BindSlots(firstInit);
    if(firstInit == false)
    {
        // Rebuild connections
        for(auto& connection : _pimpl->_connections)
        {
            if(!connection.obj.empty())
            {
                auto signal = this->GetSignal(connection.signal_name, connection.signature);
                auto slot = connection.obj->GetSlot(connection.slot_name, connection.signature);
                if(signal == nullptr)
                {
                    LOG(debug) << "Unable to find signal with name \"" << connection.signal_name << "\" and signature: " << connection.signature.name() << " in new object of type " << this->GetTypeName();
                }
                if(slot == nullptr)
                {
                    connection.obj->BindSlots(firstInit);
                    slot = connection.obj->GetSlot(connection.slot_name, connection.signature);
                    if(slot == nullptr)
                    {
                        LOG(debug) << "Unable to find slot with name \"" << connection.slot_name << "\" and signature: " << connection.signature.name() << " in new object of type " << connection.obj->GetTypeName();
                    }
                }
                if(signal && slot)
                {
                    auto connection_ = slot->Connect(signal);
                    if (connection_)
                    {
                        connection.connection = connection_;
                    }
                }
            }
        }
    }
}

int IMetaObject::SetupSignals(RelayManager* manager)
{
    return 0;
}

int IMetaObject::SetupVariableManager(IVariableManager* manager)
{
    return 0;
}

void IMetaObject::Serialize(ISimpleSerializer *pSerializer)
{
    SerializeConnections(pSerializer);
    SerializeParameters(pSerializer);
}

void IMetaObject::SerializeConnections(ISimpleSerializer* pSerializer)
{
    SERIALIZE(_pimpl->_connections);
}
void IMetaObject::SerializeParameters(ISimpleSerializer* pSerializer)
{

}
void IMetaObject::SetContext(Context* ctx)
{
    _ctx = ctx;
    for(auto& param : _pimpl->_implicit_parameters)
    {
        param.second->SetContext(ctx);
    }
    for(auto& param : _pimpl->_parameters)
    {
        param.second->SetContext(ctx);
    }
}

int IMetaObject::DisconnectByName(const std::string& name)
{
    auto signals = this->GetSignals(name);
    int count = 0;
    for(auto& sig : signals)
    {
		count += sig->Disconnect() ? 1 : 0;
    }
    return count;
}

bool IMetaObject::Disconnect(ISignal* sig)
{
    return false;
}

int IMetaObject::Disconnect(IMetaObject* obj)
{
    auto signals = obj->GetSignals();
    int count = 0;
    for(auto signal : signals)
    {
        count += Disconnect(signal.first) ? 1 : 0;
    }
    return count;
}


std::vector<IParameter*> IMetaObject::GetDisplayParameters() const
{
    std::vector<IParameter*> output;
    for(auto& param : _pimpl->_parameters)
    {
        output.push_back(param.second);
    }
    for(auto& param : _pimpl->_implicit_parameters)
    {
        output.push_back(param.second.get());
    }
    return output;
}

IParameter* IMetaObject::GetParameter(const std::string& name) const
{
    auto itr = _pimpl->_parameters.find(name);
    if(itr != _pimpl->_parameters.end())
    {
        return itr->second;
    }
    THROW(debug) << "Parameter with name \"" << name << "\" not found";
    return nullptr;
}
std::vector<IParameter*> IMetaObject::GetParameters(const std::string& filter) const
{
    std::vector<IParameter*> output;
    for(auto& itr : _pimpl->_parameters)
    {
        if(filter.size())
        {
            if(itr.first.find(filter) != std::string::npos)
                output.push_back(itr.second);
        }else
        {
            output.push_back(itr.second);
        }
    }
    return output;
}

IParameter* IMetaObject::GetParameterOptional(const std::string& name) const
{
    auto itr = _pimpl->_parameters.find(name);
    if(itr != _pimpl->_parameters.end())
    {
        return itr->second;
    }
    LOG(trace) << "Parameter with name \"" << name << "\" not found";
    return nullptr;
}

InputParameter* IMetaObject::GetInput(const std::string& name) const
{
    auto itr = _pimpl->_implicit_parameters.find(name);
    if(itr != _pimpl->_implicit_parameters.end())
    {
        if(itr->second->CheckFlags(Input_e))
        {
            return dynamic_cast<InputParameter*>(itr->second.get());
        }
    }
    auto itr2 = _pimpl->_parameters.find(name);
    if(itr2 != _pimpl->_parameters.end())
    {
        if(itr2->second->CheckFlags(Input_e))
        {
            return dynamic_cast<InputParameter*>(itr2->second);
        }
    }
    return nullptr;
}

std::vector<InputParameter*> IMetaObject::GetInputs(const std::string& name_filter) const
{
    std::vector<InputParameter*> output;
    for(auto param : _pimpl->_parameters)
    {
        if(param.second->CheckFlags(Input_e))
        {
            if(name_filter.size())
            {
                if(param.second->GetName().find(name_filter) != std::string::npos)
                    if(auto out = dynamic_cast<InputParameter*>(param.second))
                        output.push_back(out);
            }else
            {
                if(auto out = dynamic_cast<InputParameter*>(param.second))
                    output.push_back(out);
            }
        }
    }
    for(auto param : _pimpl->_implicit_parameters)
    {
        if(param.second->CheckFlags(Input_e))
        {
            if(name_filter.size())
            {
                if(param.second->GetName().find(name_filter) != std::string::npos)
                    if(auto out = dynamic_cast<InputParameter*>(param.second.get()))
                        output.push_back(out);
            }else
            {
                if(auto out = dynamic_cast<InputParameter*>(param.second.get()))
                    output.push_back(out);
            }
        }
    }
    return output;
}

std::vector<InputParameter*> IMetaObject::GetInputs(const TypeInfo& type_filter, const std::string& name_filter) const
{
    std::vector<InputParameter*> output;
    for(auto param : _pimpl->_parameters)
    {
        if(param.second->CheckFlags(Input_e))
        {
            if(param.second->GetTypeInfo() == type_filter)
            {
                if(name_filter.size())
                {
                    if(name_filter.find(param.first) != std::string::npos)
                        if(auto out = dynamic_cast<InputParameter*>(param.second))
                            output.push_back(out);
                }else
                {
                    if(auto out = dynamic_cast<InputParameter*>(param.second))
                        output.push_back(out);
                }
            }
        }
    }
    for(auto param : _pimpl->_implicit_parameters)
    {
        if(param.second->CheckFlags(Input_e))
        {
            if(param.second->GetTypeInfo() == type_filter)
            {
                if(name_filter.size())
                {
                    if(name_filter.find(param.first) != std::string::npos)
                        if(auto out = dynamic_cast<InputParameter*>(param.second.get()))
                            output.push_back(out);
                }else
                {
                    if(auto out = dynamic_cast<InputParameter*>(param.second.get()))
                        output.push_back(out);
                }
            }
        }
    }
    return output;
}

IParameter* IMetaObject::GetOutput(const std::string& name) const
{
    auto itr = _pimpl->_parameters.find(name);
    if(itr != _pimpl->_parameters.end())
    {
        return itr->second;
    }
    auto itr2 = _pimpl->_implicit_parameters.find(name);
    if(itr2 != _pimpl->_implicit_parameters.end())
    {
        return itr2->second.get();
    }
    return nullptr;
}
std::vector<IParameter*> IMetaObject::GetOutputs(const std::string& name_filter) const
{
    std::vector<IParameter*> output;
    for(auto param : _pimpl->_parameters)
    {
        if(param.second->CheckFlags(Output_e))
        {
            if(name_filter.size())
            {
                if(param.first.find(name_filter) != std::string::npos)
                {
                    output.push_back(param.second);
                }
            }else
            {
                output.push_back(param.second);
            }
        }
    }
    for(auto param : _pimpl->_implicit_parameters)
    {
        if(param.second->CheckFlags(Output_e))
        {
            if(name_filter.size())
            {
                if(param.first.find(name_filter) != std::string::npos)
                {
                    output.push_back(param.second.get());
                }
            }else
            {
                output.push_back(param.second.get());
            }
        }
    }
    return output;
}

std::vector<IParameter*> IMetaObject::GetOutputs(const TypeInfo& type_filter, const std::string& name_filter) const
{
    std::vector<IParameter*> output;
    for(auto param : _pimpl->_parameters)
    {
        if(param.second->CheckFlags(Output_e))
        {
            if(name_filter.size())
            {
                if(name_filter.find(param.first) != std::string::npos)
                {
                    if(param.second->GetTypeInfo() == type_filter)
                        output.push_back(param.second);
                }
            }else
            {
                if(param.second->GetTypeInfo() == type_filter)
                    output.push_back(param.second);
            }
        }
    }
    for(auto param : _pimpl->_implicit_parameters)
    {
        if(param.second->CheckFlags(Output_e))
        {
            if(name_filter.size())
            {
                if(name_filter.find(param.first) != std::string::npos)
                {
                    if(param.second->GetTypeInfo() == type_filter)
                        output.push_back(param.second.get());
                }
            }else
            {
                if(param.second->GetTypeInfo() == type_filter)
                    output.push_back(param.second.get());
            }
        }
    }
    return output;
}
bool IMetaObject::ConnectInput(const std::string& input_name, IParameter* output, ParameterTypeFlags type_)
{
    auto input = GetInput(input_name);
    if(input && input->AcceptsInput(output))
    {
        // Check contexts to see if a buffer needs to be setup
        auto output_ctx = output->GetContext();
        if(output_ctx && _ctx)
        {
            if(output_ctx->thread_id != _ctx->thread_id)
            {
                auto buffer = Buffer::BufferFactory::CreateProxy(output, type_);
                if(buffer)
                {
                    bool ret = input->SetInput(buffer);
                    if(ret == false)
                    {
                        LOG(debug) << "Failed to connect output " << output->GetName() << "[" << Demangle::TypeToName(output->GetTypeInfo()) 
                            << "] to input " << input_name << "[" << Demangle::TypeToName(dynamic_cast<IParameter*>(input)->GetTypeInfo()) << "]";
                    }
                    return ret;
                }else
                {
                    LOG(debug) << "No buffer of desired type found for type " << Demangle::TypeToName(output->GetTypeInfo());
                }
            }
        }else
        {
            bool ret = input->SetInput(output);
            if(ret == false)
            {
                LOG(debug) << "Failed to connect output " << output->GetName() << "[" << Demangle::TypeToName(output->GetTypeInfo()) 
                    << "] to input " << input_name << "[" << Demangle::TypeToName(dynamic_cast<IParameter*>(input)->GetTypeInfo()) << "]";
            }
            return ret;
        }
    }
    auto inputs = GetInputs();
    auto print_inputs = [inputs]()->std::string
    {
        std::stringstream ss;
        for(auto _input : inputs)
        {
            ss << dynamic_cast<IParameter*>(_input)->GetName() << ", ";
        }
        return ss.str();
    };
    LOG(debug) << "Unable to find input by name " << input_name << " in object " << this->GetTypeName() << " with inputs [" << print_inputs() << "]";
    return false;
}
IParameter* IMetaObject::AddParameter(std::shared_ptr<IParameter> param)
{
    param->SetMtx(&_mtx);
    _pimpl->_implicit_parameters[param->GetName()] = param;
    std::shared_ptr<TypedSlot<void(Context*, IParameter*)>> update_slot(
        new TypedSlot<void(Context*, IParameter*)>(
            [this](Context* ctx, IParameter* param)
            {
                this->_pimpl->_sig_parameter_updated(this, param);
            }));

    param->RegisterUpdateNotifier(update_slot.get());
    _pimpl->_parameter_update_slots[param->GetName()] = update_slot;
    _pimpl->_sig_parameter_added(this, param.get());
    return param.get();
}

IParameter* IMetaObject::AddParameter(IParameter* param)
{
    param->SetMtx(&_mtx);
    _pimpl->_parameters[param->GetName()] = param;
    std::shared_ptr < TypedSlot<void(Context*, IParameter*)>> update_slot(
        new TypedSlot<void(Context*, IParameter*)>(
            [this](Context* ctx, IParameter* param)
    {
        this->_pimpl->_sig_parameter_updated(this, param);
    }));
    param->RegisterUpdateNotifier(update_slot.get());
    _pimpl->_parameter_update_slots[param->GetName()] = update_slot;
    _pimpl->_sig_parameter_added(this, param);
    return param;
}

void IMetaObject::SetParameterRoot(const std::string& root)
{
    for(auto& param : _pimpl->_parameters)
    {
        param.second->SetTreeRoot(root);
    }
    for(auto& param : _pimpl->_implicit_parameters)
    {
        param.second->SetTreeRoot(root);
    }
}

std::vector<ParameterInfo*> IMetaObject::GetParameterInfo(const std::string& name) const
{
    auto output = GetParameterInfo();

    return output;
}
std::vector<SignalInfo*>    IMetaObject::GetSignalInfo(const std::string& name) const
{
    auto output = GetSignalInfo();

    return output;
}
std::vector<SlotInfo*> IMetaObject::GetSlotInfo(const std::string& name) const
{
    auto tmp = GetSlotInfo();
    std::vector<SlotInfo*> output;
    for(auto& itr : tmp)
    {
        if(itr->name == name)
            output.push_back(itr);
    }
    
    return output;
}


std::vector<std::pair<ISlot*, std::string>>  IMetaObject::GetSlots() const
{
	std::vector<std::pair<ISlot*, std::string>>  slots;
    for(auto itr1 : _pimpl->_slots)
    {
        for(auto itr2: itr1.second)
        {
            slots.push_back(std::make_pair(itr2.second, itr1.first));
        }
    }
    return slots;
}

std::vector<ISlot*> IMetaObject::GetSlots(const std::string& name) const
{
    std::vector<ISlot*> output;
    auto itr = _pimpl->_slots.find(name);
    if(itr != _pimpl->_slots.end())
    {
        for(auto slot : itr->second)
        {
            output.push_back(slot.second);
        }
    }
    return output;
}

std::vector<std::pair<ISlot*, std::string>> IMetaObject::GetSlots(const TypeInfo& signature) const
{
	std::vector<std::pair<ISlot*, std::string>> output;
    for(auto& type : _pimpl->_slots)
    {
        auto itr = type.second.find(signature);
        if(itr != type.second.end())
        {
            output.push_back(std::make_pair(itr->second, type.first));
        }
    }
    return output;
}

ISlot* IMetaObject::GetSlot(const std::string& name, const TypeInfo& signature) const
{
    auto itr1 = _pimpl->_slots.find(name);
    if(itr1 != _pimpl->_slots.end())
    {
        auto itr2 = itr1->second.find(signature);
        if(itr2 != itr1->second.end())
        {
            return itr2->second;
        }
    }
    return nullptr;
}

bool IMetaObject::ConnectByName(const std::string& name, ISlot* slot)
{
	auto signal = GetSignal(name, slot->GetSignature());
	if (signal)
	{
		auto connection = signal->Connect(slot);
		if (connection)
		{
			AddConnection(connection, name, "", slot->GetSignature());
			return true;
		}
	}
	return false;
}
bool IMetaObject::ConnectByName(const std::string& name, ISignal* signal)
{
	auto slot = GetSlot(name, signal->GetSignature());
	if (slot)
	{
		auto connection = slot->Connect(signal);
		if (connection)
		{
			AddConnection(connection, "", name, signal->GetSignature());
			return true;
		}
	}
	return false;
}

int IMetaObject::ConnectByName(const std::string& name, RelayManager* mgr)
{

	return 0;
}

int  IMetaObject::ConnectByName(const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name)
{
	int count = 0;
	auto signals = GetSignals(signal_name);
	auto slots = receiver->GetSlots(slot_name);
	for (auto signal : signals)
	{
		for (auto slot : slots)
		{
			if (signal->GetSignature() == slot->GetSignature())
			{
				auto connection = slot->Connect(signal);
				if (connection)
				{
					AddConnection(connection, signal_name, slot_name, slot->GetSignature(), receiver);
					++count;
					break;
				}
			}
		}
	}
	return count;
}

bool IMetaObject::ConnectByName(const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name, const TypeInfo& signature)
{
	auto signal = GetSignal(signal_name, signature);
	auto slot = receiver->GetSlot(slot_name, signature);
	if (signal && slot)
	{
		auto connection = slot->Connect(signal);
		if (connection)
		{
			AddConnection(connection, signal_name, slot_name, signature, receiver);
			return true;
		}
	}
	return false;
}

int IMetaObject::ConnectAll(RelayManager* mgr)
{
    auto signals = GetSignalInfo();
    int count = 0;
    for(auto& signal : signals)
    {
        count += ConnectByName(signal->name, mgr);
    }
    return count;
}



void IMetaObject::AddSlot(ISlot* slot, const std::string& name)
{
    _pimpl->_slots[name][slot->GetSignature()] = slot;
	slot->SetParent(this);
}
void IMetaObject::AddSignal(ISignal* sig, const std::string& name)
{
	_pimpl->_signals[name][sig->GetSignature()] = sig;
	sig->SetParent(this);
}

std::vector<std::pair<ISignal*, std::string>> IMetaObject::GetSignals() const
{
    std::vector<std::pair<ISignal*, std::string>> signals;
    for(auto& name_itr : _pimpl->_signals)
    {
        for(auto& sig_itr : name_itr.second)
        {
            signals.push_back(std::make_pair(sig_itr.second, name_itr.first));
        }
    }
    return signals;
}
std::vector<ISignal*> IMetaObject::GetSignals(const std::string& name) const
{
    std::vector<ISignal*> signals;
    auto itr = _pimpl->_signals.find(name);
    if(itr != _pimpl->_signals.end())
    {
        for(auto& sig_itr : itr->second)
        {
            signals.push_back(sig_itr.second);
        }
    }
    return signals;
}
std::vector<std::pair<ISignal*, std::string>> IMetaObject::GetSignals(const TypeInfo& type) const
{
    std::vector<std::pair<ISignal*, std::string>> signals;
    for(auto& name_itr : _pimpl->_signals)
    {
        auto type_itr = name_itr.second.find(type);
        if(type_itr != name_itr.second.end())
        {
            signals.push_back(std::make_pair(type_itr->second, name_itr.first));
        }
    }
    return signals;
}
ISignal* IMetaObject::GetSignal(const std::string& name, const TypeInfo& type) const
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
	return nullptr;
}
void IMetaObject::AddConnection(std::shared_ptr<Connection>& connection, const std::string& signal_name, const std::string& slot_name, const TypeInfo& signature, IMetaObject* obj)
{
	ConnectionInfo info;
	info.connection = connection;
	info.obj = rcc::weak_ptr<IMetaObject>(obj);
	info.signal_name = signal_name;
	info.slot_name = slot_name;
	info.signature = signature;
	_pimpl->_connections.push_back(info);
}