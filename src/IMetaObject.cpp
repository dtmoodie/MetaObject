#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Signals/ISignal.hpp"
#include "MetaObject/Signals/ISlot.hpp"
#include "MetaObject/Signals/SignalInfo.hpp"
#include "MetaObject/Signals/SlotInfo.hpp"
#include "MetaObject/Parameters/IParameter.hpp"
#include "MetaObject/Logging/Log.hpp"
#include "MetaObject/Detail/IMetaObject_pImpl.hpp"
#include "MetaObject/Parameters/InputParameter.hpp"
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
                auto signals = this->GetSignals(connection.signal_name);
                auto slots = connection.obj->GetSlots(connection.slot_name);
                for (auto signal : signals)
                {
                    for (auto slot : slots)
                    {
                        if (signal->GetSignature() == slot->GetSignature())
                        {
                            auto connection_ = slot->Connect(signal);
                            if (connection_)
                            {
                                connection.connection = connection_;
                            }
                            break;
                        }
                    }
                }
            }
        }
    }
}

void IMetaObject::Serialize(ISimpleSerializer *pSerializer)
{
    SerializeConnections(pSerializer);
}

void IMetaObject::SerializeConnections(ISimpleSerializer* pSerializer)
{
    SERIALIZE(_pimpl->_connections);
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
    return output;
}

std::vector<InputParameter*> IMetaObject::GetInputs(const TypeInfo& type_filter) const
{
    std::vector<InputParameter*> output;
    for(auto param : _pimpl->_parameters)
    {
        if(param.second->CheckFlags(Input_e))
        {
            if(param.second->GetTypeInfo() == type_filter)
                if(auto out = dynamic_cast<InputParameter*>(param.second))
                    output.push_back(out);
        }
    }
    return output;
}


std::weak_ptr<IParameter> IMetaObject::AddParameter(std::shared_ptr<IParameter> param)
{
    _pimpl->_parameters[param->GetTreeName()] = param.get();
    return param;
}

IParameter* IMetaObject::AddParameter(IParameter* param)
{
    _pimpl->_parameters[param->GetTreeName()] = param;
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