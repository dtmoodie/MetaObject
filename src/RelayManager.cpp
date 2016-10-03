#include "MetaObject/Signals/RelayManager.hpp"
#include "MetaObject/Signals/ISignalRelay.hpp"
#include "MetaObject/Signals/RelayFactory.hpp"
#include "MetaObject/Signals/ISlot.hpp"
#include "MetaObject/Signals/ISignal.hpp"
#include "MetaObject/IMetaObject.hpp"
#include <map>
#include <memory>
using namespace mo;

struct RelayManager::impl
{
	std::map<TypeInfo, std::map<std::string, std::shared_ptr<ISignalRelay>>> relays;
};

RelayManager::RelayManager()
{
	_pimpl = new impl();
}

RelayManager::~RelayManager()
{
	delete _pimpl;
}

RelayManager* g_inst = nullptr;
RelayManager* RelayManager::Instance()
{
	if (g_inst == nullptr)
		g_inst = new RelayManager();
	return g_inst;
}

void RelayManager::SetInstance(RelayManager* inst)
{
	g_inst = inst;
}

std::shared_ptr<Connection> RelayManager::Connect(ISlot* slot, const std::string& name, IMetaObject* obj)
{
	auto& relay = GetRelay(slot->GetSignature(), name);
	return slot->Connect(relay);
}

std::shared_ptr<Connection> RelayManager::Connect(ISignal* signal, const std::string& name, IMetaObject* obj)
{
	auto& relay = GetRelay(signal->GetSignature(), name);
	return signal->Connect(relay);
}

bool RelayManager::ConnectSignal(IMetaObject* obj, const std::string& name, const TypeInfo& type)
{
	auto signal = obj->GetSignal(name, type);
	if (signal)
	{
		auto connection = Connect(signal, name, obj);
		if (connection)
		{
			obj->AddConnection(connection, name, "", signal->GetSignature());
			return true;
		}
	}
	return false;
}
bool RelayManager::ConnectSlot(IMetaObject* obj, const std::string& name, const TypeInfo& type)
{
	auto slot = obj->GetSlot(name, type);
	if (slot)
	{
		auto connection = Connect(slot, name, obj);
		if (connection)
		{
			obj->AddConnection(connection, "", name, type, nullptr);
		}
	}
	return false;
}

int RelayManager::ConnectSignals(IMetaObject* obj, const std::string& name)
{
	int count = 0;
	auto signals = obj->GetSignals(name);
	for (auto signal : signals)
	{
		count += Connect(signal, name, obj) ? 1 : 0;
	}
	return count;
}

int RelayManager::ConnectSignals(IMetaObject* obj, const TypeInfo& type)
{
	int count = 0;
	auto signals = obj->GetSignals(type);
	for (auto signal : signals)
	{
		count += Connect(signal.first, signal.second, obj) ? 1 : 0;
	}
	return count;
}

int RelayManager::ConnectSignals(IMetaObject* obj)
{
	int count = 0;
	auto signals = obj->GetSignals();
	for (auto signal : signals)
	{
		count += Connect(signal.first, signal.second, obj) ? 1 : 0;
	}
	return count;


}

int RelayManager::ConnectSlots(IMetaObject* obj, const std::string& name)
{
	int count = 0;
	auto slots = obj->GetSlots(name);
	for (auto& slot : slots)
	{
		count += Connect(slot, name, obj) ? 1 : 0;
	}
	return count;
}

int RelayManager::ConnectSlots(IMetaObject* obj, const TypeInfo& type)
{
	int count = 0;
	auto slots = obj->GetSlots(type);
	for (auto& slot : slots)
	{
		count += Connect(slot.first, slot.second, obj) ? 1 : 0;
	}
	return count;
}

int RelayManager::ConnectSlots(IMetaObject* obj)
{
	int count = 0;
	auto slots = obj->GetSlots();
	for (auto& slot : slots)
	{
		count += Connect(slot.first, slot.second, obj) ? 1 : 0;
	}
	return count;
}

std::vector<std::shared_ptr<ISignalRelay>> RelayManager::GetRelays(const std::string& name)
{
    std::vector<std::shared_ptr<ISignalRelay>> relays;
    for(auto& types : _pimpl->relays)
    {
        auto itr = types.second.find(name);
        if(itr != types.second.end())
        {
            relays.push_back(itr->second);
        }
    }
    return relays;
}
std::shared_ptr<ISignalRelay>& RelayManager::GetRelay(const TypeInfo& type, const std::string& name)
{
	return _pimpl->relays[type][name];
}

bool RelayManager::exists(const std::string& name, TypeInfo type)
{
	auto itr1 = _pimpl->relays.find(type);
	if (itr1 != _pimpl->relays.end())
	{
		auto itr2 = itr1->second.find(name);
		if (itr2 != itr1->second.end())
		{
			return true;
		}
	}
	return false;
}

/*#include "MetaObject/Signals/RelayManager.hpp"

#include <map>
#include <set>
#include <sstream>
using namespace mo;

struct SignalManager::impl
{
    // Signal signature, signal name, pointer to signal
    std::map<TypeInfo, std::map<std::string, std::shared_ptr<ISignal>>> _signals;

	// Mapping of which objects emit signals
	std::map<TypeInfo, std::map<std::string, std::vector<sender>>> _registered_senders;
	std::map<TypeInfo, std::map<std::string, std::vector<receiver>>> _registered_receivers;
};


signal_registry::signal_registry()
{

}

signal_registry* signal_registry::instance()
{
    static signal_registry* g_inst = nullptr;
	if(g_inst == nullptr)
		g_inst = new signal_registry();
    return g_inst;
}

void signal_registry::add_signal(TypeInfo sender, std::string name, TypeInfo signature)
{
    _static_signal_registry[sender][name].push_back(signature);
}

void signal_registry::add_slot(TypeInfo receiver, std::string name, TypeInfo signature)
{
    _static_slot_registry[receiver][name].push_back(signature);
}

void signal_registry::add_python_registration_function(TypeInfo sender, std::function<void(void)> f)
{
    _static_python_registration_functions[sender] = f;
}
std::string signal_registry::print_signal_map()
{
    std::stringstream ss;
    std::vector<TypeInfo> _printedSenders;
    for(auto& senders : _static_signal_registry)
    {
        ss << "--------------------------\n";
        ss << senders.first.name() << "\n";
        if(senders.second.size())
            ss << "  ---- SIGNALS ----\n";
        for(auto& sig : senders.second)
        {
            ss << "    " << sig.first << "\n";
            for(auto& s : sig.second)
            {
                ss << "    - " << s.name() << "\n";
            }
        }
        auto itr = _static_slot_registry.find(senders.first);
        if(itr != _static_slot_registry.end())
        {
            ss << "  ---- SLOTS ----\n";
            for(auto & slot : itr->second)
            {
                ss << "    " << slot.first << "\n";
                for(auto& overload : slot.second)
                {
                    ss << "      - " << overload.name() << "\n";
                }
            }
        }
        _printedSenders.push_back(senders.first);
    }
    // Now go through all receiver objects and print the ones that don't have signals
    for(auto& receiver : _static_slot_registry)
    {
        if(std::count(_printedSenders.begin(), _printedSenders.end(), receiver.first) == 0)
        {
            ss << "--------------------------\n";
            ss << receiver.first.name() << "\n";
            ss << "  ---- SLOTS ----\n";
            for(auto& slot : receiver.second)
            {
                ss << "    " << slot.first;
                for(auto& overload : slot.second)
                {
                    ss << "     - " << overload.name() << "\n";
                }            
            }
        }
    }
    return ss.str();
}



SignalManager::SignalManager()
{
    _pimpl = new SignalManager::impl();
}
SignalManager::~SignalManager()
{
    delete _pimpl;
}
static SignalManager* g_instance = nullptr;


SignalManager* SignalManager::Instance()
{
    if (g_instance == nullptr)
    {
        g_instance = new SignalManager();
    }
    return g_instance;
}
void SignalManager::SetInstance(SignalManager* inst)
{
    g_instance = inst;
}


std::shared_ptr<ISignal>& SignalManager::GetSignal(const std::string& name, TypeInfo type)
{
    return _pimpl->_signals[type][name];
}

std::weak_ptr<ISignal> SignalManager::GetSignalOptional(const std::string& name, const TypeInfo& type)
{
    auto itr = _pimpl->_signals.find(type);
    if(itr != _pimpl->_signals.end())
    {
        auto itr2 = itr->second.find(name);
        if(itr2 != itr->second.end())
        {
            return itr2->second;
        }
    }
    return std::weak_ptr<ISignal>();
}

std::weak_ptr<ISignal> SignalManager::GetSignalOptional(const std::string& name, const std::string& type)
{
    for(auto& signature : _pimpl->_signals)
    {
        std::string type_name = signature.first.name();
        if(type == signature.first.name())
        {
            auto itr2 = signature.second.find(name);
            if(itr2 != signature.second.end())
            {
                return itr2->second;
            }
        }
    }
    return std::weak_ptr<ISignal>();
}


/*void signal_manager::register_sender(mo::TypeInfo signal_signature, std::string signal_name, mo::TypeInfo sender_type, void* sender_ptr, std::string desc, std::string tooltip)
{
	sender s;
    s.description = desc;
    s.type = sender_type;
    s.ptr = sender_ptr;
    s.signature = signal_signature;
    s.signal_name = signal_name;
    s.tooltip = tooltip;
    _registered_senders[signal_signature][signal_name].push_back(s);
}
void signal_manager::register_sender(mo::TypeInfo signal_signature, std::string signal_name, std::string desc, std::string file_name, int line_number, std::string tooltip)
{
    sender s;
    s.description = desc;
    s.file = file_name;
    s.line = line_number;
    s.signature = signal_signature;
    s.signal_name = signal_name;
    s.tooltip = tooltip;
    _registered_senders[signal_signature][signal_name].push_back(s);
}
void signal_manager::remove_sender(void* sender_ptr, std::string name)
{

}
void signal_manager::remove_sender(std::string file_name, int line_number, std::string name)
{

}

void signal_manager::register_receiver(mo::TypeInfo signal_signature, const std::string& signal_name, mo::TypeInfo receiver_type, void* receiver_ptr, const std::string& desc, const std::string& tooltip)
{
	receiver r;
    r.description = desc;
    r.tooltip = tooltip;
    r.ptr = receiver_ptr;
    r.type = receiver_type;
    r.signature = signal_signature;
    r.signal_name = signal_name;
    _registered_receivers[signal_signature][signal_name].push_back(r);
}

void signal_manager::register_receiver(mo::TypeInfo signal_signature, const std::string& signal_name, int line_number, const std::string& file_name, const std::string& desc, const std::string& tooltip)
{
	receiver r;
    r.description = desc;
    r.tooltip = tooltip;
    r.line = line_number;
    r.file = file_name;
    r.signature = signal_signature;
    r.signal_name = signal_name;
    _registered_receivers[signal_signature][signal_name].push_back(r);
}
void signal_manager::register_receiver(mo::TypeInfo signal_signature, const std::string& signal_name, signaler* receiver)
{
    register_receiver(signal_signature, signal_name, mo::TypeInfo(typeid(*receiver)), receiver, receiver->get_description(), receiver->get_slot_description(signal_name));
}

std::vector<std::weak_ptr<ISignal>> SignalManager::GetSignals(std::string name)
{
	std::vector<std::weak_ptr<ISignal>> output;
	for (auto& itr : _pimpl->_signals)
	{
		auto itr2 = itr.second.find(name);
		if (itr2 != itr.second.end())
		{
			output.push_back(itr2->second);
		}
	}
	return output;
}
std::vector<std::weak_ptr<ISignal>> SignalManager::GetSignals(TypeInfo type)
{
	std::vector<std::weak_ptr<ISignal>> output;
	for (auto& itr : _pimpl->_signals[type])
	{
		output.push_back(itr.second);
	}
	return output;
}
std::vector<std::weak_ptr<ISignal>> SignalManager::GetSignals(TypeInfo type, std::string name)
{
	std::vector<std::weak_ptr<ISignal>> output;
	for (auto& itr : _pimpl->_signals)
	{
		for (auto& itr2 : itr.second)
		{
			output.push_back(itr2.second);
		}
	}
	return output;
}
std::vector<std::weak_ptr<ISignal>> SignalManager::GetSignals()
{
	std::vector<std::weak_ptr<ISignal>> output;

	return output;
}
std::vector<std::string> SignalManager::GetSignalNames()
{
    std::set<std::string> names;
    for(auto& sig : _pimpl->_signals)
    {
        for(auto& sig1 : sig.second)
        {
            names.insert(sig1.first);
        }
    }
    std::vector<std::string> output;
    output.insert(output.begin(), names.begin(), names.end());
    return output;
}
void SignalManager::PrintSignalMap()
{
	for (auto& type : _signals)
	{
		for (auto& sig : type.second)
		{
			std::cout << "[" << sig.first << "] <" << type.first.name() << ">\n";
			std::cout << " Senders:\n";
			//auto cleaned_type = _signal_signature_map.find(type.first);
			
				//std::cout << "   " << cleaned_type->second.name();
			auto signature = sig.second->get_signal_type();
			auto itr = _registered_sender_objects[signature][sig.first];
			for (auto sender : itr)
			{
				std::cout << "    " << std::get<0>(sender).name() << " [" << std::get<1>(sender) << "] " << std::get<2>(sender) << "\n";
			}
			auto itr2 = _registered_senders[signature][sig.first];
			for (auto sender : itr2)
			{
				std::cout << "    " << std::get<1>(sender) << " [" << std::get<0>(sender) << "] " << std::get<2>(sender) << "\n";
			}
			std::cout << " Receivers:\n";
			auto itr3 = _registered_receivers[signature][sig.first];
			for (auto receiver : itr3)
			{
				std::cout << "    " << std::get<1>(receiver) << " [" << std::get<0>(receiver) << "] " << std::get<2>(receiver) << "\n";
			}
			auto itr4 = _registered_receiver_objects[signature][sig.first];
			for (auto receiver : itr4)
			{
				std::cout << "    " << std::get<0>(receiver).name() << " [" << std::get<1>(receiver) << "] " << std::get<2>(receiver) << "\n";
			}
			
			
		}

	}
}

std::vector<receiver> signal_manager::get_receivers(mo::TypeInfo type, std::string name)
{
    return _registered_receivers[type][name];
}

std::vector<receiver> signal_manager::get_receivers(mo::TypeInfo type)
{
    std::vector<receiver> output;
    auto itr = _registered_receivers.find(type);
    if(itr != _registered_receivers.end())
    {
        for(auto& r : itr->second)
        {
            output.insert(output.end(), r.second.begin(), r.second.end());
        }
    }
    return output;
}

std::vector<receiver> signal_manager::get_receivers(std::string name)
{
    std::vector<receiver> output;
    for(auto& sig_itr : _registered_receivers)
    {
        auto name_itr = sig_itr.second.find(name);
        if(name_itr != sig_itr.second.end())
        {
            output.insert(output.end(), name_itr->second.begin(), name_itr->second.end());
        }
    }
    return output;
}
std::vector<receiver> signal_manager::get_receivers()
{
    std::vector<receiver> output;
    for(auto& sig : _registered_receivers)
    {
        for(auto& name: sig.second)
        {
            output.insert(output.end(), name.second.begin(), name.second.end());
        }
    }
    return output;
}
std::vector<sender> signal_manager::get_senders(mo::TypeInfo type, std::string name)
{
    return _registered_senders[type][name];
}

std::vector<sender> signal_manager::get_senders(mo::TypeInfo type)
{
    std::vector<sender> output;
    auto itr = _registered_senders.find(type);
    if(itr != _registered_senders.end())
    {
        for(auto& r : itr->second)
        {
            output.insert(output.end(), r.second.begin(), r.second.end());
        }
    }
    return output;
}

std::vector<sender> signal_manager::get_senders(std::string name)
{
    std::vector<sender> output;
    for(auto& sig_itr : _registered_senders)
    {
        auto name_itr = sig_itr.second.find(name);
        if(name_itr != sig_itr.second.end())
        {
            output.insert(output.end(), name_itr->second.begin(), name_itr->second.end());
        }
    }
    return output;
}
std::vector<sender> signal_manager::get_senders()
{
        std::vector<sender> output;
    for(auto& sig : _registered_senders)
    {
        for(auto& name: sig.second)
        {
            output.insert(output.end(), name.second.begin(), name.second.end());
        }
    }
    return output;
}
void Signals::register_sender(signaler* sender, const std::string& signal_name, mo::TypeInfo signal_signature, signal_manager* mgr)
{
    mgr->register_sender(signal_signature, signal_name, sender);
}
void Signals::register_receiver(signaler* receiver, const std::string& signal_name, mo::TypeInfo signal_signature, signal_manager* mgr)
{
    mgr->register_receiver(signal_signature, signal_name, receiver);
}
*/
