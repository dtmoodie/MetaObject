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

std::shared_ptr<Connection> RelayManager::connect(ISlot* slot, const std::string& name, IMetaObject* obj)
{
	auto& relay = GetRelay(slot->getSignature(), name);
	return slot->connect(relay);
}

std::shared_ptr<Connection> RelayManager::connect(ISignal* signal, const std::string& name, IMetaObject* obj)
{
	auto& relay = GetRelay(signal->getSignature(), name);
	return signal->connect(relay);
}

void RelayManager::connectSignal(IMetaObject* obj, const std::string& signal_name)
{
    auto signals = obj->getSignals(signal_name);
    for(auto signal : signals)
    {
        auto Connection = connect(signal, signal_name, obj);
        if(Connection)
        {
            obj->addConnection(Connection, signal_name, signal_name, signal->getSignature(), nullptr);
        }
    }
}

void RelayManager::connectSlot(IMetaObject* obj, const std::string& slot_name)
{
    auto slots = obj->getSlots(slot_name);
    for (auto slot : slots)
    {
        auto Connection = connect(slot, slot_name, obj);
        if (Connection)
        {
            obj->addConnection(Connection, slot_name, slot_name, slot->getSignature(), nullptr);
        }
    }
}

bool RelayManager::connectSignal(IMetaObject* obj, const std::string& name, const TypeInfo& type)
{
	auto signal = obj->getSignal(name, type);
	if (signal)
	{
		auto Connection = connect(signal, name, obj);
		if (Connection)
		{
			obj->addConnection(Connection, name, "", signal->getSignature());
			return true;
		}
	}
	return false;
}
bool RelayManager::connectSlot(IMetaObject* obj, const std::string& name, const TypeInfo& type)
{
	auto slot = obj->getSlot(name, type);
	if (slot)
	{
		auto Connection = connect(slot, name, obj);
		if (Connection)
		{
			obj->addConnection(Connection, "", name, type, nullptr);
		}
	}
	return false;
}

int RelayManager::connectSignals(IMetaObject* obj, const std::string& name)
{
	int count = 0;
	auto signals = obj->getSignals(name);
	for (auto signal : signals)
	{
		count += connect(signal, name, obj) ? 1 : 0;
	}
	return count;
}

int RelayManager::connectSignals(IMetaObject* obj, const TypeInfo& type)
{
	int count = 0;
	auto signals = obj->getSignals(type);
	for (auto signal : signals)
	{
		count += connect(signal.first, signal.second, obj) ? 1 : 0;
	}
	return count;
}

int RelayManager::connectSignals(IMetaObject* obj)
{
	int count = 0;
	auto signals = obj->getSignals();
	for (auto signal : signals)
	{
		count += connect(signal.first, signal.second, obj) ? 1 : 0;
	}
	return count;
}

int RelayManager::connectSlots(IMetaObject* obj, const std::string& name)
{
	int count = 0;
	auto slots = obj->getSlots(name);
	for (auto& slot : slots)
	{
		count += connect(slot, name, obj) ? 1 : 0;
	}
	return count;
}

int RelayManager::connectSlots(IMetaObject* obj, const TypeInfo& type)
{
	int count = 0;
    auto all_slots = obj->getSlots(type);
    for (auto& slot : all_slots)
	{
		count += connect(slot.first, slot.second, obj) ? 1 : 0;
	}
	return count;
}

int RelayManager::connectSlots(IMetaObject* obj)
{
	int count = 0;
    auto all_slots = obj->getSlots();
    for (auto& slot : all_slots )
	{
		count += connect(slot.first, slot.second, obj) ? 1 : 0;
	}
	return count;
}

std::vector<std::shared_ptr<ISignalRelay>> RelayManager::GetRelays(const std::string& name)
{
    std::lock_guard<std::mutex> lock(mtx);
    std::vector<std::shared_ptr<ISignalRelay>> relays;
    for(auto& types : _pimpl->relays)
    {
        if(name.size())
        {
            auto itr = types.second.find(name);
            if(itr != types.second.end())
            {
                relays.push_back(itr->second);
            }
        }else
        {
            for(auto& relay : types.second)
            {
                relays.push_back(relay.second);
            }
        }
    }
    return relays;
}
std::vector<std::pair<std::shared_ptr<ISignalRelay>, std::string>> RelayManager::GetAllRelays()
{
    std::lock_guard<std::mutex> lock(mtx);
    std::vector<std::pair<std::shared_ptr<ISignalRelay>, std::string>> output;
    for(auto& types : _pimpl->relays)
    {
        for(auto& relay : types.second)
        {
            output.emplace_back(relay.second, relay.first);
        }
    }
    return output;
}

std::shared_ptr<ISignalRelay>& RelayManager::GetRelay(const TypeInfo& type, const std::string& name)
{
    std::lock_guard<std::mutex> lock(mtx);
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
