#include "MetaObject/core/singletons.hpp"
#include "MetaObject/object/RelayManager.hpp"
#include "MetaObject/object/IMetaObject.hpp"
#include "MetaObject/signals/ISignal.hpp"
#include "MetaObject/signals/ISignalRelay.hpp"
#include "MetaObject/signals/ISlot.hpp"
#include "MetaObject/signals/RelayFactory.hpp"
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

std::shared_ptr<RelayManager> RelayManager::instance()
{
    return mo::getSingleton<RelayManager>();
}

std::shared_ptr<Connection> RelayManager::connect(ISlot* slot, const std::string& name, IMetaObject* obj)
{
    auto& relay = getRelay(slot->getSignature(), name);
    return slot->connect(relay);
}

std::shared_ptr<Connection> RelayManager::connect(ISignal* signal, const std::string& name, IMetaObject* obj)
{
    auto& relay = getRelay(signal->getSignature(), name);
    return signal->connect(relay);
}

void RelayManager::connectSignal(IMetaObject* obj, const std::string& signal_name)
{
    auto signals = obj->getSignals(signal_name);
    for (auto signal : signals)
    {
        auto connection = connect(signal, signal_name, obj);
        if (connection)
        {
            obj->addConnection(std::move(connection), signal_name, signal_name, signal->getSignature(), nullptr);
        }
    }
}

void RelayManager::connectSlot(IMetaObject* obj, const std::string& slot_name)
{
    auto slots = obj->getSlots(slot_name);
    for (auto slot : slots)
    {
        auto connection = connect(slot, slot_name, obj);
        if (connection)
        {
            obj->addConnection(std::move(connection), slot_name, slot_name, slot->getSignature(), nullptr);
        }
    }
}

bool RelayManager::connectSignal(IMetaObject* obj, const std::string& name, const TypeInfo& type)
{
    auto signal = obj->getSignal(name, type);
    if (signal)
    {
        auto connection = connect(signal, name, obj);
        if (connection)
        {
            obj->addConnection(std::move(connection), name, "", signal->getSignature());
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
        auto connection = connect(slot, name, obj);
        if (connection)
        {
            obj->addConnection(std::move(connection), "", name, type, nullptr);
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
    for (auto& slot : all_slots)
    {
        count += connect(slot.first, slot.second, obj) ? 1 : 0;
    }
    return count;
}

std::vector<std::shared_ptr<ISignalRelay>> RelayManager::getRelays(const std::string& name)
{
    std::lock_guard<std::mutex> lock(mtx);
    std::vector<std::shared_ptr<ISignalRelay>> relays;
    for (auto& types : _pimpl->relays)
    {
        if (name.size())
        {
            auto itr = types.second.find(name);
            if (itr != types.second.end())
            {
                relays.push_back(itr->second);
            }
        }
        else
        {
            for (auto& relay : types.second)
            {
                relays.push_back(relay.second);
            }
        }
    }
    return relays;
}
std::vector<std::pair<std::shared_ptr<ISignalRelay>, std::string>> RelayManager::getAllRelays()
{
    std::lock_guard<std::mutex> lock(mtx);
    std::vector<std::pair<std::shared_ptr<ISignalRelay>, std::string>> output;
    for (auto& types : _pimpl->relays)
    {
        for (auto& relay : types.second)
        {
            output.emplace_back(relay.second, relay.first);
        }
    }
    return output;
}

std::shared_ptr<ISignalRelay>& RelayManager::getRelay(const TypeInfo& type, const std::string& name)
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
