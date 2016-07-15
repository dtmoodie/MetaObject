#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Signals/ICallback.hpp"
#include "MetaObject/Signals/ISignal.hpp"
#include "MetaObject/Signals/ISlot.hpp"
#include "MetaObject/Signals/SignalInfo.hpp"
#include "MetaObject/Signals/SlotInfo.hpp"
#include <map>
#include <set>
using namespace mo;


struct IMetaObject::impl
{
    std::map<ISignal*, std::shared_ptr<Connection>> connections;
    std::map<std::string, std::set<ICallback*>>        _callback_name_map;
    std::map<TypeInfo, std::set<ICallback*>>           _callback_signature_map;
    std::set<ICallback*>                  _explicit_callbacks;
    std::map<std::string, std::map<TypeInfo, std::weak_ptr<ISignal>>> _signals;
    std::map<std::string, std::map<TypeInfo, ISlot*>> _slots;
};

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
    InitCallbacks(firstInit);
    InitParameters(firstInit);
    BindSlots();
}
void IMetaObject::SetContext(Context* ctx)
{
    _ctx = ctx;
}
int IMetaObject::DisconnectByName(const std::string& name)
{
    auto signals = this->GetSignals(name);
    int count = 0;
    for(auto& sig : signals)
    {
        if(_pimpl->connections.find(sig.get()) != _pimpl->connections.end())
        {
            _pimpl->connections.erase(sig.get());
            ++count;
        }
    }
    auto itr = _pimpl->_callback_name_map.find(name);
    if(itr != _pimpl->_callback_name_map.end())
    {
        for(auto& cb : itr->second)
        {
            cb->Disconnect();
            ++count;
        }
    }

    return count;
}
bool IMetaObject::Disconnect(ISignal* sig)
{
    if(_pimpl->connections.find(sig) != _pimpl->connections.end())
    {
        _pimpl->connections.erase(sig);
        return true;
    }
    return false;
}

int IMetaObject::Disconnect(IMetaObject* obj)
{
    auto signals = obj->GetSignals();
    int count = 0;
    for(auto signal : signals)
    {
        count += Disconnect(signal.get()) ? 1 : 0;
    }
    // Disconnect callbacks
    for(auto callback : _pimpl->_explicit_callbacks)
    {
        if(callback->receiver == obj)
        {
            callback->Disconnect();
            ++count;
        }
    }
    return count;
}
void IMetaObject::AddConnection(std::shared_ptr<Connection>& connection, ISignal* sig)
{
    _pimpl->connections[sig] = connection;
}

std::vector<std::weak_ptr<IParameter>> IMetaObject::GetDisplayParameters() const
{
    std::vector<std::weak_ptr<IParameter>> output;
    for(auto& param : _explicit_parameters)
    {
        output.push_back(std::weak_ptr<IParameter>(param));
    }
    return output;
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
std::vector<CallbackInfo*>  IMetaObject::GetCallbackInfo(const std::string& name) const
{
    auto output = GetCallbackInfo();

    return output;
}
void IMetaObject::AddCallback(ICallback* cb, const std::string& name)
{
    _pimpl->_callback_name_map[name].insert(cb);
    _pimpl->_callback_signature_map[cb->GetSignature()].insert(cb);
    _pimpl->_explicit_callbacks.insert(cb);
}
std::vector<ISlot*> IMetaObject::GetSlots() const
{
    std::vector<ISlot*> slots;
    for(auto itr1 : _pimpl->_slots)
    {
        for(auto itr2: itr1.second)
        {
            slots.push_back(itr2.second);
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

std::vector<ISlot*> IMetaObject::GetSlots(const TypeInfo& signature) const
{
    std::vector<ISlot*> output;
    for(auto& type : _pimpl->_slots)
    {
        auto itr = type.second.find(signature);
        if(itr != type.second.end())
        {
            output.push_back(itr->second);
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

std::vector<ICallback*> IMetaObject::GetCallbacks() const
{
    std::vector<ICallback*> output;
    output.insert(output.end(), _pimpl->_explicit_callbacks.begin(), _pimpl->_explicit_callbacks.end());
    return output;
}

std::vector<ICallback*> IMetaObject::GetCallbacks(const std::string& name) const
{
    std::vector<ICallback*> output;
    auto itr = _pimpl->_callback_name_map.find(name);
    if(itr != _pimpl->_callback_name_map.end())
    {
        output.insert(output.end(), itr->second.begin(), itr->second.end());
    }
    return output;
}

std::vector<ICallback*> IMetaObject::GetCallbacks(const TypeInfo& signature) const
{
    std::vector<ICallback*> output;
    auto itr = _pimpl->_callback_signature_map.find(signature);
    if(itr != _pimpl->_callback_signature_map.end())
    {
        output.insert(output.end(), itr->second.begin(), itr->second.end());
    }
    return output;
}

ICallback* IMetaObject::GetCallback(const std::string& name, const TypeInfo& signature) const
{
    std::vector<ICallback*> output;
    auto itr1 = _pimpl->_callback_name_map.find(name);
    if(itr1 != _pimpl->_callback_name_map.end())
    {
        for(auto& cb : itr1->second)
        {
            if(cb->GetSignature() == signature)
                return cb;
        }
    }
    return nullptr;
}

int IMetaObject::ConnectCallbacks(const std::string& callback_name, const std::string& slot_name, IMetaObject* obj, bool force_queue)
{
    auto slots = obj->GetSlots(slot_name);
    auto cbs = GetCallbacks(callback_name);
    int count = 0;
    for(auto& slot : slots)
    {
        bool found = false;
        for(auto cb : cbs)
        {
            found = slot->Connect(cb);
        }
        if(found)
        {
            ++count;
            break;
        }
    }
    return count;
}

bool IMetaObject::ConnectCallback(const TypeInfo& signature, const std::string& callback_name, const std::string& slot_name, IMetaObject* slot_owner, bool force_queue)
{
    auto slot = slot_owner->GetSlot(slot_name, signature);
    auto cb = this->GetCallback(callback_name, signature);
    if(slot && cb)
    {
        return slot->Connect(cb);
    }
    return false;
}
bool IMetaObject::ConnectCallback(ICallback* callback, const std::string& name, bool force_queue)
{
    auto itr = _pimpl->_slots.find(name);
    if(itr != _pimpl->_slots.end())
    {
        auto itr2 = itr->second.find(callback->GetSignature());
        if(itr2 != itr->second.end())
        {
            return itr2->second->Connect(callback);
        }
    }
    return false;
}
bool IMetaObject::ConnectCallback(ISlot* slot, const std::string& callback_name, bool force_queue)
{
    auto itr = _pimpl->_callback_name_map.find(callback_name);
    if(itr != _pimpl->_callback_name_map.end())
    {
        for(auto& cb : itr->second)
        {
            if(slot->Connect(cb))
                return true;
        }
    }
    return false;
}

int IMetaObject::ConnectCallbacks(IMetaObject* obj, bool force_queue)
{
    return 0;
}
void IMetaObject::AddSignal(std::weak_ptr<ISignal> signal, const std::string& name)
{
    _pimpl->_signals[name][signal.lock()->GetSignature()] = (signal);
}
int IMetaObject::ConnectAll(SignalManager* mgr)
{
    auto signals = GetSignalInfo();
    int count = 0;
    for(auto& signal : signals)
    {
        count += ConnectByName(signal->name, mgr);
    }
    return count;
}
int IMetaObject::ConnectByName(const std::string& name, IMetaObject* obj)
{
    int count = 0;
    auto signals = obj->GetSignals(name);
    for(auto& signal : signals)
    {
        count += ConnectByName(name, signal) ? 1 : 0;
    }
    return count;
}

void IMetaObject::AddSlot(ISlot* slot, const std::string& name)
{
    _pimpl->_slots[name][slot->GetSignature()] = slot;
}

std::vector<std::shared_ptr<ISignal>> IMetaObject::GetSignals() const
{
    std::vector<std::shared_ptr<ISignal>> signals;
    for(auto& name_itr : _pimpl->_signals)
    {
        for(auto& sig_itr : name_itr.second)
        {
            if(!sig_itr.second.expired())
                signals.push_back(sig_itr.second.lock());
        }
    }
    return signals;
}
std::vector<std::shared_ptr<ISignal>> IMetaObject::GetSignals(const std::string& name) const
{
    std::vector<std::shared_ptr<ISignal>> signals;
    auto itr = _pimpl->_signals.find(name);
    if(itr != _pimpl->_signals.end())
    {
        for(auto& sig_itr : itr->second)
        {
            if(!sig_itr.second.expired())
                signals.push_back(sig_itr.second.lock());
        }
    }
    return signals;
}
std::vector<std::shared_ptr<ISignal>> IMetaObject::GetSignals(const TypeInfo& type) const
{
    std::vector<std::shared_ptr<ISignal>> signals;
    for(auto& name_itr : _pimpl->_signals)
    {
        auto type_itr = name_itr.second.find(type);
        if(type_itr != name_itr.second.end())
        {
            if(!type_itr->second.expired())
                signals.push_back(type_itr->second.lock());
        }
    }
    return signals;
}