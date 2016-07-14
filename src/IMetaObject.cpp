#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Signals/ICallback.hpp"
#include "MetaObject/Signals/ISignal.hpp"
#include "MetaObject/Signals/ISlot.hpp"
#include <map>
using namespace mo;


struct IMetaObject::impl
{
    std::map<ISignal*, std::shared_ptr<Connection>> connections;
    std::map<std::string, std::vector<ICallback*>>        _callback_name_map;
    std::map<TypeInfo, std::vector<ICallback*>>           _callback_signature_map;
    std::vector<ICallback*>                  _explicit_callbacks;
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
    auto signals = obj->GetAllSignals();
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
std::vector<SlotInfo*>      IMetaObject::GetSlotInfo(const std::string& name) const
{
    auto output = GetSlotInfo();

    return output;
}
std::vector<CallbackInfo*>  IMetaObject::GetCallbackInfo(const std::string& name) const
{
    auto output = GetCallbackInfo();

    return output;
}
void IMetaObject::AddCallback(ICallback* cb, const std::string& name)
{
    _pimpl->_callback_name_map[name].push_back(cb);
    _pimpl->_callback_signature_map[cb->GetSignature()].push_back(cb);
    _pimpl->_explicit_callbacks.push_back(cb);
}

int IMetaObject::ConnectCallbacks(const std::string& callback_name, const std::string& slot_name, IMetaObject* obj, bool force_queue)
{
    return 0;
}

bool IMetaObject::ConnectCallback(ICallback* callback, const std::string& name, bool force_queue)
{
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

void IMetaObject::AddSlot(ISlot* slot, const std::string& name)
{
    _pimpl->_slots[name][slot->GetSignature()] = slot;
}

std::vector<std::shared_ptr<ISignal>> IMetaObject::GetAllSignals() const
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