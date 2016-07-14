#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Signals/ICallback.hpp"
#include <map>
using namespace mo;


struct IMetaObject::impl
{
    std::map<ISignal*, std::shared_ptr<Connection>> connections;
};

IMetaObject::IMetaObject()
{
    _pimpl = new impl();
}

IMetaObject::~IMetaObject()
{
    delete _pimpl;
}

void IMetaObject::SetContext(Context* ctx)
{
    _ctx = ctx;
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
        count += Disconnect(signal) ? 1 : 0;
    }
    // Disconnect callbacks
    for(auto callback : _explicit_callbacks)
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