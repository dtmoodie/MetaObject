#include "MetaObject/IMetaObject.hpp"
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
    if(_pimpl->connections.find(sig) != _pimpl->connections.end());
    {
        _pimpl->connections.erase(sig);
        return true;
    }
    return false;
}

void IMetaObject::AddConnection(std::shared_ptr<Connection>& connection, ISignal* sig)
{
    _pimpl->connections[sig] = connection;
}