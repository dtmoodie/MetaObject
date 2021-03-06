#include "MetaObject/signals/Connection.hpp"
#include "MetaObject/signals/ISignal.hpp"
#include "MetaObject/signals/ISignalRelay.hpp"
#include "MetaObject/signals/ISlot.hpp"
//#include "MetaObject/thread/InterThread.hpp"

using namespace mo;

Connection::~Connection()
{
}

SlotConnection::SlotConnection(ISlot* slot, const std::shared_ptr<ISignalRelay>& relay)
    : _slot(slot)
    , _relay(relay)
{
}

SlotConnection::~SlotConnection()
{
    // disconnect();
}
bool SlotConnection::disconnect()
{
    if (_slot)
    {
        return _slot->disconnect(_relay);
    }
    return false;
}

ClassConnection::ClassConnection(ISlot* slot, std::shared_ptr<ISignalRelay> relay, IMetaObject* obj)
    : SlotConnection(slot, relay)
    , _obj(obj)
{
}

ClassConnection::~ClassConnection()
{
    disconnect();
}

bool ClassConnection::disconnect()
{
    if (SlotConnection::disconnect())
    {
        // ThreadSpecificQueue::removeFromQueue(_obj);
        return true;
    }
    return false;
}

SignalConnection::SignalConnection(ISignal* signal, const std::shared_ptr<ISignalRelay>& relay)
    : _signal(signal)
    , _relay(relay)
{
}

bool SignalConnection::disconnect()
{
    if (_signal)
    {
        auto relay = _relay.lock();
        if (relay)
        {
            return _signal->disconnect(*relay);
        }
    }
    return false;
}

ConnectionSet::ConnectionSet(std::vector<Connection::Ptr_t>&& connections)
    : m_connections(connections)
{
}

bool ConnectionSet::disconnect()
{
    m_connections.clear();
    return true;
}
