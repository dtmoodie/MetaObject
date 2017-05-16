#include "MetaObject/signals/Connection.hpp"
#include "MetaObject/thread/InterThread.hpp"
#include "MetaObject/signals/ISlot.hpp"
#include "MetaObject/signals/ISignalRelay.hpp"
#include "MetaObject/signals/ISignal.hpp"

using namespace mo;


Connection::~Connection()
{

}

SlotConnection::SlotConnection(ISlot* slot, std::shared_ptr<ISignalRelay> relay):
	_slot(slot), _relay(relay)
{

}
SlotConnection::~SlotConnection()
{
    //disConnect();
}
bool SlotConnection::disConnect()
{
	if (_slot)
	{
		return _slot->disConnect(_relay);
	}
	return false;
}

ClassConnection::ClassConnection(ISlot* slot, std::shared_ptr<ISignalRelay> relay, IMetaObject* obj):
	SlotConnection(slot, relay), _obj(obj)
{

}

ClassConnection::~ClassConnection()
{
    disConnect();	
}

bool ClassConnection::disConnect()
{
	if (SlotConnection::disConnect())
	{
		ThreadSpecificQueue::RemoveFromQueue(_obj);
		return true;
	}
	return false;
}

SignalConnection::SignalConnection(ISignal* signal, std::shared_ptr<ISignalRelay> relay):
	_signal(signal), _relay(relay)
{

}
bool SignalConnection::disConnect()
{
	if (_signal)
	{
		return _signal->disConnect(_relay);
	}
    return false;
}