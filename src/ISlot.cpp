#include "MetaObject/Signals/ISlot.hpp"
#include "MetaObject/Thread/InterThread.hpp"

using namespace mo;
ISlot::~ISlot()
{
	ThreadSpecificQueue::RemoveFromQueue(this);
}