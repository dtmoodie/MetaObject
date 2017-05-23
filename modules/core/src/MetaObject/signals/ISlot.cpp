#include "MetaObject/signals/ISlot.hpp"
#include "MetaObject/thread/InterThread.hpp"
#include "MetaObject/signals/ISignalRelay.hpp"

using namespace mo;

ISlot::~ISlot()
{
    ThreadSpecificQueue::removeFromQueue(this);
    if (_parent)
    {
        ThreadSpecificQueue::removeFromQueue(_parent);
    }
}

void ISlot::setParent(IMetaObject* parent)
{
    _parent = parent;
}

IMetaObject* ISlot::getParent() const
{
    return _parent;
}
const Context* ISlot::getContext() const
{
    if(_ctx)
        return _ctx;
    if(_parent)
    {
        //return _parent->getContext();
    }
    return nullptr;
}
void ISlot::setContext(Context* ctx)
{
    _ctx = ctx;
}
