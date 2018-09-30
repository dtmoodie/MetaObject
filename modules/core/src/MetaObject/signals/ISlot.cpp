#include "MetaObject/signals/ISlot.hpp"
#include "MetaObject/signals/ISignalRelay.hpp"
#include "MetaObject/thread/InterThread.hpp"

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
Context* ISlot::getContext() const
{
    if (_ctx)
        return _ctx;
    if (_parent)
    {
        // return _parent->getContext();
    }
    return nullptr;
}
void ISlot::setContext(Context* ctx)
{
    _ctx = ctx;
}
