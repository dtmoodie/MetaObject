#include "MetaObject/Signals/ISignal.hpp"
#include "MetaObject/IMetaObject.hpp"
using namespace mo;

const Context* ISignal::getContext() const
{
    if(_ctx)
        return _ctx;
    if(_parent)
    {
        return _parent->getContext();
    }
    return nullptr;
}
void ISignal::setContext(Context* ctx)
{
    _ctx = ctx;
}

IMetaObject* ISignal::getParent() const
{
    return _parent;
}

void ISignal::setParent(IMetaObject* parent)
{
    _parent = parent;
}
