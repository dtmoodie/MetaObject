#include "MetaObject/Signals/ISignal.hpp"
#include "MetaObject/IMetaObject.hpp"
using namespace mo;

const Context* ISignal::GetContext() const
{
    if(_ctx)
        return _ctx;
    if(_parent)
    {
        return _parent->GetContext();
    }
    return nullptr;
}
void ISignal::SetContext(Context* ctx)
{
    _ctx = ctx;
}

IMetaObject* ISignal::GetParent() const
{
    return _parent;
}

void ISignal::SetParent(IMetaObject* parent)
{
    _parent = parent;
}
