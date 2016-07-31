#include "MetaObject/Signals/ISignal.hpp"

using namespace mo;

IMetaObject* ISignal::GetParent() const
{
	return _parent;
}

void ISignal::SetParent(IMetaObject* parent)
{
	_parent = parent;
}