#pragma once
#include "MetaObject/object/IMetaObject.hpp"
#include "MetaObject/detail/MetaObjectMacros.hpp"

struct printable: public mo::IMetaObject
{
	MO_BEGIN(printable)
	MO_END;
    virtual void print();
};