#pragma once
#include "MetaObject/object/MetaObject.hpp"
#include "MetaObject/object/detail/MetaObjectMacros.hpp"

struct printable: public mo::MetaObject
{
	MO_BEGIN(printable)
	MO_END;
    virtual void print();
};
