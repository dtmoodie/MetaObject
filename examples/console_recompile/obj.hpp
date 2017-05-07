#pragma once
#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Detail/MetaObjectMacros.hpp"

struct printable: public mo::IMetaObject
{
	MO_BEGIN(printable)
	MO_END;
    virtual void print();
};