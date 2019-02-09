#pragma once
#include "MetaObject/object/MetaObject.hpp"
#include "MetaObject/object/detail/MetaObjectMacros.hpp"
#include "MetaObject/params/ParamMacros.hpp"

struct printable : public mo::MetaObject
{
    MO_BEGIN
    // TODO remove once we sort out structs with no fields
    PARAM(int, tmp, 0)
    MO_END;
    virtual void print();
};
