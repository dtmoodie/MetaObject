#pragma once
#include "MetaObject/Parameters/TypedParameterPtr.hpp"
#include "detail/ParameterMacrosImpl.hpp"

#define PARAM(type, name, init) \
mo::TypedParameterPtr<type> name##_param; \
type name = init; \
PARAM_(type, name, init, __COUNTER__)

#define RANGED_PARAM(type, name, init, min, max)

#define INPUT(type, name, init) \
type* name = init; \
mo::TypedInputParameterPtr<type> name##_param; \
void init_parameters_(bool firstInit, mo::_counter_<__COUNTER__> dummy) \
{ \
    name##_param.SetMtx(&_mtx); \
    name##_param.SetUserDataPtr(&name); \
    name##_param.SetName(#name); \
    AddParameter(&name##_param); \
    init_parameters_(firstInit, --dummy); \
}


#define PROPERTY(type, name, init) \
type name = init; \
SERIALIZE_(name, __COUNTER__)

#define STATUS(type, name, init)\
mo::TypedParameterPtr<type> name##_param; \
type name = init; \
STATUS_(type, name, init, __COUNTER__)

#define TOOLTIP(name, TOOLTIP) TOOLTIP_(name, TOOLTIP, __COUNTER__)

#define DESCRIPTION(name, DESCRIPTION)

#define OUTPUT(type, name, init) \
mo::TypedParameterPtr<type> name##_param; \
OUTPUT_(type, name, init, __COUNTER__); \
type name = init;