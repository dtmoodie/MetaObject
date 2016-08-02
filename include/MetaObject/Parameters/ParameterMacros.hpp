#pragma once
#include "detail/ParameterMacrosImpl.hpp"

#define PARAM(type, name, init) \
mo::TypedParameterPtr<type> name##_param; \
type name = init; \
PARAM_(type, name, init, __COUNTER__)

#define RANGED_PARAM(type, name, init, min, max)

#define INPUT_PARAM(type, name, init) \
type* name = init; \
mo::TypedInputParameterPtr<type> name##_param; \
INPUT_PARAM_(type, name, init, __COUNTER__)


#define PROPERTY(type, name, init) \
type name = init; \
SERIALIZE_(name, __COUNTER__)

#define STATUS(type, name, init)\
mo::TypedParameterPtr<type> name##_param; \
type name = init; \
STATUS_(type, name, init, __COUNTER__)

#define TOOLTIP(name, TOOLTIP) TOOLTIP_(name, TOOLTIP, __COUNTER__)