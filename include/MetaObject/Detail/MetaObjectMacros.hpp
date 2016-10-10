#pragma once
#include "MetaObjectMacrosImpl.hpp"

#define MO_BEGIN(CLASS_NAME) MO_BEGIN_1(CLASS_NAME, __COUNTER__)
#define MO_DERIVE(CLASS_NAME, ...) MO_DERIVE_(__COUNTER__, CLASS_NAME, __VA_ARGS__)
#define MO_END MO_END_(__COUNTER__)




