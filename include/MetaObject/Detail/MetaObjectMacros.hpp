#pragma once
#include "MetaObjectMacrosImpl.hpp"

#define MO_BEGIN(CLASS_NAME) MO_BEGIN_1(CLASS_NAME, __COUNTER__)
#define MO_DERIVE(CLASS_NAME, BASE_NAME) MO_BEGIN_2(CLASS_NAME, BASE_NAME, __COUNTER__)
#define MO_END MO_END_(__COUNTER__)




