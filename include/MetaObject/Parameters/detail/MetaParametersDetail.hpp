#pragma once
#include "MetaObject/Parameters/TypedInputParameter.hpp"
#include "MetaObject/Parameters/TypedParameterPtr.hpp"
#include "MetaObject/Parameters/detail/TypedInputParameterPtrImpl.hpp"
#include "MetaObject/Parameters/detail/TypedParameterPtrImpl.hpp"

#define INSTANTIATE_META_PARAMETER(TYPE) \
static mo::MetaParameter<TYPE, __COUNTER__> COMBINE(g_meta_parameter, __LINE__)(#TYPE); \
template class mo::TypedInputParameterPtr<TYPE>; \
template class mo::TypedParameterPtr<TYPE>;
