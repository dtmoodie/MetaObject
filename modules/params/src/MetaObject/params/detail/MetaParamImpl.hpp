#pragma once
#include "MetaObject/params/TInputParam.hpp"
#include "MetaObject/params/TParamPtr.hpp"
#include "MetaObject/params/detail/TInputParamPtrImpl.hpp"
#include "MetaObject/params/detail/TParamPtrImpl.hpp"

#define INSTANTIATE_META_Param(TYPE) \
static mo::MetaParam<TYPE, __COUNTER__> COMBINE(g_meta_Param, __LINE__)(#TYPE); \
template class mo::TInputParamPtr<TYPE>; \
template class mo::TParamPtr<TYPE>; \
template class mo::TParamOutput<TYPE>;
