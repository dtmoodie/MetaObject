#pragma once
#include "MetaObject/Params/TInputParam.hpp"
#include "MetaObject/Params/TParamPtr.hpp"
#include "MetaObject/Params/Detail/TInputParamPtrImpl.hpp"
#include "MetaObject/Params/Detail/TParamPtrImpl.hpp"

#define INSTANTIATE_META_Param(TYPE) \
static mo::MetaParam<TYPE, __COUNTER__> COMBINE(g_meta_Param, __LINE__)(#TYPE); \
template class mo::TInputParamPtr<TYPE>; \
template class mo::TParamPtr<TYPE>;
