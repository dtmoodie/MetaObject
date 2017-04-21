#pragma once
#include "MetaObject/Parameters/TypedInputParameter.hpp"
#include "MetaObject/Parameters/TypedParameterPtr.hpp"
#include "MetaObject/Parameters/detail/TypedInputParameterPtrImpl.hpp"
#include "MetaObject/Parameters/detail/TypedParameterPtrImpl.hpp"

#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__)
  #define TEMPLATE_EXTERN extern
#elif defined __GNUC__ && __GNUC__ >= 4
  #define TEMPLATE_EXTERN
#else
  #define TEMPLATE_EXTERN
#endif

#define INSTANTIATE_META_PARAMETER(TYPE) \
static mo::MetaParameter<TYPE, __COUNTER__> COMBINE(g_meta_parameter, __LINE__)(#TYPE); \
TEMPLATE_EXTERN template class mo::TypedInputParameterPtr<TYPE>; \
TEMPLATE_EXTERN template class mo::TypedParameterPtr<TYPE>;
