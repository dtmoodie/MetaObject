#pragma once
#include "MetaObject/params/TInputParam.hpp"
#include "MetaObject/params/TParam.hpp"
#include "MetaObject/params/TParamPtr.hpp"
#include <MetaObject/params/detail/ParamMacrosImpl.hpp>

#define EXTERN_TYPE(TYPE)                                                                                              \
    template class mo::TInputParamPtr<mo::argument_type<void(TYPE)>::type>;                                            \
    template class mo::TParamPtr<mo::argument_type<void(TYPE)>::type>;                                                 \
    template class mo::TParam<mo::argument_type<void(TYPE)>::type>;                                                    \
    template class mo::TParamOutput<mo::argument_type<void(TYPE)>::type>

#define INSTANTIATE_META_PARAM(TYPE) mo::MetaParam<TYPE, __COUNTER__> COMBINE(g_meta_Param, __LINE__)(table, #TYPE);
